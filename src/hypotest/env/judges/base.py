"""Judge registry: one function per benchmark.

A judge is a single async function that takes the graded material and a model,
and returns a :class:`JudgeResult`. It owns everything benchmark-specific — the
prompt, the response schema, how many LLM calls it makes (including none — see
``bioagent.py``, a deterministic file scorer), and how labels/levels become
points. Everything downstream of it (first_wrong_step derivation, reward
normalization, ``score_info.json``) is shared and lives in ``InterpreterEnv``.

Adding a benchmark means adding a module next to this one::

    @judge("mybench")
    async def mybench_judge(ctx: JudgeContext, model: LiteLLMModel) -> JudgeResult:
        parsed, meta = await call_json(model, MY_PROMPT.format(...), MySchema)
        return JudgeResult(raw_score=..., max_score=..., criteria=..., metadata=meta)

...importing it from ``judges/__init__.py`` so the decorator runs, and setting
``judge: "mybench"`` on the problems the converter emits. No existing file needs
an edit, and there is no dispatch table to extend.

This package deliberately depends only on pydantic/lmi and the pure-python rubric
parsers, never on ``interpreter_env`` — so offline re-graders (``scripts/eval/regrade.py``,
``scripts/eval/biomni_judge.py``) can import the real judges instead of re-implementing
them.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

from lmi import LiteLLMModel
from pydantic import BaseModel, Field

from ..biomni_judge import parse_rubric_levels
from ..problem import ProblemInstance

logger = logging.getLogger(__name__)

TSchema = TypeVar("TSchema", bound=BaseModel)


@dataclass
class JudgeContext:
    """Everything a judge is allowed to look at.

    LLM judges read ``notebook``/``solution``; deterministic scorers read files under
    ``work_dir`` and compare them against ``truth_dir``.
    """

    problem: ProblemInstance
    notebook: str  # the rollout's notebook, already rendered by view_notebook
    solution: str  # the agent's submitted final answer
    # The rollout's workspace, holding whatever files the agent produced. Scored before
    # InterpreterEnv.close() moves or deletes it, so the outputs are still present.
    work_dir: Path | None = None
    # The answer key for this problem — ``<capsule_dir>/_truth/<input_data_path>/`` when the
    # converter staged one. Never copied into work_dir, so the agent cannot read it.
    truth_dir: Path | None = None


class JudgeResult(BaseModel):
    """What every judge returns, regardless of its internal protocol."""

    raw_score: int
    max_score: int
    # Per-criterion detail, as plain dicts, for score_info.json. Judges should include
    # "criterion", "score", "justification" and (where meaningful) "relevant_steps" so
    # the trajectory/fork tooling keeps working across benchmarks.
    criteria: list[dict[str, Any]] = Field(default_factory=list)
    # Prompt(s), raw response(s) and reasoning traces — merged into score_metadata.
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Judge-specific notion of "fully correct". None means the default: full marks.
    correct: bool | None = None

    def is_correct(self) -> bool:
        return self.correct if self.correct is not None else self.raw_score >= self.max_score


# The model is optional: a deterministic scorer (see bioagent.py) ignores it entirely and is
# registered with needs_model=False, which lets it run with no rubric_model configured at all.
JudgeFn = Callable[[JudgeContext, "LiteLLMModel | None"], Awaitable[JudgeResult]]


class Judge(NamedTuple):
    """A registered judge: its name, its function, and whether it needs an LLM."""

    name: str
    fn: JudgeFn
    needs_model: bool = True


JUDGES: dict[str, Judge] = {}


def judge(name: str, needs_model: bool = True) -> Callable[[JudgeFn], JudgeFn]:
    """Register a judge function under ``name``.

    Pass ``needs_model=False`` for judges that score without an LLM; scoring then runs even
    when the environment has no ``rubric_model``.
    """

    def deco(fn: JudgeFn) -> JudgeFn:
        if name in JUDGES:
            raise ValueError(f"judge {name!r} is already registered by {JUDGES[name].fn.__module__}")
        JUDGES[name] = Judge(name=name, fn=fn, needs_model=needs_model)
        return fn

    return deco


async def call_json(
    model: LiteLLMModel, prompt: str, schema: type[TSchema], timeout: float = 3 * 60, key: str = ""
) -> tuple[TSchema, dict[str, Any]]:
    """One structured LLM call: send ``prompt``, parse the response as ``schema``.

    Returns the parsed model plus a metadata dict carrying the rendered prompt, the raw
    response text, any reasoning tokens, and any chain-of-thought emitted before the JSON.
    ``key`` prefixes those metadata keys, for judges that make more than one call (the
    unprefixed names are what ``score_info.json`` and ``scripts/eval/regrade.py`` expect).
    """
    resp = await model.call_single(prompt, output_type=schema, timeout=timeout)
    if not resp.text:
        raise ValueError("No response from rubric model")

    meta: dict[str, Any] = {f"{key}prompt": prompt, f"{key}response": resp.text}
    if resp.reasoning_content:
        meta[f"{key}reasoning"] = resp.reasoning_content

    try:
        start = resp.text.index("{")
        end = resp.text.rindex("}") + 1
    except ValueError as e:
        raise ValueError("Failed to parse score from response") from e
    if start > 0:
        meta[f"{key}chain_of_thought"] = resp.text[:start].strip()

    try:
        parsed = schema.model_validate_json(resp.text[start:end])
    except Exception as e:
        raise ValueError("Failed to parse score from response") from e
    return parsed, meta


def resolve_judge(problem: ProblemInstance, config: Any = None) -> Judge:
    """Pick the judge for ``problem``: problem field, then env config, then rubric sniffing.

    The sniffing fallback exists only for datasets written before ``ProblemInstance.judge``:
    a rubric in BiomniBench-DA's ``Criterion N:`` + ``Levels: A=X B=Y C=0`` shape gets the
    A/B/C judge, everything else gets hypotest's integer-per-criterion one — exactly the
    behavior of the old ``biomni_grading="auto"``. New benchmarks set ``judge`` and never
    touch this function.
    """
    name = problem.judge or getattr(config, "judge", None) or "auto"
    if name == "auto":
        name = "biomni" if parse_rubric_levels(problem.rubric) else "hypotest"
    try:
        return JUDGES[name]
    except KeyError:
        raise ValueError(f"unknown judge {name!r}; registered: {sorted(JUDGES)}") from None


# ── shared post-processing (rubric-format aware, judge-agnostic) ──────────────

# Per-criterion point value in hypotest's own rubrics. Two layouts occur in the wild:
# numbered — "1. (1 point) Loads both datasets…" — and bixbench's bullets — "* 1 point: Loads…"
# (also "- 5 points: …"). Both alternatives capture into the same group index via `|`.
_HYPOTEST_CRITERION_POINTS = re.compile(
    r"^\s*(?:\d+\.\s*\(\s*(\d+)\s*points?\s*\)|[-*]\s*(\d+)\s*points?\s*:)", re.MULTILINE
)


def parse_criterion_max_scores(rubric: str) -> list[int]:
    """Per-criterion maximum points, in rubric order; ``[]`` if the rubric isn't parseable.

    Handles both rubric families: BiomniBench-DA's ``Criterion N:`` + ``Levels: A=X B=Y C=0``
    (max is the highest level value) and hypotest's ``N. (X points) …`` / ``* X points: …``.
    Criteria are matched to judge output by position, the same way
    ``biomni_judge.score_rich_levels`` does.
    """
    levels = parse_rubric_levels(rubric)
    if levels:
        return [max(v.values()) for v in levels.values() if v]
    return [int(m.group(1) or m.group(2)) for m in _HYPOTEST_CRITERION_POINTS.finditer(rubric)]


def derive_first_wrong_step(criteria: list[dict], rubric: str | None = None) -> list[dict]:
    """Inject ``"first_wrong_step"`` into each criterion dict, computed in Python.

    The judges no longer emit this field (see the note above ``_RUBRIC_SCORE_PROMPT_TAIL``
    in prompts.py); it is derived here as the earliest ``relevant_steps`` entry with
    ``"correct": false``, or ``None`` when every relevant step was correct. Note the
    prompt now tells the judge to mark a step correct if its error was repaired by a
    later cell, so this only ever points at a problem that survives to the end of the
    notebook. Downstream consumers (``scripts/fork/fork_trajectory.py``,
    ``scripts/eval/inspect_trajectory.py``) read the key from score_info.json unchanged.

    When ``rubric`` is given, a criterion awarded full marks gets ``None`` regardless of
    its relevant_steps — restoring the "null if this criterion received full marks" rule
    the judge-emitted field used to carry, so a fully-satisfied criterion never drives a
    fork. Without a parseable rubric the gate is skipped and every criterion falls back to
    the earliest-incorrect-step rule.

    Annotates in place and returns the same list, so it can wrap a criteria expression.
    """
    try:
        max_scores = parse_criterion_max_scores(rubric) if rubric else []
    except Exception as parse_err:
        logger.warning("failed to parse criterion max scores: %s", parse_err)
        max_scores = []

    for i, c in enumerate(criteria):
        max_pts = max_scores[i] if i < len(max_scores) else None
        score = c.get("score")
        if max_pts is not None and isinstance(score, (int, float)) and not isinstance(score, bool) and score >= max_pts:
            c["first_wrong_step"] = None
            continue
        wrong = [
            s["step"]
            for s in c.get("relevant_steps") or []
            if isinstance(s, dict) and not s.get("correct", True) and isinstance(s.get("step"), int)
        ]
        c["first_wrong_step"] = min(wrong) if wrong else None
    return criteria


class StepEvidence(BaseModel):
    """One notebook cell bearing on a criterion. Shared by every judge's schema."""

    step: int
    note: str
    correct: bool
