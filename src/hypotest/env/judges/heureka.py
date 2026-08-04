"""HeurekaBench's open-ended judge: atomic-fact labelling.

Upstream (``scheurekabench/geval_prompts/eval_prompts.py``) scores an open-ended
sub-question by decomposing its ground-truth answer into atomic facts, labelling each
against the agent's answer, and reading a 1-5 correctness rating off that label pattern.

It is implemented here the way ``biomni.py`` implements BiomniBench's judge: **the LLM
only labels, Python does the arithmetic.** The model emits one fact list per sub-question
with a PRESENT/PARTIAL/MISSING/INCORRECT label each, and :func:`_band` turns those labels
into points. The labels survive into ``score_info.json``, so a rating can be audited fact
by fact.

Note this deviates from the rating a single-call G-Eval judge would emit: bands are now
mechanical, so the same labels always give the same number. It also deviates from upstream
in the two ways ``scripts/convert_heurekabench.py`` already documents — the judge sees the
whole notebook, and reward is normalized to [0, 1].

Only the open-ended split is covered. HeurekaBench's MCQ split has no judge here; MCQ
problems fall back to whatever ``resolve_judge`` picks, i.e. hypotest's integer judge
reading ``convert_heurekabench.MCQ_RUBRIC_PREAMBLE``.
"""

from __future__ import annotations

from typing import Literal

from lmi import LiteLLMModel
from pydantic import BaseModel, Field

from .base import JudgeContext, JudgeResult, StepEvidence, call_json, judge

FactStatus = Literal["PRESENT", "PARTIAL", "MISSING", "INCORRECT"]

# Shared tail: how to report evidence from the notebook. Mirrors the contract in
# prompts._RUBRIC_SCORE_PROMPT_TAIL so the fork/inspect tooling reads these the same way.
_EVIDENCE_CONTRACT = """
For every sub-question also report:
  - "relevant_steps": every notebook cell (labelled "### Cell N:" above) bearing on this sub-question, in
    cell order — those that supported the answer and those that did not; omit unrelated cells. Never leave
    this empty. Omit environment and tooling cells entirely, however they turn out: package installation and
    dependency resolution (pip, conda, install.packages, BiocManager), failed or retried installs,
    missing-package and import errors, version conflicts, and kernel restarts. These are setup noise, not
    scientific work. Each entry has "step" (the cell index), "note" (one sentence on what happened there),
    and "correct" (whether it holds up in the notebook's end state, not in isolation — true if a later cell
    fixed it or the agent reached the answer another way; false only for problems still standing at the end).
  - "justification": one or two sentences summarizing the sub-question's outcome.
  - "feedback": forward-looking guidance for a fresh policy model resuming from the earliest
    "correct": false step — what to do at that juncture; null when nothing went wrong. Phrase it purely as
    the correct action, never as a critique. Target the earliest foundational step that went wrong (loading,
    normalization, QC, cohort setup), name a concrete method, and end with a verification clause. Never leak
    or prescribe the expected result or conclusion.
""".strip()

HEUREKA_OE_PROMPT = """
Your task is to evaluate an agent's answers to a set of single-cell research sub-questions, against
ground-truth answers derived from the paper the data comes from.

The agent was given this task:
<question>
{hypothesis}
</question>

Here is the rubric. Each numbered criterion is one sub-question with its ground-truth (GT) answer:
<rubric>
{rubric}
</rubric>

Here is the agent's notebook:
<notebook>
{notebook}
</notebook>

The final answer derived from the notebook:
<proposed-solution>
{proposed_solution}
</proposed-solution>

For each criterion, in rubric order, decompose that criterion's GT answer into atomic facts — one fact per
distinct claim (cell type/condition, direction/magnitude of change, gene/pathway name, statistical evidence,
method, conclusion). Split thoroughly; a GT answer naming three cell populations and their direction of
change carries at least three facts. Then label each fact against the agent's answer:

  - PRESENT:   same meaning AND tied to dataset-derived quantitative/statistical output or dataset cluster
               and subtype identifiers (percentages, fold changes, p-values, cluster IDs, enrichment scores).
  - PARTIAL:   correct meaning but supported only by descriptive biology, lists of plausible options, hedged
               language ("likely", "typically", "e.g."), or general biological recall rather than this
               dataset's evidence.
  - MISSING:   not mentioned.
  - INCORRECT: wrong, or contradicts a GT fact.

Set "answered" to false only if the agent produced no answer at all to that sub-question.

Do not output points or ratings of any kind. The rating for each sub-question is computed programmatically
from the labels you assign, so label carefully and label every fact. Grade only against the GT answer;
ignore outside knowledge. Extra information beyond the GT neither helps nor hurts unless it contradicts the
GT. Style, length, and paraphrasing do not matter. All GT facts within a criterion are weighted equally.

{evidence}
""".strip()


class FactLabel(BaseModel):
    fact: str
    label: FactStatus
    note: str


class HeurekaOECriterion(BaseModel):
    criterion: str
    answered: bool
    facts: list[FactLabel] = Field(default_factory=list)
    justification: str
    relevant_steps: list[StepEvidence] = Field(default_factory=list)
    feedback: str | None = None


class HeurekaOEScore(BaseModel):
    criteria: list[HeurekaOECriterion]


def _band(labels: list[str], answered: bool) -> int:
    """HeurekaBench's 0-5 G-Eval correctness scale, computed from atomic-fact labels.

    Transcribed from the scale in ``convert_heurekabench.OE_RUBRIC_PREAMBLE``: 5 is a fully
    dataset-grounded answer, 4 allows omissions but no errors, 3 is partial, 2 is generic
    biological recall, 1 is absent/wrong, 0 is no answer at all. "Most" is read as a
    majority of the criterion's facts.
    """
    if not answered:
        return 0
    n = len(labels)
    if n == 0:
        return 1
    present = labels.count("PRESENT")
    partial = labels.count("PARTIAL")
    incorrect = labels.count("INCORRECT")

    if incorrect == 0 and present == n:
        return 5
    if incorrect == 0 and present > 0 and 2 * present >= n:
        return 4
    if 2 * incorrect > n:  # most facts wrong / major contradictions of GT
        return 1
    if present > 0:
        return 3
    if partial > 0:
        return 2
    return 1


@judge("heureka")
async def heureka_judge(ctx: JudgeContext, model: LiteLLMModel | None) -> JudgeResult:
    """Open-ended: the LLM labels atomic facts, Python reads the 0-5 band off the labels."""
    assert model is not None, f"{__name__} requires a rubric model"
    prompt = HEUREKA_OE_PROMPT.format(
        hypothesis=ctx.problem.hypothesis,
        rubric=ctx.problem.rubric,
        notebook=ctx.notebook,
        proposed_solution=ctx.solution,
        evidence=_EVIDENCE_CONTRACT,
    )
    parsed, meta = await call_json(model, prompt, HeurekaOEScore)

    criteria: list[dict] = []
    total = 0
    for c in parsed.criteria:
        score = _band([f.label for f in c.facts], c.answered)
        total += score
        criteria.append({**c.model_dump(), "score": score})

    return JudgeResult(
        raw_score=total,
        max_score=ctx.problem.max_score,
        criteria=criteria,
        metadata=meta,
    )
