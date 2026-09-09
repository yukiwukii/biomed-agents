"""BiomniBench-DA's original judging method (A/B/C level-picking), as a reusable core.

Faithful port of the judge shipped in the ``phylobio/BiomniBench-DA`` dataset
(``da-*/tests/llm_judge.py``). Its defining property: the LLM only *chooses a
level* (A / B / C) per rubric criterion — it never does arithmetic. Python maps
each chosen letter to that criterion's rubric-defined point value
(``Levels: A=X B=Y C=0``), sums to a 0-100 score, and clamps. This eliminates
judge arithmetic noise, unlike hypotest's default judge (which asks the model to
emit an integer score per criterion directly).

This module is pure (stdlib only — no LLM client, no I/O) so it can be shared by:
  - ``InterpreterEnv._score_solution`` — live grading, auto-selected for
    biomni-style rubrics (see ``is_biomni_rubric``).
  - ``scripts/eval/biomni_judge.py`` — offline re-grading of saved runs, and grading
    native BiomniBench-DA ``trace.md`` / ``answer.txt`` files.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def parse_rubric_levels(rubric_text: str) -> dict[str, dict[str, int]]:
    """Parse the rubric into ``{criterion_<N>: {"A": pts, "B": pts, "C": pts}}``.

    Verbatim from the original judge. Supports the current rubric format (single
    ``Levels: A=X B=Y C=0`` header per criterion) and the legacy per-line
    ``[A] (X points): ...`` format. Returns ``{}`` for rubrics that have neither
    (e.g. hypotest's own free-text rubrics), which is how ``is_biomni_rubric``
    distinguishes the two families.

    Level values may be **negative**: every BiomniBench-DA rubric ends with a penalty
    criterion declaring ``Levels: A=0 B=-5 C=-10``. The ``-?`` in each pattern is what
    makes those parse — without it the header regex stops at the first negative value
    (capturing only ``"A=0 "``), the penalty silently becomes 0, and
    ``score_rich_levels`` inflates the total by up to 10 points out of 100.
    """
    out: dict[str, dict[str, int]] = {}
    parts = re.split(r"^Criterion\s+(\d+)\s*:", rubric_text, flags=re.MULTILINE)
    for i in range(1, len(parts), 2):
        n = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        levels: dict[str, int] = {}
        m = re.search(r"Levels:\s*((?:[A-Z]=-?\d+\s*)+)", body)
        if m:
            for lm in re.finditer(r"([A-Z])=(-?\d+)", m.group(1)):
                levels[lm.group(1).upper()] = int(lm.group(2))
        if not levels:  # legacy fallback: "[A] (N points)"
            for lm in re.finditer(r"\[([A-Z])\]\s*\(\s*(-?\d+)\s*points?\s*\)", body):
                levels[lm.group(1).upper()] = int(lm.group(2))
        if levels:
            out[f"criterion_{n}"] = levels
    return out


def is_biomni_rubric(rubric_text: str) -> bool:
    """True if the rubric uses BiomniBench-DA's ``Criterion N:`` + ``Levels: A=..`` format."""
    return bool(parse_rubric_levels(rubric_text))


def build_judge_prompt(rubric: str, trace: str, answer: str) -> str:
    """The original judge prompt: pick ONE level (A/B/C) per criterion, no arithmetic."""
    return f"""You are an expert evaluator for a data analysis task.

Evaluate the agent's work using the following rubric:

{rubric}

Here is the agent's analysis trace:

<trace>
{trace or "[No trace file provided]"}
</trace>

Here is the agent's final answer:

<answer>
{answer or "[No answer file provided]"}
</answer>

For each criterion in the rubric, choose ONE level: A, B, or C — based purely on which level description best describes the agent's work. Do not output numerical points; the score for each level is computed automatically from the rubric.

You MUST respond with a JSON object in exactly this format:
{{
  "criteria": {{
    "criterion_1": {{"level": "A", "reason": "<one-sentence explanation>"}},
    "criterion_2": {{"level": "B", "reason": "<one-sentence explanation>"}},
    ...
  }},
  "overall_reasoning": "<short summary>"
}}

Each "level" value must be exactly the single character "A", "B", or "C". Only output the JSON object, nothing else."""


def _extract_json_object(text: str) -> dict:
    """Brace-match the first top-level JSON object in the response (as the original does)."""
    start = text.find("{")
    if start == -1:
        return json.loads(text)  # no object → let json raise for the caller to handle
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    return json.loads(text[start:])  # unbalanced — let json raise


def score_rich_levels(criteria: list[Any], rubric: str) -> int:
    """Map each rich criterion's A/B/C ``level`` → points via the rubric's ``Levels:`` table.

    For the biomni grading path that emits hypotest's *rich* rubric schema (a list of
    criteria carrying a ``level`` instead of a ``score`` — see
    ``InterpreterEnv._score_solution``). Criteria are matched to the rubric's
    ``Criterion N:`` blocks by position. Injects the computed ``"score"`` into each
    criterion dict in place and returns the summed total clamped to [0, 100], mirroring
    ``score_from_response``'s letter→points logic.

    Per-criterion points may be negative — BiomniBench-DA's trailing *Source Reliability*
    criterion is a pure penalty (``A=0 B=-5 C=-10``). A penalty therefore subtracts from
    the total, but the ``max(0, …)`` clamp keeps the reward from going below zero.
    """
    try:
        level_maps = list(parse_rubric_levels(rubric).values())
    except Exception as parse_err:
        logger.warning("failed to parse rubric levels: %s", parse_err)
        level_maps = []
    total = 0
    for i, c in enumerate(criteria):
        # criteria is parsed from model output, so a malformed entry is possible
        # even though the happy path is always a dict. Skip rather than raise.
        if not isinstance(c, dict):
            continue
        allowed = level_maps[i] if i < len(level_maps) else {}
        level = (c.get("level") or "").strip().upper()
        if level in allowed:
            pts = allowed[level]
        elif allowed:
            pts = min(allowed.values())  # unrecognized level → lowest defined value
        else:
            pts = 0
        c["score"] = pts
        total += pts
    return max(0, min(100, total))


def score_from_response(response_text: str, rubric: str) -> tuple[int, dict, str]:
    """Map judge levels → points via the rubric, sum, clamp to [0, 100].

    Returns ``(total_score, criteria_with_scores, reasoning)``. Mirrors the
    original ``llm_judge.py`` exactly, including its legacy numeric fallback and
    its "score 0 on unparseable output" behavior (it never raises — a malformed
    judge response yields 0 rather than an exception).
    """
    try:
        result = _extract_json_object(response_text)
        criteria = result.get("criteria", {})
        reasoning = result.get("overall_reasoning", result.get("reasoning", "No reasoning provided"))

        try:
            criterion_levels = parse_rubric_levels(rubric)
        except Exception as parse_err:
            logger.warning("failed to parse rubric levels: %s", parse_err)
            criterion_levels = {}

        for k, c in list(criteria.items()):
            if not isinstance(c, dict):
                continue
            allowed = criterion_levels.get(k) or {}
            level = (c.get("level") or "").strip().upper()
            if level in allowed:
                c["score"] = allowed[level]
            elif "score" in c:  # legacy: numeric score → snap to nearest allowed value
                try:
                    stated = int(c.get("score", 0))
                except (TypeError, ValueError):
                    stated = 0
                if allowed:
                    c["score"] = min(allowed.values(), key=lambda v: abs(v - stated))
            else:
                c["score"] = 0

        if criteria:
            total = 0
            for c in criteria.values():
                if isinstance(c, dict):
                    with contextlib.suppress(TypeError, ValueError):
                        total += int(c.get("score", 0))
        else:
            total = int(result.get("total_score", result.get("score", 0)))

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("failed to parse judge JSON: %s", e)
        # `-?` matters here too: against `"total_score": -5` a bare `(\d+)` matches the digits
        # *after* the minus sign and reads it as +5, rather than failing to match.
        m = re.search(r'"total_score"\s*:\s*(-?\d+)', response_text) or re.search(
            r'"score"\s*:\s*(-?\d+)', response_text
        )
        total = int(m.group(1)) if m else 0
        criteria = {}
        reasoning = f"Failed to parse full response: {e}"

    total = max(0, min(100, total))
    return total, criteria, reasoning
