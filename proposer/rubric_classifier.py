"""Shared LLM-based classifier: sorts grading-rubric criteria into six axes.

A criterion can span more than one axis (e.g. a criterion that checks both data
loading AND statistical testing), so classification is multi-label: each
criterion gets one or more categories, not exactly one.

Used by ``classify_rubric_criteria.py`` (the generated proposer rubrics) and
``classify_original_rubrics.py`` (the HF dataset's ground-truth rubrics). One
model call classifies every criterion belonging to a single rubric at once —
that keeps the criteria in their natural context (the hypothesis + siblings)
and is far cheaper than one call per criterion.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Literal

from lmi import LiteLLMModel
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "claude-sonnet-4-6"

Category = Literal[
    "data_handling",
    "method_selection",
    "statistical_rigor",
    "biological_interpretation",
    "scientific_reasoning",
    "source_reliability",
]

CATEGORY_DESCRIPTIONS: dict[Category, str] = {
    "data_handling": "Quality of data loading, cleaning, preprocessing, transformation.",
    "method_selection": "Appropriateness of analytical methods and models.",
    "statistical_rigor": "Correctness and completeness of statistical analysis.",
    "biological_interpretation": "Accuracy and relevance of biological conclusions.",
    "scientific_reasoning": "Coherence, logic, and support of the reasoning chain.",
    "source_reliability": "Use of credible sources and appropriate citations.",
}


def load_env(path: Path = ROOT / ".env") -> None:
    """Populate os.environ from a KEY=VALUE .env file (does not overwrite)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


class ClassifiedCriterion(BaseModel):
    index: int
    categories: list[Category]


class RubricClassification(BaseModel):
    classifications: list[ClassifiedCriterion]


def build_classify_prompt(hypothesis: str, criteria: list[str]) -> str:
    cats = "\n".join(f"- {name}: {desc}" for name, desc in CATEGORY_DESCRIPTIONS.items())
    items = "\n".join(f"[{i}] {c}" for i, c in enumerate(criteria))
    return f"""You are classifying items from a scientific grading rubric used to score an
agent's data-analysis notebook against a hypothesis. Each item is one grading
criterion (a specific, checkable thing the grader looks for in the agent's work).

Classify EACH criterion below into ONE OR MORE of these six categories — a criterion
often checks more than one aspect at once (e.g. both loading/preprocessing the data AND
running a statistical test), so assign every category that genuinely applies, not just
the single best match:

{cats}

Hypothesis being graded: {hypothesis}

Criteria to classify (index them exactly as given):
{items}

Return one entry per criterion, each with a non-empty list of one or more categories,
covering every index from 0 to {len(criteria) - 1} exactly once, in any order."""


def _first_json_object(text: str) -> dict:
    """Decode the first complete JSON object in ``text``, ignoring trailing data.

    Models sometimes wrap the object in a ```` ```json ```` fence or append prose after the
    closing brace; ``raw_decode`` stops at the end of the first value so that trailing content
    (which ``json.loads`` would reject as "Extra data") is harmless.
    """
    obj, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
    return obj


async def classify_rubric(
    model: LiteLLMModel, hypothesis: str, criteria: list[str], retries: int = 3
) -> tuple[list[list[Category]], str | None]:
    """Classify all criteria of one rubric in a single call.

    Returns ``(categories, reasoning)`` where ``categories`` is a list of category-lists
    aligned to input order (each criterion may map to more than one category), and
    ``reasoning`` is the model's thinking content for this call (``None`` when the model
    returned no reasoning, e.g. thinking disabled).
    """
    if not criteria:
        return [], None
    prompt = build_classify_prompt(hypothesis, criteria)
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            resp = await model.call_single(prompt, output_type=RubricClassification, timeout=3 * 60)
            if not resp.text:
                raise ValueError("empty response from model")
            parsed = RubricClassification.model_validate(_first_json_object(resp.text))
            by_index = {c.index: c.categories for c in parsed.classifications}
            if set(by_index) != set(range(len(criteria))):
                raise ValueError(f"index mismatch: got {sorted(by_index)}, expected 0..{len(criteria) - 1}")
            if any(not cats for cats in by_index.values()):
                raise ValueError("at least one criterion was returned with an empty category list")
            return [by_index[i] for i in range(len(criteria))], resp.reasoning_content
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_err = e
    raise last_err  # type: ignore[misc]


async def classify_many(
    model: LiteLLMModel,
    jobs: list[tuple[str, list[str]]],
    concurrency: int = 8,
) -> list[tuple[list[list[Category]], str | None]]:
    """Classify multiple rubrics concurrently. ``jobs`` is a list of (hypothesis, criteria) pairs.

    Each result is the ``(categories, reasoning)`` tuple from ``classify_rubric``, aligned to
    ``jobs`` order.
    """
    sem = asyncio.Semaphore(concurrency)

    async def run(hyp: str, crit: list[str]) -> tuple[list[list[Category]], str | None]:
        async with sem:
            return await classify_rubric(model, hyp, crit)

    return await asyncio.gather(*(run(hyp, crit) for hyp, crit in jobs))
