"""BiomniBench-DA's judge: the LLM picks an A/B/C level per criterion, never a number.

Python maps the chosen letter to that criterion's rubric-defined point value
(``Levels: A=X B=Y C=0``) and sums, which removes judge arithmetic noise. The
letter→points mapping lives in ``env/biomni_judge.py`` so the offline re-grader
(``scripts/biomni_judge.py``) shares it.
"""

from __future__ import annotations

from lmi import LiteLLMModel
from pydantic import BaseModel, Field

from ..biomni_judge import score_rich_levels
from ..prompts import RUBRIC_LEVEL_PROMPT, RUBRIC_LEVEL_PROMPT_QUESTION
from .base import JudgeContext, JudgeResult, StepEvidence, call_json, judge


class CriterionLevelScore(BaseModel):
    """Same rich shape as CriterionScore, but graded by A/B/C ``level`` instead of ``score``."""

    criterion: str
    level: str
    justification: str
    relevant_steps: list[StepEvidence] = Field(default_factory=list)
    # first_wrong_step disabled for now — see the note above _RUBRIC_SCORE_PROMPT_TAIL in prompts.py.
    # first_wrong_step: int | None = None
    feedback: str | None = None


class RubricLevelScore(BaseModel):
    criteria: list[CriterionLevelScore]


@judge("biomni")
async def biomni_judge(ctx: JudgeContext, model: LiteLLMModel | None) -> JudgeResult:
    """One call; the judge picks A/B/C per criterion and Python maps letters to points."""
    assert model is not None, f"{__name__} requires a rubric model"
    problem = ctx.problem
    if problem.task_style == "hypothesis":
        prompt = RUBRIC_LEVEL_PROMPT.format(
            hypothesis=problem.hypothesis,
            accepted=problem.accepted,
            rubric=problem.rubric,
            notebook=ctx.notebook,
            proposed_solution=ctx.solution,
        )
    else:
        prompt = RUBRIC_LEVEL_PROMPT_QUESTION.format(
            hypothesis=problem.hypothesis,
            rubric=problem.rubric,
            notebook=ctx.notebook,
            proposed_solution=ctx.solution,
        )

    parsed, meta = await call_json(model, prompt, RubricLevelScore)
    criteria = [c.model_dump() for c in parsed.criteria]
    # injects "score" into each criterion dict, and clamps the total to [0, 100]
    raw_score = score_rich_levels(criteria, problem.rubric)
    return JudgeResult(
        raw_score=raw_score,
        max_score=problem.max_score,
        criteria=criteria,
        metadata=meta,
    )
