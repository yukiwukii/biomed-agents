"""hypotest's default judge: the LLM awards integer points per rubric criterion.

For rubrics in hypotest's own ``N. (X points) …`` / ``* X points: …`` format. The
score is the plain sum of the per-criterion integers.
"""

from __future__ import annotations

from lmi import LiteLLMModel
from pydantic import BaseModel, Field

from ..prompts import RUBRIC_SCORE_PROMPT, RUBRIC_SCORE_PROMPT_QUESTION
from .base import JudgeContext, JudgeResult, StepEvidence, call_json, judge


class CriterionScore(BaseModel):
    criterion: str
    score: int
    justification: str
    relevant_steps: list[StepEvidence] = Field(default_factory=list)
    # first_wrong_step disabled for now — see the note above _RUBRIC_SCORE_PROMPT_TAIL in prompts.py.
    # first_wrong_step: int | None = None
    feedback: str | None = None


class RubricScore(BaseModel):
    criteria: list[CriterionScore]


@judge("hypotest")
async def hypotest_judge(ctx: JudgeContext, model: LiteLLMModel | None) -> JudgeResult:
    """One call; the judge emits integer points per criterion, summed in Python."""
    assert model is not None, f"{__name__} requires a rubric model"
    problem = ctx.problem
    if problem.task_style == "hypothesis":
        prompt = RUBRIC_SCORE_PROMPT.format(
            hypothesis=problem.hypothesis,
            accepted=problem.accepted,
            rubric=problem.rubric,
            notebook=ctx.notebook,
            proposed_solution=ctx.solution,
        )
    else:
        prompt = RUBRIC_SCORE_PROMPT_QUESTION.format(
            hypothesis=problem.hypothesis,
            rubric=problem.rubric,
            notebook=ctx.notebook,
            proposed_solution=ctx.solution,
        )

    parsed, meta = await call_json(model, prompt, RubricScore)
    criteria = [c.model_dump() for c in parsed.criteria]
    return JudgeResult(
        raw_score=sum(c.score for c in parsed.criteria),
        max_score=problem.max_score,
        criteria=criteria,
        metadata=meta,
    )
