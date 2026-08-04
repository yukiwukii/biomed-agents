"""``ProblemInstance``: the dataset-side definition of one task.

Split out of ``interpreter_env.py`` so the judges package and the offline grading
scripts can import it without pulling in that module's docker/jupyter import side
effects. ``interpreter_env`` re-exports it, so existing imports keep working.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, JsonValue, model_validator


class ProblemInstance(BaseModel):
    id: UUID
    hypothesis: str
    protocol: str
    # Optional: only meaningful for task_style="hypothesis" (see below), where it is
    # the known accept/reject ground truth. Left unset for task_style="question".
    accepted: bool | None = Field(default=None, alias="answer")
    rubric: str
    max_score: int = Field(alias="max_points")
    input_data_path: str = ""
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    # str(NBLanguage.PYTHON); spelled literally to keep this module free of the
    # kernel_server import (jupyter).
    nb_primary_language: str = Field(default="python")
    # "hypothesis" (default, existing behavior): the agent is asked to substantiate or
    # reject `hypothesis`, and grading is told the known accept/reject outcome.
    # "question": the agent is asked to answer `hypothesis` as an open research
    # question; grading is rubric-only with no accept/reject boolean.
    task_style: Literal["hypothesis", "question"] = "hypothesis"
    # Which registered judge grades this task (see env/judges/). Set by the dataset
    # converter — it is a property of the benchmark, not of the run. None/"auto"
    # falls back to the env config, then to rubric-format sniffing, so datasets
    # written before this field existed grade exactly as they did before.
    judge: str | None = None

    @model_validator(mode="before")
    @classmethod
    def handle_language(cls, data: dict) -> dict:
        if data.get("nb_primary_language") is None:
            data["nb_primary_language"] = "python"
        return data

    @model_validator(mode="after")
    def check_accepted_required_for_hypothesis(self) -> "ProblemInstance":
        if self.task_style == "hypothesis" and self.accepted is None:
            raise ValueError("accepted (answer) is required when task_style='hypothesis'")
        return self
