"""Per-benchmark judges, one function each — LLM-driven or deterministic. See ``base.py``.

Importing a judge module here is what registers it — the modules are imported for
their side effect, so a new benchmark is one new file plus one line below.
"""

from . import bioagent, biomni, biomystery, bixbench, heureka, hypotest_judge  # noqa: F401  (imported to register)
from .base import (
    JUDGES,
    Judge,
    JudgeContext,
    JudgeFn,
    JudgeResult,
    StepEvidence,
    call_json,
    derive_first_wrong_step,
    judge,
    parse_criterion_max_scores,
    resolve_judge,
)

__all__ = [
    "JUDGES",
    "Judge",
    "JudgeContext",
    "JudgeFn",
    "JudgeResult",
    "StepEvidence",
    "call_json",
    "derive_first_wrong_step",
    "judge",
    "parse_criterion_max_scores",
    "resolve_judge",
]
