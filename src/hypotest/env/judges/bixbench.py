"""BixBench's judge: open-answer grading of the final answer against the question's ``ideal``.

futurehouse/BixBench v1.5 flattened the benchmark to one question per row and made open-answer
the preferred setting (the MCQ distractors are still shipped, but as known-wrong alternatives
rather than as options presented to the agent). Each question carries the answer key in three
fields:

  - ``ideal``       the correct answer, as a short string ("0.0002", "166", "10.6%", "(1.50,1.54)")
  - ``distractors`` three plausible but incorrect alternatives
  - ``eval_mode``   how upstream compares a submission to ``ideal``:
                      * ``str_verifier``   — the answer must be that value
                      * ``range_verifier`` — ``ideal`` is a literal interval ``(lo,hi)`` and the
                                             answer must fall inside it
                      * ``llm_verifier``   — free-form; a model decides equivalence

Two properties define this benchmark, and this module is built around them:

1. **The final answer is graded, not the notebook.** BixBench scores whether the agent arrived at
   the right value, not how. So nothing here is shown the notebook — same structural choice as
   ``biomystery.py``, and the reason ``judge: hypotest`` (which grades the procedure) must not be
   forced onto these problems if the number is to stay comparable to upstream.
2. **Scoring is binary.** One question, one point.

The split of labour follows ``biomni.py``/``bioagent.py``: the LLM labels, Python does the
arithmetic. Call 1 only *extracts* the value the agent committed to — it is deliberately not shown
``ideal``, so it cannot be steered into reading the key back out of a hedged answer. Python then
applies the verifier. A second, semantic call happens only when the comparison cannot be settled
numerically (``llm_verifier``, non-numeric ideals like "1-50", or an unparseable extraction), so
the common case costs a single call.

Upstream has not published its grader prompt, so the prompts below are written to the dataset's own
stated criteria rather than copied from an implementation. The two documented leniencies —
``STR_REL_TOL`` and percent rescaling — are stated where they are applied.
"""

from __future__ import annotations

import re

from lmi import LiteLLMModel
from pydantic import BaseModel

from .base import JudgeContext, JudgeResult, call_json, judge

# ``str_verifier`` answers are values the question asks to be reported at a stated precision
# ("rounded to 4 decimal points"), so comparing the parsed numbers with a small relative tolerance
# is closer to the intent than string equality. 1% is far tighter than the gap to the nearest
# distractor on every question in v1.5, so it cannot turn a distractor into a pass.
STR_REL_TOL = 0.01

# ``ideal`` for range_verifier, e.g. "(1.50,1.54)", "(0.6 , 0.7)", "(20,25)".
_INTERVAL_RE = re.compile(r"^\s*[(\[]\s*([^,\s]+)\s*,\s*([^,\s)\]]+)\s*[)\]]\s*$")

# First numeric token in a string: optional sign, digits with optional decimal part and exponent.
# Thousands separators and a trailing percent sign are handled by the caller.
_NUMBER_RE = re.compile(r"[-+−]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_number(text: str) -> tuple[float, bool] | None:
    """Parse the first number out of ``text``.

    Returns ``(value, is_percent)`` where ``is_percent`` records that the number was written with a
    ``%`` sign, or None if there is no number. ``value`` is the number as written — the caller
    decides whether the percent form should also be considered as ``value / 100``.
    """
    cleaned = text.replace(",", "").replace("−", "-")
    m = _NUMBER_RE.search(cleaned)
    if m is None:
        return None
    try:
        value = float(m.group())
    except ValueError:  # pragma: no cover - the regex only matches parseable literals
        return None
    return value, "%" in cleaned[m.end() : m.end() + 2]


def candidate_values(text: str) -> list[float]:
    """The numeric readings of ``text``, most literal first.

    A percentage gets a second reading as its fraction ("65%" -> 65.0 and 0.65), because whether a
    question's ideal is written as a percentage or a fraction varies across the benchmark and the
    agent's phrasing need not match it.
    """
    parsed = parse_number(text)
    if parsed is None:
        return []
    value, is_percent = parsed
    return [value, value / 100] if is_percent else [value]


def ideal_values(ideal: str) -> list[float]:
    """``candidate_values``, but only when the whole ideal *is* a number.

    ``parse_number`` deliberately picks the first number out of free text, which is right for an
    agent's answer but wrong for an answer key: "1-50" and "chr7:117,559,590" would each reduce to
    a single number that means nothing. An ideal that is not a bare numeric literal is a semantic
    comparison, so this returns [] and the caller defers to the model.
    """
    stripped = ideal.strip().removesuffix("%").replace(",", "").replace("−", "-").strip()
    if _NUMBER_RE.fullmatch(stripped) is None:
        return []
    return candidate_values(ideal)


def parse_interval(ideal: str) -> tuple[float, float] | None:
    """Parse a range_verifier ``ideal`` such as ``"(1.50,1.54)"`` into ``(lo, hi)``."""
    m = _INTERVAL_RE.match(ideal)
    if m is None:
        return None
    bounds = [parse_number(part) for part in m.groups()]
    if any(b is None for b in bounds):
        return None
    lo, hi = (b[0] for b in bounds if b is not None)
    return (lo, hi) if lo <= hi else (hi, lo)


def normalize(text: str) -> str:
    """Casefold and strip punctuation/whitespace, for the cheap exact-match path."""
    return re.sub(r"[\s,_'\"]+", "", text).strip(".").casefold()


class Verdict(BaseModel):
    """A settled comparison: the outcome plus how it was reached, for score_info.json."""

    correct: bool
    method: str
    detail: str


def verify_deterministically(mode: str, ideal: str, extracted: str) -> Verdict | None:
    """Apply the eval_mode's verifier in Python. None means "ask the model instead"."""
    if not extracted.strip():
        return Verdict(correct=False, method="no_answer", detail="the agent committed to no answer")

    if normalize(extracted) == normalize(ideal):
        return Verdict(correct=True, method="exact", detail=f"{extracted!r} matches ideal {ideal!r}")

    if mode == "range_verifier":
        interval = parse_interval(ideal)
        values = candidate_values(extracted)
        if interval is None or not values:
            return None  # malformed ideal, or nothing numeric to test — fall through to the model
        lo, hi = interval
        hit = next((v for v in values if lo <= v <= hi), None)
        return Verdict(
            correct=hit is not None,
            method="range",
            detail=f"{values[0]!r} {'in' if hit is not None else 'not in'} [{lo}, {hi}]",
        )

    if mode == "str_verifier":
        # A non-numeric ideal ("1-50", a gene name) is a semantic comparison, not an arithmetic one.
        targets = ideal_values(ideal)
        values = candidate_values(extracted)
        if not targets or not values:
            return None
        # Cross-compare readings so a percent written on either side matches its fraction on the
        # other ("10.6%" vs "0.106"), which the questions do not standardise.
        hit = any(abs(v - t) <= abs(t) * STR_REL_TOL for t in targets for v in values)
        return Verdict(
            correct=hit,
            method="numeric",
            detail=f"{values[0]!r} vs ideal {targets[0]!r} (tolerance {STR_REL_TOL:.0%})",
        )

    return None  # llm_verifier, or an eval_mode this dataset version did not have


EXTRACT_PROMPT = """
You are reading one agent's final answer to a bioinformatics question and pulling out the value it
committed to. You are NOT judging whether that value is right, and you are deliberately not being
shown the correct answer.

Here is the question the agent was asked:
<question>
{question}
</question>

Here is the agent's final answer, in full:
<final-answer>
{answer}
</final-answer>

Report the single value that answers the question, exactly as the agent gave it — the number with
its units or percent sign, the gene name, the category, whichever the question asked for. Rules:

  - Copy the agent's value; do not round it, convert it, or tidy it up.
  - If the answer works through several intermediate numbers, take the one the agent presents as
    the answer to this question, not the last number that happens to appear.
  - If the agent hedges between several candidates without committing, or gives a range where a
    single value was asked for, report exactly what it said rather than choosing for it.
  - If the agent gave no answer, declined, or only described how it would find one, return "".

Respond with a JSON object with these keys:
  - "extracted_answer": string — the value, or "" if there is none
  - "committed": boolean — false if the agent hedged, declined, or gave no answer
  - "justification": one sentence saying where in the answer you took it from
""".strip()

EQUIVALENCE_PROMPT = """
You are grading one answer to a bioinformatics question against the question's official answer key.

Here is the question the agent was asked:
<question>
{question}
</question>

Here is the correct answer:
<ideal>
{ideal}
</ideal>
{distractors}
Here is the agent's final answer, in full:
<final-answer>
{answer}
</final-answer>

This is the value the agent committed to, as extracted from that answer: {extracted}

Decide one thing only: does the agent's answer give the same result as the ideal answer?

  - Judge the answer itself. You are deliberately not being shown the agent's analysis, because
    this benchmark grades the final answer rather than the path taken to reach it. Do not speculate
    about how the answer was obtained, and do not reward apparent effort.
  - Formatting, wording and units-that-mean-the-same do not matter; the value does. Equivalent
    forms of the same number are correct (0.0002 = 2e-4, 10.6% = 0.106), and a synonym or an
    equivalent identifier for exactly the thing the ideal names is correct.
  - Where the ideal is stated to a given precision, an answer that rounds to it is correct; an
    answer that differs in the value itself is not. Do not widen the ideal with alternatives of
    your own, and do not narrow it either.
  - An answer matching one of the incorrect alternatives above is incorrect, however well argued.
  - An answer that hedges between several candidates, or that states the correct value only as one
    possibility among others, is incorrect. The agent must commit.
  - If the agent gave no answer, or declined, the answer is incorrect.

Respond with a JSON object with these keys:
  - "answer_correct": boolean — whether the agent's answer matches the ideal
  - "justification": one or two sentences comparing the agent's value to the ideal

Do not output a score. The score is computed programmatically from your boolean.
""".strip()

DISTRACTOR_BLOCK = """
These are plausible but INCORRECT answers to the same question. An answer equivalent to any of them
is wrong:
<incorrect-alternatives>
{items}
</incorrect-alternatives>
"""


class BixbenchExtraction(BaseModel):
    extracted_answer: str = ""
    committed: bool = True
    justification: str = ""


class BixbenchEquivalence(BaseModel):
    answer_correct: bool
    justification: str = ""


@judge("bixbench")
async def bixbench_judge(ctx: JudgeContext, model: LiteLLMModel | None) -> JudgeResult:
    """Extract the agent's value, then verify it against ``ideal`` per the question's eval_mode."""
    assert model is not None, f"{__name__} requires a rubric model"

    meta = ctx.problem.metadata
    ideal = str(meta.get("ideal") or "")
    mode = str(meta.get("eval_mode") or "llm_verifier")
    distractors = [str(d) for d in (meta.get("distractors") or [])]  # type: ignore[union-attr]

    # Call 1 — extraction. Never sees ``ideal``: a model shown the key can talk itself into finding
    # it in a hedged answer, which is exactly the failure the "must commit" rule exists to catch.
    # Keyed unprefixed so score_info.json keeps the "prompt"/"response" names the re-graders read.
    extraction, extract_meta = await call_json(
        model,
        EXTRACT_PROMPT.format(question=ctx.problem.hypothesis, answer=ctx.solution),
        BixbenchExtraction,
    )
    extracted = extraction.extracted_answer.strip() if extraction.committed else ""

    # Python applies the verifier. ``ideal`` missing means this problem did not come through the
    # converter, so there is nothing to compare arithmetically — go straight to the semantic call.
    verdict = verify_deterministically(mode, ideal, extracted) if ideal else None

    equiv_meta: dict = {}
    if verdict is None:
        # Call 2 — semantic equivalence, for llm_verifier and for anything Python could not settle.
        block = DISTRACTOR_BLOCK.format(items="\n".join(f"- {d}" for d in distractors)) if distractors else ""
        equivalence, equiv_meta = await call_json(
            model,
            EQUIVALENCE_PROMPT.format(
                question=ctx.problem.hypothesis,
                ideal=ideal or ctx.problem.rubric,
                distractors=block,
                answer=ctx.solution,
                extracted=extracted or "(none — the agent did not commit to a value)",
            ),
            BixbenchEquivalence,
            key="equiv_",
        )
        verdict = Verdict(correct=equivalence.answer_correct, method="llm", detail=equivalence.justification)

    return JudgeResult(
        raw_score=int(verdict.correct),
        max_score=1,
        correct=verdict.correct,
        criteria=[
            {
                "criterion": f"Final answer matches the ideal ({mode})",
                "score": int(verdict.correct),
                "justification": f"{verdict.detail} [{verdict.method}]",
                # BixBench grades the answer, not the path, so no cell is implicated in a failure.
                "relevant_steps": [],
                "extracted_answer": extracted,
                "ideal": ideal,
            }
        ],
        metadata={
            **extract_meta,
            **equiv_meta,
            "extracted_answer": extracted,
            "committed": extraction.committed,
            "eval_mode": mode,
            "ideal": ideal,
            "verification_method": verdict.method,
            "short_id": meta.get("short_id"),
            "question_id": meta.get("question_id"),
        },
    )
