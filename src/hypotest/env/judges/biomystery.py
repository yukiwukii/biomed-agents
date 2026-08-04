"""BioMysteryBench's judge: final-answer-only correctness, ANDed with an anti-cheat check.

Anthropic/BioMysteryBench-full ships its grading criterion per problem, in the ``answer_rubric``
column, and every one of the 90 rubrics ends with the same sentence:

    "Score 1.0 if the model did not cheat AND got the answer correct. Score 0 otherwise."

Two properties define this benchmark, and this module is built around them:

1. **The final answer is graded, not the path.** Anthropic's write-up is explicit that the
   benchmark grades models "on their final answer, rather than the path the model took to get
   there" — the opposite of hypotest's own judge, which scores the notebook's procedure. So the
   correctness call here sees the question, the rubric and the submitted answer, and **nothing
   else**. That is why this is a two-call judge: the notebook is withheld from the correctness
   judgment structurally rather than by asking a single judge to ignore it.

2. **Scoring is binary and conjunctive.** A second call decides only whether the agent cheated,
   for which it must see the notebook. Python ANDs the two booleans (``score = not cheated and
   correct``) rather than asking a model to apply the rule — the same "the LLM labels, Python does
   the arithmetic" split as ``biomni.py`` and ``bioagent.py``.

Cheating is narrowly defined by the dataset (README "Rules", plus the v11 CHANGELOG note):
looking up GEO/SRA/ENA/BioProject accession IDs to identify the original dataset, source
publication or study metadata is disallowed, as is otherwise reverse-identifying the source
dataset. Standard bioinformatics database use — gene ID lookup, sequence annotation, reference
genome download, BLAST — **is** allowed, and recalling information from memory is explicitly
**not** penalized even when it happens to include the source publication.

Anthropic has not published a grader prompt, so the prompts below are written to the dataset's own
stated criterion rather than copied from an upstream implementation.
"""

from __future__ import annotations

from lmi import LiteLLMModel
from pydantic import BaseModel, Field

from .base import JudgeContext, JudgeResult, StepEvidence, call_json, judge

# Verbatim from the dataset README's "Rules" section and the v11 CHANGELOG's "Grading rule".
CHEATING_POLICY = """
Looking up GEO/SRA/ENA/BioProject accession IDs to identify the original dataset, the source
publication, or study metadata is NOT allowed, and neither is otherwise reverse-identifying the
source dataset (for example, searching a distinctive sample-ID string, a title, or a fingerprint of
the data against a public archive or the web in order to find the study it came from).

Standard bioinformatics database use IS allowed: gene ID lookup, sequence annotation, reference
genome download, and BLAST. Downloading reference data, annotations, or tools is allowed.

Recalling information from memory is NOT cheating, even if what the agent recalls happens to
include the source publication. Only an actual lookup performed during the run counts. Correctly
inferring the answer from the data provided is the intended solution path, however impressive the
inference — it is never cheating.
""".strip()

BIOMYSTERY_CORRECTNESS_PROMPT = """
You are grading one answer to a bioinformatics problem, against the problem's official answer key.

Here is the question the agent was asked:
<question>
{question}
</question>

Here is the official grading criterion. It states the expected answer:
<answer-rubric>
{rubric}
</answer-rubric>

Here is the agent's final answer, in full:
<final-answer>
{answer}
</final-answer>

Decide one thing only: does the final answer satisfy the grading criterion?

  - Judge the answer itself. You are deliberately not being shown the agent's analysis, because
    this benchmark grades the final answer rather than the path taken to reach it. Do not speculate
    about how the answer was obtained, and do not reward or penalise apparent effort.
  - Apply the criterion exactly as written, including any alternatives or tolerances it allows
    ("accept any of…", "±3 bp", "aka…"). Do not broaden it with alternatives of your own, and do
    not narrow it either.
  - Formatting, capitalisation, surrounding prose and word order do not matter; the answer needs to
    state the expected content, not to match it character for character. A synonym or an equivalent
    identifier for exactly the thing the criterion names counts as correct.
  - If the criterion requires several items (a list, a set of samples, a mapping), all required
    items must be present and correct. Extra incorrect items make the answer wrong.
  - An answer that hedges between several candidates, or that states the correct value only as one
    possibility among others, does not satisfy the criterion. The agent must commit.
  - If the agent gave no answer, or declined, the answer is not correct.

Respond with a JSON object with these keys:
  - "answer_correct": boolean — whether the final answer satisfies the criterion
  - "extracted_answer": string — the answer the agent actually committed to, quoted or closely
    paraphrased from its final answer; "" if it gave none
  - "justification": one or two sentences comparing the extracted answer to the criterion

Do not output a score. The score is computed programmatically from your boolean.
""".strip()

BIOMYSTERY_CHEAT_PROMPT = """
You are checking one rule for a bioinformatics benchmark run. The benchmark's problems are built
from anonymized public-archive data, and the agent is required to answer by analysing the data it
was given rather than by identifying which public dataset it came from.

{policy}

Here is the question the agent was asked:
<question>
{question}
</question>

Here is the agent's full analysis notebook, including code and outputs:
<notebook>
{notebook}
</notebook>

Decide one thing only: did the agent cheat, as defined above?

Set "cheated" to true only on positive evidence in the notebook of a disallowed lookup — a request,
query, or tool invocation that searches an accession, sample identifier, title, or data fingerprint
against a public archive or the web in order to identify the source dataset, publication, or study
metadata. Quote what you found. Absence of evidence is not evidence of cheating: if the agent simply
reasoned its way to the answer, or recalled something from memory, "cheated" is false. When the
evidence is ambiguous, answer false.

Respond with a JSON object with these keys:
  - "cheated": boolean
  - "evidence": array of the notebook cells bearing on this decision (empty if none), each with
    "step" (the cell index, as labelled "### Cell N:" above), "note" (what the cell did), and
    "correct" (false for a cell that performs a disallowed lookup, true otherwise)
  - "justification": one or two sentences; if true, name the disallowed lookup you found

Do not output a score. The score is computed programmatically from your boolean.
""".strip()


class BiomysteryCorrectness(BaseModel):
    answer_correct: bool
    extracted_answer: str = ""
    justification: str


class BiomysteryCheat(BaseModel):
    cheated: bool
    evidence: list[StepEvidence] = Field(default_factory=list)
    justification: str


@judge("biomystery")
async def biomystery_judge(ctx: JudgeContext, model: LiteLLMModel | None) -> JudgeResult:
    """Two calls: correctness (answer only) and cheating (notebook only). Python ANDs them."""
    assert model is not None, f"{__name__} requires a rubric model"

    # Call 1 — correctness. Deliberately does NOT receive the notebook: the benchmark grades the
    # final answer, not the path. Keyed unprefixed so score_info.json keeps the "prompt"/"response"
    # keys the offline re-graders read.
    correctness, meta = await call_json(
        model,
        BIOMYSTERY_CORRECTNESS_PROMPT.format(
            question=ctx.problem.hypothesis,
            rubric=ctx.problem.rubric,
            answer=ctx.solution,
        ),
        BiomysteryCorrectness,
    )

    # Call 2 — the anti-cheat check, the one judgment that genuinely needs the transcript.
    cheat, cheat_meta = await call_json(
        model,
        BIOMYSTERY_CHEAT_PROMPT.format(
            policy=CHEATING_POLICY,
            question=ctx.problem.hypothesis,
            notebook=ctx.notebook,
        ),
        BiomysteryCheat,
        key="cheat_",
    )

    # "Score 1.0 if the model did not cheat AND got the answer correct. Score 0 otherwise."
    passed = correctness.answer_correct and not cheat.cheated

    return JudgeResult(
        raw_score=int(passed),
        max_score=1,
        correct=passed,
        criteria=[
            {
                "criterion": "Correct answer, obtained without disallowed lookups",
                "score": int(passed),
                "justification": (
                    f"answer_correct={correctness.answer_correct}, cheated={cheat.cheated}. "
                    f"{correctness.justification} {cheat.justification}".strip()
                ),
                "relevant_steps": [s.model_dump() for s in cheat.evidence],
                "answer_correct": correctness.answer_correct,
                "cheated": cheat.cheated,
                "extracted_answer": correctness.extracted_answer,
            }
        ],
        metadata={
            **meta,
            **cheat_meta,
            "answer_correct": correctness.answer_correct,
            "cheated": cheat.cheated,
            "extracted_answer": correctness.extracted_answer,
            "human_solvable": ctx.problem.metadata.get("human_solvable"),
        },
    )
