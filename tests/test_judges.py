"""Tests for the per-benchmark judge registry (hypotest/env/judges/).

Covers judge selection and the Python-side scoring each judge does — the parts that must
not depend on an LLM call. The judges' prompts and schemas are exercised end-to-end by
TestRubricGrading in test_interpreter_env.py.
"""

import json
import shutil
from collections.abc import Awaitable
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import UUID

import pytest

from hypotest.env.biomni_judge import parse_rubric_levels, score_from_response, score_rich_levels
from hypotest.env.judges import (
    JUDGES,
    JudgeContext,
    JudgeResult,
    derive_first_wrong_step,
    parse_criterion_max_scores,
    resolve_judge,
)
from hypotest.env.judges.bioagent import CHECKS, collect_candidates
from hypotest.env.judges.bixbench import candidate_values, parse_interval, verify_deterministically
from hypotest.env.judges.heureka import _band
from hypotest.env.problem import ProblemInstance

if TYPE_CHECKING:
    from lmi import LiteLLMModel

ROOT = Path(__file__).resolve().parent.parent
BIXBENCH_JSONL = ROOT / "capsules" / "bixbench" / "bixbench.jsonl"

HYPOTEST_RUBRIC = "* 1 point: Data is loaded\n* 1 point: Correct conclusion"
BIOMNI_RUBRIC = "Criterion 1: Loads the data\nLevels: A=10 B=5 C=0\n\nCriterion 2: Concludes\nLevels: A=10 B=5 C=0"


def make_problem(**kwargs) -> ProblemInstance:
    base = {
        "id": UUID("12345678-1234-5678-1234-567812345678"),
        "hypothesis": "h",
        "protocol": "p",
        "answer": True,
        "rubric": HYPOTEST_RUBRIC,
        "max_points": 2,
    }
    return ProblemInstance.model_validate(base | kwargs)


class TestJudgeSelection:
    def test_every_registered_judge_resolves(self) -> None:
        for name in JUDGES:
            assert resolve_judge(make_problem(judge=name))[0] == name

    def test_problem_field_wins_over_config(self) -> None:
        class Config:
            judge = "biomni"

        assert resolve_judge(make_problem(judge="hypotest"), Config())[0] == "hypotest"

    def test_config_forces_judge_when_problem_is_unset(self) -> None:
        class Config:
            judge = "biomni"

        assert resolve_judge(make_problem(), Config())[0] == "biomni"

    @pytest.mark.parametrize(
        ("rubric", "expected"),
        [(HYPOTEST_RUBRIC, "hypotest"), (BIOMNI_RUBRIC, "biomni")],
    )
    def test_auto_falls_back_to_rubric_sniffing(self, rubric: str, expected: str) -> None:
        """Datasets written before ProblemInstance.judge keep their old grading method."""
        assert resolve_judge(make_problem(rubric=rubric))[0] == expected

    def test_unknown_judge_is_a_clear_error(self) -> None:
        with pytest.raises(ValueError, match="unknown judge 'nope'"):
            resolve_judge(make_problem(judge="nope"))


class TestRubricLevelParsing:
    """Negative level values must parse — every BiomniBench-DA rubric ends with a penalty criterion.

    Before the `-?` was added to the patterns in biomni_judge.parse_rubric_levels, the header
    regex stopped at the first negative value, so `A=0 B=-5 C=-10` parsed to `{"A": 0}`: the
    penalty silently scored 0 and every BiomniBench-DA total was inflated by up to 10/100.
    """

    PENALTY_RUBRIC = (
        "Criterion 1: Loads the data\nLevels: A=10 B=5 C=0\n\n"
        "Criterion 2: Source Reliability\nLevels: A=0 B=-5 C=-10"
    )

    def test_negative_levels_parse(self) -> None:
        assert parse_rubric_levels(self.PENALTY_RUBRIC) == {
            "criterion_1": {"A": 10, "B": 5, "C": 0},
            "criterion_2": {"A": 0, "B": -5, "C": -10},
        }

    def test_legacy_bracket_format_takes_negatives_too(self) -> None:
        rubric = "Criterion 1: Source Reliability\n[A] (0 points): fine\n[B] (-5 points): vague\n"
        assert parse_rubric_levels(rubric) == {"criterion_1": {"A": 0, "B": -5}}

    def test_penalty_is_subtracted_from_the_total(self) -> None:
        criteria = [{"level": "A"}, {"level": "B"}]
        assert score_rich_levels(criteria, self.PENALTY_RUBRIC) == 5  # 10 + (-5)
        assert [c["score"] for c in criteria] == [10, -5]

    def test_no_penalty_when_the_criterion_is_met(self) -> None:
        criteria = [{"level": "A"}, {"level": "A"}]
        assert score_rich_levels(criteria, self.PENALTY_RUBRIC) == 10  # 10 + 0

    def test_total_still_clamps_at_zero(self) -> None:
        """A penalty reduces a positive total but can never make the reward negative."""
        assert score_rich_levels([{"level": "C"}, {"level": "C"}], self.PENALTY_RUBRIC) == 0

    def test_full_marks_gate_no_longer_misfires_on_a_penalty_criterion(self) -> None:
        """A penalised criterion must keep its first_wrong_step (it is not "full marks")."""
        assert parse_criterion_max_scores(self.PENALTY_RUBRIC) == [10, 0]
        criteria = [
            {"score": 10, "relevant_steps": [{"step": 1, "correct": True}]},
            {"score": -5, "relevant_steps": [{"step": 3, "correct": False}]},
        ]
        derive_first_wrong_step(criteria, self.PENALTY_RUBRIC)
        assert [c["first_wrong_step"] for c in criteria] == [None, 3]
        # …but level A on that same criterion *is* full marks, so it stays None.
        met = [{"score": 0, "relevant_steps": [{"step": 3, "correct": False}]}]
        derive_first_wrong_step(met, "Criterion 1: Source Reliability\nLevels: A=0 B=-5 C=-10")
        assert met[0]["first_wrong_step"] is None

    def test_real_biomnibench_rubrics_all_carry_a_penalty_criterion(self) -> None:
        """Regression against the shipped dataset, not a synthetic rubric."""
        jsonl = ROOT / "capsules" / "biomnibench" / "biomnibench.jsonl"
        if not jsonl.is_file():
            pytest.skip("biomnibench capsules not staged")
        rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        negatives = [
            any(v < 0 for levels in parse_rubric_levels(r["rubric"]).values() for v in levels.values())
            for r in rows
        ]
        assert all(negatives), f"{negatives.count(False)}/{len(rows)} rubrics lost their penalty levels"

    def test_unparseable_total_score_keeps_its_sign(self) -> None:
        """`"total_score": -5` must not be read as +5 by the malformed-response fallback."""
        total, _, _ = score_from_response('not json at all "total_score": -5', BIOMNI_RUBRIC)
        assert total == 0  # clamped from -5, not 5


class TestHeurekaBands:
    """HeurekaBench's 0-5 G-Eval scale, computed from atomic-fact labels in Python."""

    @pytest.mark.parametrize(
        ("labels", "expected"),
        [
            (["PRESENT", "PRESENT"], 5),
            (["PRESENT", "MISSING"], 4),  # most present, nothing wrong
            (["PRESENT", "MISSING", "MISSING"], 3),  # present but a minority
            (["PRESENT", "PARTIAL", "INCORRECT"], 3),  # minor contradiction allowed
            (["PARTIAL", "MISSING"], 2),  # generic biological recall
            (["MISSING", "MISSING"], 1),
            (["INCORRECT", "INCORRECT", "PRESENT"], 1),  # most facts wrong
            ([], 1),  # answered, but the judge found no GT fact to label
        ],
    )
    def test_band(self, labels: list[str], expected: int) -> None:
        assert _band(labels, answered=True) == expected

    def test_unanswered_scores_zero(self) -> None:
        assert _band(["PRESENT", "PRESENT"], answered=False) == 0

    def test_incorrect_facts_block_full_marks(self) -> None:
        assert _band(["PRESENT", "PRESENT", "INCORRECT"], answered=True) < 4


BIOAGENT_TRUTH = ROOT / "capsules" / "bioagent-bench" / "_truth"
BIOAGENT_TASKS = sorted(CHECKS)

# Every column name any check reads, with values that match no truth file anywhere.
DECOY_TABLE = (
    "gene_id,Pathway,cluster_number,consensus_annotation,CHROM,POS,OTU,Phylum,JP4D,JC1A,"
    "cluster_id,predicted_cell_type,transcript_id,count,domain,species,chromosome,position\n"
    "FAKE1,Fake pathway,99,fake annotation,NODE_X,1,999,Fakephylum,1.0,1.0,42,Fake cell,"
    "ENST9999.1,7,Bacteria,Fake species,1,1\n"
)


def write_results(work_dir: Path, files: dict[str, str] | None = None, copy_truth: str | None = None) -> Path:
    """Stage a `results/` dir the way the agent is asked to (convert_bioagent_bench.build_hypothesis)."""
    results = work_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    for name, content in (files or {}).items():
        (results / name).write_text(content)
    if copy_truth:
        for f in (BIOAGENT_TRUTH / copy_truth).iterdir():
            shutil.copy(f, results / f.name)
    return results


@pytest.mark.skipif(not BIOAGENT_TRUTH.is_dir(), reason="bioagent-bench truth files not staged")
class TestBioagentChecks:
    """bioagent-bench's deterministic checks — no LLM involved."""

    @pytest.mark.parametrize("task", BIOAGENT_TASKS)
    def test_truth_satisfies_its_own_check(self, task: str, tmp_path: Path) -> None:
        """The answer key, handed in as the agent's output, must pass. Catches parsing mistakes."""
        write_results(tmp_path, copy_truth=task)
        passed, detail = CHECKS[task](collect_candidates(tmp_path), BIOAGENT_TRUTH / task)
        assert passed, f"{task}: {detail}"

    @pytest.mark.parametrize("task", BIOAGENT_TASKS)
    def test_unrelated_output_fails(self, task: str, tmp_path: Path) -> None:
        write_results(tmp_path, {"out.csv": DECOY_TABLE})
        passed, _ = CHECKS[task](collect_candidates(tmp_path), BIOAGENT_TRUTH / task)
        assert not passed

    def test_no_results_dir_yields_no_candidates(self, tmp_path: Path) -> None:
        assert collect_candidates(tmp_path) == []

    def test_any_produced_table_may_satisfy_the_rule(self, tmp_path: Path) -> None:
        """The agent is given no filename, so a passing table alongside junk still counts."""
        write_results(tmp_path, {"scratch.csv": DECOY_TABLE}, copy_truth="deseq")
        passed, _ = CHECKS["deseq"](collect_candidates(tmp_path), BIOAGENT_TRUTH / "deseq")
        assert passed

    @pytest.mark.asyncio
    async def test_judge_returns_binary_reward(self, tmp_path: Path) -> None:
        problem = make_problem(judge="bioagent", metadata={"task_id": "deseq"}, task_style="question", answer=None)
        ctx = JudgeContext(
            problem=problem,
            notebook="",
            solution="answer",
            work_dir=tmp_path,
            truth_dir=BIOAGENT_TRUTH / "deseq",
        )
        write_results(tmp_path, copy_truth="deseq")
        result = await JUDGES["bioagent"].fn(ctx, None)
        assert (result.raw_score, result.max_score, result.is_correct()) == (1, 1, True)
        assert result.criteria[0]["score"] == 1

        shutil.rmtree(tmp_path / "results")
        result = await JUDGES["bioagent"].fn(ctx, None)
        assert (result.raw_score, result.max_score, result.is_correct()) == (0, 1, False)

    @pytest.mark.asyncio
    async def test_missing_truth_dir_is_a_clear_error(self, tmp_path: Path) -> None:
        problem = make_problem(judge="bioagent", metadata={"task_id": "deseq"})
        ctx = JudgeContext(problem=problem, notebook="", solution="a", work_dir=tmp_path, truth_dir=None)
        with pytest.raises(ValueError, match="needs a staged truth dir"):
            await JUDGES["bioagent"].fn(ctx, None)

    @pytest.mark.asyncio
    async def test_unknown_task_id_is_a_clear_error(self, tmp_path: Path) -> None:
        problem = make_problem(judge="bioagent", metadata={"task_id": "giab"})
        ctx = JudgeContext(problem=problem, notebook="", solution="a", work_dir=tmp_path, truth_dir=tmp_path)
        with pytest.raises(ValueError, match="no bioagent-bench check for task_id 'giab'"):
            await JUDGES["bioagent"].fn(ctx, None)


class TestModelRequirement:
    def test_deterministic_judge_declares_it_needs_no_model(self) -> None:
        assert JUDGES["bioagent"].needs_model is False

    @pytest.mark.parametrize("name", ["hypotest", "biomni", "heureka", "biomystery"])
    def test_llm_judges_need_a_model(self, name: str) -> None:
        assert JUDGES[name].needs_model is True


class StubResponse:
    """Minimal stand-in for lmi's response object, as call_json consumes it."""

    def __init__(self, text: str):
        self.text = text
        self.reasoning_content = None


class StubModel:
    """Returns canned JSON per call, in order. Records the prompts it was given."""

    def __init__(self, *payloads: dict):
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    async def call_single(self, prompt: str, output_type=None, timeout=None) -> StubResponse:
        self.prompts.append(prompt)
        return StubResponse(json.dumps(self.payloads.pop(0)))


def run_judge(name: str, ctx: JudgeContext, model: "StubModel") -> Awaitable[JudgeResult]:
    """Invoke a judge with a StubModel — duck-typed for call_json, not a real LiteLLMModel."""
    return JUDGES[name].fn(ctx, cast("LiteLLMModel", model))

BIOMYSTERY_RUBRIC = (
    "The answer is Bacillus licheniformis. "
    "Score 1.0 if the model did not cheat AND got the answer correct. Score 0 otherwise."
)


class TestBiomysteryJudge:
    """BioMysteryBench: binary, conjunctive, and graded on the final answer only."""

    @staticmethod
    def problem() -> ProblemInstance:
        return make_problem(
            judge="biomystery",
            rubric=BIOMYSTERY_RUBRIC,
            max_points=1,
            task_style="question",
            answer=None,
            hypothesis="What bacteria is found in this sequenced dataset?",
            metadata={"problem_id": "hb002", "human_solvable": "yes"},
        )

    @staticmethod
    def stub(*, correct: bool, cheated: bool) -> StubModel:
        return StubModel(
            {"answer_correct": correct, "extracted_answer": "Bacillus licheniformis", "justification": "j"},
            {"cheated": cheated, "evidence": [], "justification": "j"},
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("correct", "cheated", "expected"),
        [(True, False, 1), (False, False, 0), (True, True, 0), (False, True, 0)],
    )
    async def test_score_is_the_conjunction(self, correct: bool, cheated: bool, expected: int) -> None:
        """'Score 1.0 if the model did not cheat AND got the answer correct. Score 0 otherwise.'"""
        ctx = JudgeContext(problem=self.problem(), notebook="### Cell 1:\nprint(1)", solution="Bacillus licheniformis")
        result = await run_judge("biomystery", ctx, self.stub(correct=correct, cheated=cheated))
        assert (result.raw_score, result.max_score) == (expected, 1)
        assert result.is_correct() is bool(expected)
        assert result.criteria[0]["score"] == expected

    @pytest.mark.asyncio
    async def test_correctness_call_never_sees_the_notebook(self) -> None:
        """The benchmark grades the final answer, not the path — enforced structurally."""
        notebook = "### Cell 1:\nSECRET_NOTEBOOK_MARKER"
        ctx = JudgeContext(problem=self.problem(), notebook=notebook, solution="Bacillus licheniformis")
        model = self.stub(correct=True, cheated=False)
        await run_judge("biomystery", ctx, model)

        correctness_prompt, cheat_prompt = model.prompts
        assert "SECRET_NOTEBOOK_MARKER" not in correctness_prompt
        assert "SECRET_NOTEBOOK_MARKER" in cheat_prompt
        # ...and conversely, the cheat check is not shown the answer key.
        assert "Bacillus licheniformis" in correctness_prompt
        assert "Bacillus licheniformis" not in cheat_prompt

    @pytest.mark.asyncio
    async def test_metadata_keeps_both_calls_and_the_split(self) -> None:
        ctx = JudgeContext(problem=self.problem(), notebook="nb", solution="a")
        result = await run_judge("biomystery", ctx, self.stub(correct=True, cheated=False))
        # Unprefixed keys keep scripts/regrade.py working; the second call is prefixed.
        assert {"prompt", "response", "cheat_prompt", "cheat_response"} <= set(result.metadata)
        assert result.metadata["human_solvable"] == "yes"
        assert result.metadata["extracted_answer"] == "Bacillus licheniformis"


BIXBENCH_METADATA = {
    "source": "futurehouse/BixBench",
    "short_id": "bix-1",
    "question_id": "bix-1-q1",
    "eval_mode": "str_verifier",
    "ideal": "0.0002",
    "distractors": ["7.820659E-05", "0.0003", "1.847038E-05"],
}


class TestBixbenchVerifiers:
    """The Python half of the bixbench judge: parsing `ideal` and applying each eval_mode."""

    @pytest.mark.parametrize(
        ("ideal", "expected"),
        [
            ("(1.50,1.54)", (1.50, 1.54)),
            ("(0.6 , 0.7)", (0.6, 0.7)),
            ("[20,25]", (20.0, 25.0)),
            ("(0.7,0.6)", (0.6, 0.7)),  # bounds normalised low-to-high
            ("0.0002", None),
            ("1-50", None),
        ],
    )
    def test_parse_interval(self, ideal: str, expected: tuple[float, float] | None) -> None:
        assert parse_interval(ideal) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("0.0002", [0.0002]),
            ("1.9E-05", [1.9e-05]),
            ("approximately 1,234 genes", [1234.0]),
            ("10.6%", [10.6, 0.106]),  # a percentage also reads as its fraction
            ("no number here", []),
        ],
    )
    def test_candidate_values(self, text: str, expected: list[float]) -> None:
        assert candidate_values(text) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("mode", "ideal", "extracted", "correct", "method"),
        [
            # str_verifier: exact first, then numeric within STR_REL_TOL.
            ("str_verifier", "0.0002", "0.0002", True, "exact"),
            ("str_verifier", "0.0002", "2e-4", True, "numeric"),
            ("str_verifier", "0.0002", "0.0003", False, "numeric"),
            ("str_verifier", "10.6%", "0.106", True, "numeric"),
            ("str_verifier", "166", "166 genes", True, "numeric"),
            # range_verifier: inclusive interval membership.
            ("range_verifier", "(1.50,1.54)", "1.52", True, "range"),
            ("range_verifier", "(1.50,1.54)", "1.50", True, "range"),
            ("range_verifier", "(1.50,1.54)", "1.24", False, "range"),
            # An empty extraction is a miss without spending a second call.
            ("str_verifier", "0.0002", "", False, "no_answer"),
            ("llm_verifier", "0.05", "", False, "no_answer"),
        ],
    )
    def test_deterministic_verdicts(self, mode: str, ideal: str, extracted: str, correct: bool, method: str) -> None:
        verdict = verify_deterministically(mode, ideal, extracted)
        assert verdict is not None, "should not have needed the model"
        assert (verdict.correct, verdict.method) == (correct, method)

    @pytest.mark.parametrize(
        ("mode", "ideal", "extracted"),
        [
            ("llm_verifier", "0.05", "0.05 lower in fungi"),  # always semantic
            ("str_verifier", "1-50", "between 1 and 50"),  # non-numeric ideal
            ("str_verifier", "0.0002", "it depends on the model"),  # nothing numeric to compare
            ("range_verifier", "0.0002", "0.0003"),  # ideal is not an interval
        ],
    )
    def test_falls_through_to_the_model(self, mode: str, ideal: str, extracted: str) -> None:
        assert verify_deterministically(mode, ideal, extracted) is None


class TestBixbenchJudge:
    """BixBench: binary, graded on the final answer only, one LLM call in the common case."""

    @staticmethod
    def problem(**metadata) -> ProblemInstance:
        return make_problem(
            judge="bixbench",
            rubric="Correct answer: 0.0002",
            max_points=1,
            task_style="question",
            answer=None,
            hypothesis="What is the adjusted p-value for regulation of T cell activation?",
            metadata=BIXBENCH_METADATA | metadata,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("extracted", "expected"),
        [("0.0002", 1), ("2e-4", 1), ("0.0003", 0)],
    )
    async def test_numeric_answers_are_scored_without_a_second_call(self, extracted: str, expected: int) -> None:
        model = StubModel({"extracted_answer": extracted, "committed": True, "justification": "j"})
        ctx = JudgeContext(problem=self.problem(), notebook="### Cell 1:\nprint(1)", solution=f"padj = {extracted}")
        result = await run_judge("bixbench", ctx, model)

        assert (result.raw_score, result.max_score) == (expected, 1)
        assert result.is_correct() is bool(expected)
        assert len(model.prompts) == 1, "the deterministic verifier should have settled it"

    @pytest.mark.asyncio
    async def test_llm_verifier_makes_the_equivalence_call(self) -> None:
        model = StubModel(
            {"extracted_answer": "0.05 lower in fungi", "committed": True, "justification": "j"},
            {"answer_correct": True, "justification": "same difference"},
        )
        problem = self.problem(eval_mode="llm_verifier", ideal="0.05", distractors=["0.015", "0.105"])
        ctx = JudgeContext(problem=problem, notebook="nb", solution="fungi are 0.05 lower")
        result = await run_judge("bixbench", ctx, model)

        assert result.raw_score == 1
        assert result.criteria[0]["justification"].endswith("[llm]")
        # The distractors reach the equivalence call as known-wrong answers, never the agent.
        assert "0.015" in model.prompts[1]

    @pytest.mark.asyncio
    async def test_hedging_is_not_a_committed_answer(self) -> None:
        """`committed: false` empties the extraction, which the verifier scores as a miss."""
        model = StubModel({"extracted_answer": "0.0002 or 0.0003", "committed": False, "justification": "j"})
        ctx = JudgeContext(problem=self.problem(), notebook="nb", solution="either 0.0002 or 0.0003")
        result = await run_judge("bixbench", ctx, model)

        assert result.raw_score == 0
        assert result.metadata["verification_method"] == "no_answer"
        assert len(model.prompts) == 1

    @pytest.mark.asyncio
    async def test_neither_call_ever_sees_the_notebook(self) -> None:
        """BixBench grades the final answer, not the path — enforced structurally."""
        notebook = "### Cell 1:\nSECRET_NOTEBOOK_MARKER"
        model = StubModel(
            {"extracted_answer": "between 1 and 50", "committed": True, "justification": "j"},
            {"answer_correct": False, "justification": "j"},
        )
        ctx = JudgeContext(problem=self.problem(ideal="1-50"), notebook=notebook, solution="between 1 and 50")
        await run_judge("bixbench", ctx, model)

        assert len(model.prompts) == 2
        assert not any("SECRET_NOTEBOOK_MARKER" in p for p in model.prompts)
        # ...and the extraction call is not shown the answer key it is meant to find unaided.
        assert "1-50" not in model.prompts[0]
        assert "1-50" in model.prompts[1]

    @pytest.mark.asyncio
    async def test_metadata_keeps_both_calls_and_the_answer_key(self) -> None:
        model = StubModel(
            {"extracted_answer": "x", "committed": True, "justification": "j"},
            {"answer_correct": False, "justification": "j"},
        )
        ctx = JudgeContext(problem=self.problem(eval_mode="llm_verifier"), notebook="nb", solution="x")
        result = await run_judge("bixbench", ctx, model)

        # Unprefixed keys keep scripts/regrade.py working; the second call is prefixed.
        assert {"prompt", "response", "equiv_prompt", "equiv_response"} <= set(result.metadata)
        assert result.metadata["ideal"] == "0.0002"
        assert result.metadata["question_id"] == "bix-1-q1"


class TestBixbenchDataset:
    """The converted capsules/bixbench/bixbench.jsonl follows the same shape as the others."""

    @pytest.mark.skipif(not BIXBENCH_JSONL.is_file(), reason="bixbench capsules not staged")
    def test_every_row_is_a_valid_problem_instance(self) -> None:
        rows = [json.loads(line) for line in BIXBENCH_JSONL.read_text().splitlines() if line.strip()]
        assert rows, "converted dataset is empty"
        for row in rows:
            problem = ProblemInstance.model_validate(row)
            assert resolve_judge(problem)[0] == "bixbench"
            assert problem.task_style == "question"
            assert problem.accepted is None
            assert problem.max_score == 1
            assert problem.metadata["eval_mode"] in {"str_verifier", "range_verifier", "llm_verifier"}
            assert problem.metadata["ideal"]

    @pytest.mark.skipif(not BIXBENCH_JSONL.is_file(), reason="bixbench capsules not staged")
    def test_input_data_path_points_at_a_shared_capsule(self) -> None:
        """Ids are per question, capsules are per paper — so the server's id fallback cannot apply."""
        rows = [json.loads(line) for line in BIXBENCH_JSONL.read_text().splitlines() if line.strip()]
        for row in rows:
            assert row["input_data_path"] == f"CapsuleData-{row['metadata']['capsule_uuid']}"
            assert (BIXBENCH_JSONL.parent / row["input_data_path"]).is_dir()
