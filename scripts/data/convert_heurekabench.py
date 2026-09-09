#!/usr/bin/env python3
"""Convert mlbio-epfl/HeurekaBench (sc-HeurekaBench) into hypotest's ProblemInstance jsonl + capsule format.

HeurekaBench (Panigrahi, Videnovic & Brbic, ICLR 2026; arXiv:2601.01678) ships its
benchmark as six JSON files under ``scheurekabench/benchmark/``:

    {mcq,oeq}.json        full split      (12-13 papers, 37-41 insights, 50 sub-questions)
    {mcq,oeq}_lite.json   lite split      (computationally lighter subset)
    {mcq,oeq}_tu.json     tool-use split  (questions needing external tools/databases)

Each file is ``{paper_id: {insight_id: entry}}`` where ``entry`` has:

  - ``summary`` / ``how`` / ``relevant``: the paper-derived insight, its derivation, and the
    verbatim quote from the article it came from. All three LEAK THE ANSWER, so none of them
    is ever put in front of the agent — they go into ``metadata`` for provenance only.
  - ``data``: {path -> one-line description} for the data files the question needs. Paths are
    repo-relative (``benchmark/scdata/paper1/data/dataset1.h5ad``); the actual bytes are the
    44 GB Google Drive tarball the README tells you to unpack into ``benchmark/scdata/``.
  - ``oe_question`` / ``mcq_question``: a markdown blob holding 1-N ``**QuestionN:**`` /
    ``**AnswerN:**`` pairs. The question text is the task; the answer text is ground truth.

Upstream, agents are run once per sub-question and scored per sub-question: open-ended answers
get a 1-5 "correctness" rating from a G-Eval LLM judge against the GT answer (see
``scheurekabench/geval_prompts/eval_prompts.py``), and MCQ answers are scored by exact match of
the selected option set. The reported metric is the mean rating / accuracy over sub-questions.

Mapping onto hypotest
---------------------
hypotest grades a rollout with a single rubric-driven judge call over the whole notebook, so the
conversion turns HeurekaBench's per-sub-question scoring into per-criterion scoring:

  - ``hypothesis``      <- the sub-question text(s), verbatim. With the default
                           ``--granularity insight`` all sub-questions of one insight become one
                           task (they share a dataset, and re-running a multi-GB single-cell
                           analysis once per sub-question is pure waste); ``--granularity
                           question`` emits one task per sub-question, matching upstream exactly.
  - ``protocol``        <- a "## Data Files" section built from ``data`` (basename + description),
                           mirroring what upstream hands the agent. NOT a step-by-step protocol,
                           but it is the field hypotest renders into the task's <objectives>.
  - ``rubric``          <- one criterion per sub-question, in hypotest's ``N. (X points)`` format
                           (so ``is_biomni_rubric`` stays false and the integer-per-criterion
                           hypotest judge is used, not the biomni A/B/C one):
                             * open-ended: 5 points/criterion, with the G-Eval 1-5 scale and the
                               GT answer inlined -> mean criterion score is directly comparable
                               to the paper's mean rating.
                             * MCQ: 1 point/criterion, awarded only on an exact option-set match.
  - ``max_score``       <- 5*n_subquestions (oe) or n_subquestions (mcq)
  - ``input_data_path`` <- the PAPER id. There is one capsule per paper — 13 in total, matching
                           HeurekaBench's own ``scdata/paperN/data/`` layout — holding the union
                           of every file that paper's questions reference, hardlinked flat so the
                           agent sees them by basename. Every task derived from a paper shares
                           that capsule, so a multi-GB h5ad is staged once instead of once per
                           insight. The union is taken over all six question JSONs, not just the
                           split being converted, so one capsule_dir serves every split.
  - ``task_style``      <- "question" (open research question, rubric-only grading; no
                           accept/reject ground truth exists here)
  - ``id``              <- uuid5(NAMESPACE, task_id), deterministic across machines/re-runs

Two deliberate deviations from upstream, both flagged in the README notes this script prints:

  1. hypotest's judge sees the full notebook, not just the final answer, and the rubric asks it
     to score the procedure as well. Absolute scores will therefore not be numerically identical
     to published HeurekaBench numbers; they are comparable across models run this way.
  2. hypotest normalizes reward to [0, 1] as score/max_score, so a 1/5 G-Eval rating maps to 0.2
     rather than the paper's floor of 1. Criterion score 0 is reserved for "no answer at all".

The paper PDFs shipped in the repo (``benchmark_validation/insights_*/papers/``) are NEVER copied
into a capsule: the GT answers are quoted from them verbatim.

Usage:
    # 1. clone the repo and unpack the 44 GB scdata tarball as per its README:
    #      cat scdata.part_* > scdata.tar.zst && tar -I zstd -xf scdata.tar.zst
    # 2. convert (open-ended, full split):
    .venv/bin/python scripts/convert_heurekabench.py \
        --repo-dir /path/to/HeurekaBench \
        --q-type oe --split full \
        --out-jsonl problems/problems_heurekabench_oe.jsonl \
        --capsule-dir capsules_heurekabench/

    # multiple-choice, lite split, one task per sub-question:
    .venv/bin/python scripts/convert_heurekabench.py --repo-dir ... \
        --q-type mcq --split lite --granularity question

    # generate the jsonl before the 44 GB download lands (capsules will be empty):
    .venv/bin/python scripts/convert_heurekabench.py --repo-dir ... --allow-missing-data
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

REPO_ID = "mlbio-epfl/HeurekaBench"
# Deterministic namespace so re-running the conversion (or converting on another machine)
# always yields the same problem ids for the same task id.
UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, f"hypotest:{REPO_ID}")

# Points per criterion. Open-ended criteria use HeurekaBench's 1-5 G-Eval correctness scale
# (0 added for "no answer"); MCQ criteria are exact-match, so binary.
OE_POINTS_PER_QUESTION = 5
MCQ_POINTS_PER_QUESTION = 1

# Verbatim from scheurekabench/evaluate_agent_answer.py — keep in lockstep so this conversion
# splits the question blobs into exactly the same sub-questions upstream scores.
MCQ_PATTERN = re.compile(
    r"\*\*Question(\d+):\*\*\s*(.*?)\s*((?:[A-D]\).*?(?:\s+))+)\*\*Answer\1:\*\*\s*([A-D](?:,[A-D])*)", re.DOTALL
)
OE_PATTERN = re.compile(
    r"\*\*Question(\d+):\*\*\s*(.*?)\s*\*\*Answer\1:\*\*\s*(.*?)(?=\s*\*\*Question\d+:|\Z)", re.DOTALL
)
MCQ_OPTION_PATTERN = re.compile(r"([A-D])\)\s*(.*)")

# Grading preamble prepended to every open-ended rubric. Condensed from HeurekaBench's
# G_EVAL_BASIC_TEMPLATE (scheurekabench/geval_prompts/eval_prompts.py) — the atomic-fact
# decomposition and the PRESENT/PARTIAL/MISSING/INCORRECT scale are reproduced; the parts of the
# upstream prompt that duplicate hypotest's own judge contract (output format, "grade only
# against GT") are dropped because hypotest's RUBRIC_SCORE_PROMPT_QUESTION already states them.
OE_RUBRIC_PREAMBLE = """\
Each criterion below is one research sub-question with its ground-truth (GT) answer. Score each
criterion independently, 0-5, using HeurekaBench's G-Eval correctness scale.

For a criterion, first decompose its GT answer into atomic facts (cell type/condition,
direction/magnitude of change, gene/pathway names, statistical evidence, method, conclusion),
then label each fact against the agent's answer:
  - PRESENT:   same meaning AND tied to dataset-derived quantitative/statistical output or
               dataset cluster/subtype identifiers (percentages, fold changes, p-values,
               cluster IDs, enrichment scores).
  - PARTIAL:   correct meaning but supported only by descriptive biology, lists of plausible
               options, hedged language ("likely", "typically", "e.g."), or general biological
               recall rather than this dataset's evidence.
  - MISSING:   not mentioned.
  - INCORRECT: wrong, or contradicts a GT fact.

Then award points:
  5 - all GT facts PRESENT; no MISSING, INCORRECT, or contradicted facts.
  4 - most or all GT facts PRESENT; MISSING allowed, INCORRECT not allowed. A detailed,
      dataset-grounded finding counts even if it does not repeat the GT's broader wording.
  3 - some (not all) GT facts PRESENT, and at least one PARTIAL or MISSING; minor
      contradictions allowed.
  2 - no GT fact PRESENT, some PARTIAL; answer reads as generic biological recall rather than
      dataset evidence.
  1 - all GT facts MISSING, or most INCORRECT, or major contradictions of GT, or the agent
      states it cannot answer.
  0 - the agent produced no answer to this sub-question at all.

Grade only against the GT answer; ignore outside knowledge. Extra information beyond the GT
neither helps nor hurts unless it contradicts the GT. Style, length, and paraphrasing do not
matter. All GT facts within a criterion are weighted equally."""

MCQ_RUBRIC_PREAMBLE = """\
Each criterion below is one multiple-choice sub-question with its correct option set. Award the
point only if the agent's final answer selects exactly that set of options — no missing options
and no extra ones. A correct option letter reached without the corresponding analysis in the
notebook, or a selection stated only as one possibility among several, does not earn the point.
Award 0 otherwise. No partial credit."""


def load_env(path: Path = ROOT / ".env") -> None:
    """Populate os.environ from a KEY=VALUE .env file (does not overwrite)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def parse_oe_questions(text: str) -> list[dict[str, str]]:
    """Split an ``oe_question`` blob into [{num, question, answer}] (upstream's parse_oe_questions)."""
    return [
        {"num": num, "question": q.strip(), "answer": a.strip()} for num, q, a in OE_PATTERN.findall(text)
    ]


def parse_mcq_questions(text: str) -> list[dict[str, Any]]:
    """Split an ``mcq_question`` blob into [{num, question, options, answer}] (upstream's parse_mcq_question)."""
    out = []
    for num, q_text, options_block, ans in MCQ_PATTERN.findall(text):
        out.append({
            "num": num,
            "question": q_text.strip(),
            "options": dict(MCQ_OPTION_PATTERN.findall(options_block)),
            "answer": ans.strip(),
        })
    return out


def render_question_text(sub: dict[str, Any], q_type: str, index: int | None = None) -> str:
    """The agent-facing text for one sub-question (options included for MCQ, answer never)."""
    prefix = f"{index}. " if index is not None else ""
    if q_type == "mcq":
        opts = "\n".join(f"{letter}) {text}" for letter, text in sorted(sub["options"].items()))
        return f"{prefix}{sub['question']}\n{opts}"
    return f"{prefix}{sub['question']}"


def build_hypothesis(subs: list[dict[str, Any]], q_type: str) -> str:
    """The <question> block. Numbered only when an insight bundles several sub-questions."""
    if len(subs) == 1:
        body = render_question_text(subs[0], q_type)
    else:
        body = "\n\n".join(render_question_text(s, q_type, i + 1) for i, s in enumerate(subs))
    if q_type == "mcq":
        note = (
            "Answer every question above by selecting all options that apply. One or more options may be "
            "correct. Base each selection on your own analysis of the provided data, and state the selected "
            "option letters explicitly in your final answer."
        )
        return f"{body}\n\n{note}"
    return body


def build_protocol(data: dict[str, str]) -> str:
    """The <objectives> block: the data-file manifest, by the basename the agent will see."""
    lines = [f"- `{Path(path).name}`: {desc}" for path, desc in data.items()]
    return (
        "## Data Files\n"
        "The following files are available in your working directory:\n\n" + "\n".join(lines)
    )


def build_rubric(task_label: str, subs: list[dict[str, Any]], q_type: str) -> tuple[str, int]:
    """Return (rubric_text, max_score) in hypotest's ``N. (X points)`` criterion format."""
    per_q = MCQ_POINTS_PER_QUESTION if q_type == "mcq" else OE_POINTS_PER_QUESTION
    max_score = per_q * len(subs)
    preamble = MCQ_RUBRIC_PREAMBLE if q_type == "mcq" else OE_RUBRIC_PREAMBLE

    criteria = []
    for i, sub in enumerate(subs, start=1):
        head = f"{i}. ({per_q} point{'s' if per_q != 1 else ''}) Sub-question {i}: {sub['question']}"
        if q_type == "mcq":
            opts = "\n".join(f"     {letter}) {text}" for letter, text in sorted(sub["options"].items()))
            body = f"{opts}\n   Correct option set: {sub['answer']}"
        else:
            body = f"   Ground-truth answer: {sub['answer']}"
        criteria.append(f"{head}\n{body}")

    rubric = (
        f"RUBRIC: {task_label}\n\n"
        f"Total Points: {max_score}/{max_score}\n\n"
        f"{preamble}\n\n"
        + "\n\n".join(criteria)
    )
    return rubric, max_score


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def collect_paper_data(benchmark_dir: Path) -> dict[str, dict[str, str]]:
    """Union of every data file referenced by each paper, across ALL question JSONs present.

    One capsule per paper means the capsule must be split-agnostic: converting the mcq split
    must not leave behind a capsule that is missing files the oe split's tasks reference. So the
    union is taken over all six ``{mcq,oeq}{,_lite,_tu}.json`` rather than just the file being
    converted, making the staging step idempotent no matter which split you run first.
    """
    per_paper: dict[str, dict[str, str]] = {}
    for name in ("oeq.json", "oeq_lite.json", "oeq_tu.json", "mcq.json", "mcq_lite.json", "mcq_tu.json"):
        path = benchmark_dir / name
        if not path.exists():
            continue
        for paper_id, insights in json.loads(path.read_text()).items():
            for entry in insights.values():
                per_paper.setdefault(paper_id, {}).update(entry.get("data", {}))
    return per_paper


def link_into(src: Path, target: Path) -> None:
    """Hardlink src -> target, falling back to a copy across filesystems.

    Hardlink rather than copy: these h5ad files run to several GB each and capsule contents are
    never modified in place (the dataset server copies them again into a throwaway per-rollout
    work_dir). Keep the scdata source read-only (``chmod -R a+r``, per the HeurekaBench README)
    so a stray in-place write can never reach the shared inode.
    """
    if target.exists():
        target.unlink()
    try:
        os.link(src, target)
    except OSError:
        shutil.copy2(src, target)


def stage_paper_capsules(
    benchmark_dir: Path, capsule_dir: Path, allow_missing: bool, whole_paper_dir: bool
) -> tuple[dict[str, int], list[str]]:
    """Build one capsule per paper: ``<capsule_dir>/<paper_id>/<basename>``.

    Every task derived from a paper shares that paper's capsule, so a multi-GB h5ad is staged
    once rather than once per insight. Returns (files staged per paper, missing source paths).
    """
    staged: dict[str, int] = {}
    missing: list[str] = []

    for paper_id, data in sorted(collect_paper_data(benchmark_dir).items()):
        dest = capsule_dir / paper_id
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

        if whole_paper_dir:
            # Everything shipped for the paper, referenced by a question or not.
            src_dir = benchmark_dir / "scdata" / paper_id / "data"
            if not src_dir.is_dir():
                missing.append(str(src_dir))
                staged[paper_id] = 0
                continue
            for src in sorted(p for p in src_dir.rglob("*") if p.is_file()):
                if (dest / src.name).exists():
                    raise ValueError(f"basename collision flattening {src_dir} into {dest}: {src.name}")
                link_into(src, dest / src.name)
            staged[paper_id] = len(list(dest.iterdir()))
            continue

        for rel_path in data:
            # Keys look like "benchmark/scdata/paper1/data/x.h5ad" — resolve against --benchmark-dir.
            rel = rel_path.split("benchmark/", 1)[-1] if "benchmark/" in rel_path else rel_path
            src = benchmark_dir / rel
            if not src.exists():
                missing.append(str(src))
                continue
            link_into(src, dest / src.name)
        staged[paper_id] = len(list(dest.iterdir()))

    if missing and not allow_missing:
        raise FileNotFoundError(f"missing data files (pass --allow-missing-data to skip): {missing}")
    return staged, missing


def build_problems(
    dataset: dict[str, dict[str, dict[str, Any]]],
    q_type: str,
    split: str,
    granularity: str,
) -> list[dict[str, Any]]:
    key = f"{q_type if q_type == 'mcq' else 'oe'}_question"
    problems: list[dict[str, Any]] = []

    for paper_id, insights in dataset.items():
        for insight_id, entry in insights.items():
            if key not in entry:
                print(f"[skip] {paper_id}/{insight_id}: no '{key}'")
                continue
            subs = parse_mcq_questions(entry[key]) if q_type == "mcq" else parse_oe_questions(entry[key])
            if not subs:
                print(f"[skip] {paper_id}/{insight_id}: no sub-questions parsed from '{key}'")
                continue

            groups = [subs] if granularity == "insight" else [[s] for s in subs]
            for group in groups:
                insight_slug = slugify(insight_id)  # "insight #1" -> "insight1"
                task_id = f"{paper_id}-{insight_slug}"
                if granularity == "question":
                    task_id += f"-q{group[0]['num']}"
                task_label = f"{REPO_ID} {q_type}/{split} — {paper_id} / {insight_id}"

                rubric, max_score = build_rubric(task_label, group, q_type)
                problems.append({
                    "id": str(uuid.uuid5(UUID_NAMESPACE, f"{q_type}:{split}:{task_id}")),
                    "hypothesis": build_hypothesis(group, q_type),
                    "protocol": build_protocol(entry.get("data", {})),
                    "answer": None,
                    "rubric": rubric,
                    "max_points": max_score,
                    # One capsule per paper, shared by every task derived from it. The workspace
                    # therefore also holds files this particular question does not need; the
                    # <objectives> manifest lists only the relevant ones.
                    "input_data_path": paper_id,
                    "task_style": "question",
                    "nb_primary_language": "python",
                    # Open-ended tasks are graded by HeurekaBench's own judging protocol, as a
                    # registered judge (see hypotest/env/judges/heureka.py): the LLM labels the
                    # GT's atomic facts and Python computes the 0-5 band. There is no MCQ judge —
                    # leaving `judge` unset sends those to hypotest's integer judge, which reads
                    # MCQ_RUBRIC_PREAMBLE.
                    **({} if q_type == "mcq" else {"judge": "heureka"}),
                    "metadata": {
                        "source": REPO_ID,
                        "q_type": q_type,
                        "split": split,
                        "task_id": task_id,
                        "paper": paper_id,
                        "insight": insight_id,
                        "sub_questions": [s["num"] for s in group],
                        "n_sub_questions": len(group),
                        # Provenance from the source article. Answer-revealing — judge-side and
                        # debugging only; never rendered into the agent's task prompt.
                        "insight_summary": entry.get("summary"),
                        "insight_how": entry.get("how"),
                        "insight_quote": entry.get("relevant"),
                        "data_files": entry.get("data", {}),
                    },
                })
                print(f"[ok] {task_id} -> capsule {paper_id} ({len(group)} sub-question(s), {max_score} pts)")

    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-dir", type=Path, default=None, help="Clone of mlbio-epfl/HeurekaBench")
    ap.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Path to scheurekabench/benchmark (defaults to <repo-dir>/scheurekabench/benchmark)",
    )
    ap.add_argument("--q-type", choices=("oe", "mcq"), default="oe")
    ap.add_argument("--split", choices=("full", "lite", "tu"), default="full")
    ap.add_argument(
        "--granularity",
        choices=("insight", "question"),
        default="insight",
        help="'insight' bundles an insight's sub-questions into one task (default); "
        "'question' emits one task per sub-question, matching upstream scoring granularity",
    )
    ap.add_argument("--out-jsonl", type=Path, default=None)
    ap.add_argument("--capsule-dir", type=Path, default=ROOT / "capsules_heurekabench")
    ap.add_argument("--limit", type=int, default=None, help="Only convert the first N tasks")
    ap.add_argument("--only", default=None, help="Only tasks whose id contains this substring")
    ap.add_argument(
        "--allow-missing-data",
        action="store_true",
        help="Emit tasks whose scdata files are not on disk yet (capsules will be incomplete)",
    )
    ap.add_argument(
        "--whole-paper-dir",
        action="store_true",
        help="Stage all of scdata/<paper>/data/ into each capsule instead of only the files the "
        "questions reference (larger capsules; includes anything the benchmark ships but never cites)",
    )
    args = ap.parse_args()

    load_env()

    benchmark_dir = args.benchmark_dir or (args.repo_dir / "scheurekabench" / "benchmark" if args.repo_dir else None)
    if benchmark_dir is None:
        sys.exit("pass --repo-dir (clone of mlbio-epfl/HeurekaBench) or --benchmark-dir")
    if not benchmark_dir.is_dir():
        sys.exit(f"not a directory: {benchmark_dir}")

    suffix = {"full": "", "lite": "_lite", "tu": "_tu"}[args.split]
    json_name = f"{'mcq' if args.q_type == 'mcq' else 'oeq'}{suffix}.json"
    json_path = benchmark_dir / json_name
    if not json_path.exists():
        sys.exit(f"missing {json_path}")

    out_jsonl = args.out_jsonl or ROOT / "problems" / f"problems_heurekabench_{args.q_type}_{args.split}.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.capsule_dir.mkdir(parents=True, exist_ok=True)

    # Capsules are staged once for every paper in the benchmark, independently of which split is
    # being converted, so all splits can be served from the same capsule_dir.
    staged, missing = stage_paper_capsules(
        benchmark_dir, args.capsule_dir, args.allow_missing_data, args.whole_paper_dir
    )
    for paper_id, n_files in staged.items():
        print(f"[capsule] {paper_id}: {n_files} file(s)")

    dataset = json.loads(json_path.read_text())
    problems = build_problems(dataset, args.q_type, args.split, args.granularity)

    if args.only:
        problems = [p for p in problems if args.only in p["metadata"]["task_id"]]
    if args.limit is not None:
        problems = problems[: args.limit]

    with out_jsonl.open("w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")

    n_sub = sum(p["metadata"]["n_sub_questions"] for p in problems)
    n_papers = len({p["input_data_path"] for p in problems})
    print(f"\nWrote {len(problems)} problems ({n_sub} sub-questions, {n_papers} papers) -> {out_jsonl}")
    print(f"Capsules -> {args.capsule_dir} ({len(staged)} papers)")
    if missing:
        print(f"\nWARNING: {len(missing)} data file(s) not found on disk, e.g. {missing[:3]}")
        print("Unpack the scdata tarball per the HeurekaBench README, then re-run without --allow-missing-data.")
    print(
        "\nserver.yaml:\n"
        f"  capsule_dir: {args.capsule_dir}/\n"
        f"  problem_jsonl: {out_jsonl}\n"
        "  include_protocol: true\n"
        "\n(The judge is set per problem — see the `judge` field — so no server.yaml override is needed.)\n"
    )


if __name__ == "__main__":
    main()
