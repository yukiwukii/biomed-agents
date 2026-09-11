#!/usr/bin/env python3
"""Convert Anthropic/BioMysteryBench-full into hypotest's ProblemInstance jsonl + capsule format.

The dataset (gated; clone it from https://huggingface.co/datasets/Anthropic/BioMysteryBench-full)
ships as:

    problems.csv / problems.parquet   one row per problem
    data/<id>.zip                     that problem's data files (~145 GB unpacked across 90)
    README.md / CHANGELOG.md          the rules and the grading rule

``problems.csv`` columns:

  - ``id``              problem identifier (``hb001`` … or an Airtable-style ``rec…`` id)
  - ``question``        the task prompt shown to the model
  - ``answer_rubric``   the grading criterion, CONTAINING THE EXPECTED ANSWER — judge-side only,
                        never shown to the agent
  - ``allowed_domains`` network domains the solving environment may reach (identical for all 90)
  - ``human_solvable``  "yes" if at least one human benchmarker solved it, else "no"
                        (v11: 73 solvable / 17 hard) — kept in metadata so results can be split
                        the way Anthropic reports them

Mapping onto hypotest
---------------------
  - ``hypothesis``      <- ``question``, verbatim
  - ``rubric``          <- ``answer_rubric``, verbatim. Every one of the 90 ends with
                           "Score 1.0 if the model did not cheat AND got the answer correct.
                           Score 0 otherwise."
  - ``max_points``      <- 1 (the benchmark is binary; the judge also returns max_score=1)
  - ``task_style``      <- "question"
  - ``judge``           <- "biomystery" (hypotest/env/judges/biomystery.py)
  - ``protocol``        <- "" deliberately. Upstream shows the model the question and the extracted
                           data files, nothing else; hypotest renders ``protocol`` into an
                           <objectives> block, so leaving it empty is the faithful choice. (An
                           empty protocol renders as prompts.NO_PROTOCOL_MSG.)
  - ``input_data_path`` <- the problem id; one capsule per problem, from ``data/<id>.zip``
  - ``id``              <- uuid5(NAMESPACE, problem_id), deterministic across machines/re-runs

Faithfulness notes
------------------
  1. Grading is on the FINAL ANSWER, not the notebook — Anthropic's write-up says the benchmark
     grades "on their final answer, rather than the path the model took to get there". The
     ``biomystery`` judge enforces that structurally: its correctness call never sees the notebook.
     Do NOT set ``judge: hypotest`` on these problems; that judge grades the procedure and would
     produce a different, non-comparable number.
  2. ``allowed_domains`` is recorded in metadata but NOT enforced here — hypotest has no per-problem
     network policy. The judge's anti-cheat check is what catches disallowed accession lookups,
     which is also how the benchmark itself defines the rule.
  3. Nothing in ``problems.csv`` beyond ``question`` may reach the agent: ``answer_rubric`` spells
     out the answer, and metadata is judge-side only (never rendered into the task prompt).

Usage:
    # 1. clone the dataset (requires accepting its gated terms) into capsules/biomysterybench/
    # 2. write the jsonl only (fast; capsules must already exist or be extracted later):
    .venv/bin/python scripts/data/convert_biomysterybench.py

    # extract data/<id>.zip -> capsules/biomysterybench/<id>/ as well (~145 GB):
    .venv/bin/python scripts/data/convert_biomysterybench.py --extract

    # a single problem, e.g. to try one end to end:
    .venv/bin/python scripts/data/convert_biomysterybench.py --extract --only hb002
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

REPO_ID = "Anthropic/BioMysteryBench-full"
UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, f"hypotest:{REPO_ID}")

# The benchmark is all-or-nothing: "Score 1.0 if the model did not cheat AND got the answer
# correct. Score 0 otherwise."
MAX_SCORE = 1

# Present in all 90 rubrics; used only to sanity-check that the loaded CSV is the v11 release.
SCORING_SENTENCE = "Score 1.0 if the model did not cheat AND got the answer correct. Score 0 otherwise."


def load_problems_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    required = {"id", "question", "answer_rubric", "allowed_domains", "human_solvable"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        sys.exit(f"{path} is missing expected column(s): {sorted(missing)}")
    return rows


def build_problem(row: dict[str, str]) -> dict[str, Any]:
    problem_id = row["id"].strip()
    return {
        "id": str(uuid.uuid5(UUID_NAMESPACE, problem_id)),
        "hypothesis": row["question"].strip(),
        # Empty on purpose — see "Faithfulness notes" above.
        "protocol": "",
        "answer": None,
        "rubric": row["answer_rubric"].strip(),
        "max_points": MAX_SCORE,
        "input_data_path": problem_id,
        "task_style": "question",
        "nb_primary_language": "python",
        # Graded by hypotest/env/judges/biomystery.py: correctness on the final answer alone,
        # ANDed with an anti-cheat check over the notebook. Binary, per the dataset's own rule.
        "judge": "biomystery",
        "metadata": {
            "source": REPO_ID,
            "problem_id": problem_id,
            # "yes" if at least one human benchmarker solved it. Anthropic reports the two splits
            # separately, so keep it for post-hoc slicing.
            "human_solvable": row["human_solvable"].strip(),
            # Recorded for provenance; hypotest does not enforce a per-problem network policy.
            "allowed_domains": [d.strip() for d in row["allowed_domains"].split(",") if d.strip()],
        },
    }


def extract_capsule(zip_path: Path, dest: Path, dry_run: bool = False) -> tuple[int, int]:
    """Unpack one problem's archive into its capsule dir. Returns (n_files, n_bytes)."""
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        total = sum(i.file_size for i in infos)
        if dry_run:
            return len(infos), total
        dest.mkdir(parents=True, exist_ok=True)
        zf.extractall(dest)
    return len(infos), total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "capsules" / "biomysterybench",
        help="Clone of Anthropic/BioMysteryBench-full (holds problems.csv and data/)",
    )
    ap.add_argument(
        "--capsule-dir",
        type=Path,
        default=None,
        help="Where to extract per-problem capsules (default: <dataset-dir>, flat like the other benchmarks)",
    )
    ap.add_argument(
        "--out-jsonl",
        type=Path,
        default=None,
        help="Output jsonl (default: <dataset-dir>/problems_biomysterybench.jsonl)",
    )
    ap.add_argument("--extract", action="store_true", help="Also unpack data/<id>.zip into the capsule dir")
    ap.add_argument("--only", nargs="*", default=None, help="Only these problem ids")
    ap.add_argument("--dry-run", action="store_true", help="Report what would be written/extracted")
    args = ap.parse_args()

    dataset_dir: Path = args.dataset_dir
    capsule_dir: Path = args.capsule_dir or dataset_dir
    out_jsonl: Path = args.out_jsonl or dataset_dir / "problems_biomysterybench.jsonl"

    csv_path = dataset_dir / "problems.csv"
    if not csv_path.exists():
        sys.exit(f"missing {csv_path} — clone {REPO_ID} into {dataset_dir} first")

    rows = load_problems_csv(csv_path)
    if args.only:
        wanted = set(args.only)
        rows = [r for r in rows if r["id"].strip() in wanted]
        if not rows:
            sys.exit(f"no problems matched --only {sorted(wanted)}")

    off_spec = [r["id"] for r in rows if not r["answer_rubric"].strip().endswith(SCORING_SENTENCE)]
    if off_spec:
        print(
            f"WARNING: {len(off_spec)} rubric(s) do not end with the v11 all-or-nothing scoring "
            f"sentence, e.g. {off_spec[:3]}. The biomystery judge scores binary regardless; check "
            f"you are on the v11 release."
        )

    problems = [build_problem(r) for r in rows]

    n_files = n_bytes = staged = missing = 0
    for p in problems:
        zip_path = dataset_dir / "data" / f"{p['input_data_path']}.zip"
        if not zip_path.exists():
            missing += 1
            print(f"[warn] {p['input_data_path']}: no {zip_path.name} — capsule will be empty")
            continue
        if not args.extract:
            continue
        files, size = extract_capsule(zip_path, capsule_dir / p["input_data_path"], args.dry_run)
        n_files += files
        n_bytes += size
        staged += 1
        print(f"[ok] {p['input_data_path']}: {files} file(s), {size / 1e6:.1f} MB")

    if not args.dry_run:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        out_jsonl.write_text("\n".join(json.dumps(p) for p in problems) + "\n", encoding="utf-8")

    solvable = sum(1 for p in problems if p["metadata"]["human_solvable"] == "yes")
    verb = "Would write" if args.dry_run else "Wrote"
    hard = len(problems) - solvable
    print(f"\n{verb} {len(problems)} problems ({solvable} human-solvable, {hard} hard) -> {out_jsonl}")
    if args.extract:
        print(f"Capsules -> {capsule_dir}/ ({staged} staged, {n_files} files, {n_bytes / 1e9:.1f} GB)")
    else:
        print(f"Capsules NOT staged — re-run with --extract to unpack data/*.zip into {capsule_dir}/")
    if missing:
        print(f"WARNING: {missing} problem(s) have no data/<id>.zip")

    print(
        "\nserver.yaml:\n"
        f"  capsule_dir: {capsule_dir}/\n"
        f"  problem_jsonl: {out_jsonl}\n"
        "  include_protocol: false\n"
        "\n(The judge is set per problem — see the `judge` field — so no server.yaml override is\n"
        "needed. Do NOT force `judge: hypotest`: that grades the notebook's procedure, which this\n"
        "benchmark deliberately does not.)\n"
    )


if __name__ == "__main__":
    main()
