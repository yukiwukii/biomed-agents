#!/usr/bin/env python3
"""Convert futurehouse/BixBench v1.5 into hypotest's ProblemInstance jsonl format.

The upstream dataset (https://huggingface.co/datasets/futurehouse/BixBench) ships as:

    BixBench.jsonl                    one row per QUESTION since the 2025-09-23 v1.5 reflatten
    CapsuleFolder-<uuid>.zip          that question's capsule, extracted here as CapsuleData-<uuid>/

``BixBench.jsonl`` columns:

  - ``id`` / ``question_id`` / ``short_id``   question identifiers ("bix-1-q1", "bix-1")
  - ``capsule_uuid`` / ``data_folder``        which capsule the question is asked about
  - ``question``                              the task prompt shown to the model
  - ``ideal``                                 the correct answer, short-form ("0.0002", "(1.50,1.54)")
  - ``distractors``                           three plausible but incorrect answers
  - ``eval_mode``                             str_verifier | range_verifier | llm_verifier
  - ``hypothesis`` / ``result`` / ``answer``  the CAPSULE-level hypothesis, the paper's finding, and
                                              whether the hypothesis holds — shared by every
                                              question on that capsule
  - ``categories`` / ``paper`` / ``canary``   provenance

Mapping onto hypotest
---------------------
One ProblemInstance per question (205 of them across 59 capsules), not per capsule:

  - ``hypothesis``      <- ``question``, verbatim. Despite the field name this is a question, which
                           is what ``task_style="question"`` tells the prompt builder and the judge.
  - ``protocol``        <- "" deliberately. Upstream shows the model the question and the capsule's
                           data files, nothing else; hypotest renders ``protocol`` into an
                           <objectives> block, so leaving it empty is the faithful choice. (An
                           empty protocol renders as prompts.NO_PROTOCOL_MSG.)
  - ``rubric``          <- the answer key, built from ``ideal`` + ``eval_mode`` + ``distractors``.
                           Judge-side only, never shown to the agent.
  - ``max_points``      <- 1. One question, one point; the judge also returns max_score=1.
  - ``answer``          <- None. The capsule-level accept/reject bool belongs to the hypothesis
                           framing of the dataset, not to these questions; it is kept in metadata.
  - ``task_style``      <- "question"
  - ``judge``           <- "bixbench" (hypotest/env/judges/bixbench.py)
  - ``input_data_path`` <- ``CapsuleData-<capsule_uuid>``. Several questions share one capsule, so
                           this cannot rely on the server's ``CapsuleData-<problem.id>`` fallback —
                           problem ids are per question, capsules are per paper.
  - ``id``              <- uuid5(NAMESPACE, question_id), deterministic across machines/re-runs.

``nb_primary_language`` is not in the upstream file, so it is taken from the
``EdisonScientific/bixbench_hypothesis`` dataset, which recorded it per capsule (51 of the 59;
the rest default to python). ``--no-hf`` skips that lookup entirely.

The upstream jsonl and the converted one are BOTH called ``bixbench.jsonl`` by convention — the
first because that is what the HF repo ships, the second because ``capsules/<bench>/<bench>.jsonl``
is where the dataset server looks. So when the input and output paths collide, the upstream file is
preserved as ``BixBench.jsonl`` (its name in the HF repo's own README) before being overwritten.

Faithfulness notes
------------------
  1. Grading is on the FINAL ANSWER, not the notebook. The ``bixbench`` judge enforces that
     structurally: neither of its calls sees the notebook. Do NOT set ``judge: hypotest`` on these
     problems; that judge grades the procedure and would produce a different, non-comparable number.
  2. Open-answer, not multiple choice. v1.5's release note calls open-answer the preferred setting,
     so ``distractors`` are passed to the judge as known-wrong alternatives and are never shown to
     the agent as options.
  3. Nothing in the upstream row beyond ``question`` may reach the agent: ``ideal`` is the answer,
     ``result`` states the paper's finding, and metadata is judge-side only (``ProblemInstance.
     metadata`` is never rendered into the task prompt).

Usage:
    # write the converted jsonl (capsules must already be extracted next to it):
    .venv/bin/python scripts/data/convert_bixbench.py

    # inspect without writing anything:
    .venv/bin/python scripts/data/convert_bixbench.py --dry-run

    # a single capsule or question, e.g. to try one end to end:
    .venv/bin/python scripts/data/convert_bixbench.py --only bix-1 --out-jsonl /tmp/one.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

REPO_ID = "futurehouse/BixBench"
UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, f"hypotest:{REPO_ID}")

# Where nb_primary_language comes from: the hypothesis-framed derivative of the same capsules.
LANGUAGE_DATASET = "EdisonScientific/bixbench_hypothesis"
DEFAULT_LANGUAGE = "python"

# One question, one point.
MAX_SCORE = 1

# Upstream columns this converter relies on.
REQUIRED_COLUMNS = {"id", "question", "ideal", "distractors", "capsule_uuid", "question_id", "eval_mode"}

# How each eval_mode compares a submission to `ideal`, spelled out for the judge's prompt.
EVAL_MODE_RULES = {
    "str_verifier": "The answer must be this value, at the precision the question asks for.",
    "range_verifier": "The correct answer is the interval above; the agent's value must fall inside it.",
    "llm_verifier": "The answer must be equivalent to this value; judge the meaning, not the wording.",
}


def build_rubric(row: dict[str, Any]) -> str:
    """The answer key, as judge-side text. Also what a human reads in score_info.json."""
    mode = str(row["eval_mode"])
    rule = EVAL_MODE_RULES.get(mode, EVAL_MODE_RULES["llm_verifier"])
    lines = [
        f"Correct answer: {row['ideal']}",
        "",
        f"Comparison mode: {mode}. {rule}",
        "Scoring is binary: 1 point for the correct answer, 0 otherwise.",
    ]
    if row.get("distractors"):
        lines += ["", "Plausible but INCORRECT answers:"]
        lines += [f"* {d}" for d in row["distractors"]]
    return "\n".join(lines)


def build_problem(row: dict[str, Any], language: str) -> dict[str, Any]:
    question_id = str(row["question_id"]).strip()
    capsule_uuid = str(row["capsule_uuid"]).strip()
    categories = row.get("categories") or ""
    return {
        "id": str(uuid.uuid5(UUID_NAMESPACE, question_id)),
        "hypothesis": str(row["question"]).strip(),
        # Empty on purpose — see "Mapping onto hypotest" above.
        "protocol": "",
        # The capsule-level accept/reject bool is not this question's answer; see metadata.
        "answer": None,
        "rubric": build_rubric(row),
        "max_points": MAX_SCORE,
        "input_data_path": f"CapsuleData-{capsule_uuid}",
        "task_style": "question",
        "nb_primary_language": language,
        # Graded by hypotest/env/judges/bixbench.py: the final answer against `ideal`, per eval_mode.
        "judge": "bixbench",
        "metadata": {
            "source": REPO_ID,
            "version": str(row.get("version", "")),
            "short_id": str(row.get("short_id", "")),
            "question_id": question_id,
            "capsule_uuid": capsule_uuid,
            # The answer key, in the structured form the judge's verifiers consume.
            "eval_mode": str(row["eval_mode"]),
            "ideal": str(row["ideal"]),
            "distractors": [str(d) for d in row.get("distractors") or []],
            # Capsule-level context, shared by every question on this capsule. `capsule_result`
            # states the paper's finding, so like `ideal` it is judge-side only.
            "capsule_hypothesis": str(row.get("hypothesis", "")),
            "capsule_result": str(row.get("result", "")),
            "capsule_hypothesis_accepted": row.get("answer"),
            "categories": [c.strip() for c in categories.split(",") if c.strip()],
            "paper": str(row.get("paper", "")),
            "data_folder": str(row.get("data_folder", "")),
            "canary": str(row.get("canary", "")),
            "upstream_id": str(row.get("id", "")),
        },
    }


def load_upstream(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        sys.exit(f"{path} is empty")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        sys.exit(
            f"{path} is missing expected column(s): {sorted(missing)}. "
            f"This converter reads the flattened v1.5 layout (one row per question); the v1.0 tag "
            f"nests questions inside each capsule and is not supported."
        )
    return rows


def load_language_map() -> dict[str, str]:
    """Capsule uuid -> notebook language, from the hypothesis-framed derivative of these capsules."""
    try:
        from datasets import load_dataset  # noqa: PLC0415  (optional, and slow to import)

        ds = load_dataset(LANGUAGE_DATASET)["train"]
    except Exception as e:  # noqa: BLE001 - offline/no-auth is expected; fall back to the default
        print(f"[warn] could not read {LANGUAGE_DATASET} ({type(e).__name__}: {e});")
        print(f"[warn] every problem will get nb_primary_language={DEFAULT_LANGUAGE!r}. Use --no-hf to silence.")
        return {}
    return {
        str(r["input_data_path"]).removeprefix("CapsuleFolder-").removesuffix(".zip"): r["nb_primary_language"]
        for r in ds
        if r.get("nb_primary_language")
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=ROOT / "capsules" / "bixbench",
                    help="Clone of futurehouse/BixBench (holds the jsonl and the CapsuleData-* dirs)")
    ap.add_argument("--in-jsonl", type=Path, default=None,
                    help="Upstream jsonl (default: <dataset-dir>/BixBench.jsonl, else bixbench.jsonl)")
    ap.add_argument("--out-jsonl", type=Path, default=None,
                    help="Converted jsonl (default: <dataset-dir>/bixbench.jsonl)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only these short_ids, question_ids or capsule uuids")
    ap.add_argument("--no-hf", action="store_true",
                    help=f"Skip the {LANGUAGE_DATASET} lookup; everything gets {DEFAULT_LANGUAGE}")
    ap.add_argument("--dry-run", action="store_true", help="Report what would be written")
    args = ap.parse_args()

    dataset_dir: Path = args.dataset_dir
    out_jsonl: Path = args.out_jsonl or dataset_dir / "bixbench.jsonl"

    in_jsonl: Path | None = args.in_jsonl
    if in_jsonl is None:
        for candidate in (dataset_dir / "BixBench.jsonl", dataset_dir / "bixbench.jsonl"):
            if candidate.exists():
                in_jsonl = candidate
                break
    if in_jsonl is None or not in_jsonl.exists():
        sys.exit(f"missing BixBench.jsonl — download {REPO_ID} into {dataset_dir} first")

    rows = load_upstream(in_jsonl)
    if args.only:
        wanted = set(args.only)
        rows = [r for r in rows if wanted & {r["short_id"], r["question_id"], r["capsule_uuid"]}]
        if not rows:
            sys.exit(f"no questions matched --only {sorted(wanted)}")

    languages = {} if args.no_hf else load_language_map()
    problems = [build_problem(r, languages.get(str(r["capsule_uuid"]), DEFAULT_LANGUAGE)) for r in rows]

    seen_ids = Counter(p["id"] for p in problems)
    if dupes := [i for i, n in seen_ids.items() if n > 1]:
        sys.exit(f"question_id is not unique — {len(dupes)} colliding problem id(s), e.g. {dupes[:3]}")

    missing_capsules = sorted({p["input_data_path"] for p in problems if not (dataset_dir / p["input_data_path"]).is_dir()})
    for name in missing_capsules:
        print(f"[warn] no {name}/ under {dataset_dir} — extract CapsuleFolder-*.zip, or the server will fail on it")

    if not args.dry_run:
        # Both files are conventionally named bixbench.jsonl (see the module docstring); keep the
        # upstream one under its HF-repo name rather than overwriting it with the conversion.
        if in_jsonl.resolve() == out_jsonl.resolve():
            preserved = dataset_dir / "BixBench.jsonl"
            shutil.copy2(in_jsonl, preserved)
            print(f"Preserved upstream {in_jsonl.name} -> {preserved}")
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        out_jsonl.write_text("\n".join(json.dumps(p) for p in problems) + "\n")

    capsules = {p["input_data_path"] for p in problems}
    modes = Counter(p["metadata"]["eval_mode"] for p in problems)
    langs = Counter(p["nb_primary_language"] for p in problems)
    verb = "Would write" if args.dry_run else "Wrote"
    print(f"\n{verb} {len(problems)} questions across {len(capsules)} capsules -> {out_jsonl}")
    print(f"  eval_mode: {dict(modes)}")
    print(f"  nb_primary_language: {dict(langs)}" + ("" if languages else f"  (defaulted; no {LANGUAGE_DATASET})"))
    if missing_capsules:
        print(f"WARNING: {len(missing_capsules)} capsule dir(s) missing")

    print(
        "\nserver.yaml:\n"
        f"  capsule_dir: {dataset_dir}/\n"
        f"  problem_jsonl: {out_jsonl}\n"
        "  include_protocol: false\n"
        "\n(The judge is set per problem — see the `judge` field — so no server.yaml override is\n"
        "needed. Do NOT force `judge: hypotest`: that grades the notebook's procedure, which this\n"
        "benchmark deliberately does not.)\n"
    )


if __name__ == "__main__":
    main()
