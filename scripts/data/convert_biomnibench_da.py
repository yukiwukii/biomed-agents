#!/usr/bin/env python3
"""Convert phylobio/BiomniBench-DA into hypotest's ProblemInstance jsonl + capsule format.

BiomniBench-DA ships 50 task folders (``da-{paper}-{task}/``), each with:

  - ``instruction.md``: an open research question ("characterize X", "compare Y")
    plus a description of the data files, plus a "Required Outputs" section asking
    for ``trace.md``/``answer.txt`` files (Harbor-sandbox-specific submission
    mechanics that don't apply here — see below).
  - ``task.toml``: Harbor sandbox config (difficulty/category/task_type metadata,
    cpu/mem/timeouts) — not needed by hypotest, which has its own ExecutionConfig.
  - ``tests/rubric.txt``: an expert-authored, multi-criterion rubric with A/B/C point
    levels summing to "Total Points: 100/100" (verified consistent across all 50
    tasks). No accept/reject ground truth exists anywhere in the task data — grading
    is rubric-only.
  - ``environment/data/*``: the raw dataset files referenced by instruction.md.

This script maps each task folder onto one ``ProblemInstance`` (see
``hypotest.env.interpreter_env``) with ``task_style="question"`` (Patch 16 in
patches.md — an additive framework change so open research-question tasks get an
accurate grading prompt instead of being force-fit into the accept/reject-hypothesis
framing that ``task_style="hypothesis"`` uses):

  - ``hypothesis``       <- the "## Question" section (the research question itself)
  - ``protocol``         <- the "## Data Files" section (column/format documentation
                            the agent needs — NOT a step-by-step protocol, but that's
                            the field hypotest renders into the task's <objectives>
                            block, and this is the right content to put there)
  - ``rubric``            <- tests/rubric.txt, verbatim (already in the same
                             "criteria with integer point scores" shape the grading
                             prompt asks the judge model to output)
  - ``max_score``         <- 100 (parsed from "Total Points: X/100")
  - ``input_data_path``   <- the task id (e.g. "da-1-3"); environment/data/* is
                             copied into <capsule_dir>/<task_id>/ verbatim (flat,
                             matching how instruction.md references file basenames)
  - ``id``                <- uuid5(NAMESPACE, task_id) — deterministic, reproducible
  - ``metadata``           <- source dataset name, task id, difficulty/category/
                              task_type from task.toml, and the full instruction.md
                              (for reference / debugging)

The "Required Outputs" section of instruction.md (trace.md/answer.txt file-writing
instructions) is deliberately dropped: hypotest already captures the full notebook
trajectory as the trace (fed verbatim into the grading prompt) and the
``submit_answer`` tool call as the final answer, so those Harbor-specific submission
mechanics don't apply and would only distract the agent into writing files that are
never read.

Usage:
    .venv/bin/python scripts/data/convert_biomnibench_da.py \
        --out-jsonl problems_biomnibench_da.jsonl --capsule-dir capsules_biomnibench_da/

    # re-use an already-downloaded snapshot instead of hitting the Hub again:
    .venv/bin/python scripts/data/convert_biomnibench_da.py --snapshot-dir /path/to/snapshot

    # only convert a subset while iterating:
    .venv/bin/python scripts/data/convert_biomnibench_da.py --limit 3
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

ROOT = Path(__file__).resolve().parents[2]

REPO_ID = "phylobio/BiomniBench-DA"
# Deterministic namespace so re-running the conversion (or converting on another
# machine) always yields the same problem ids for the same task id.
UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, f"hypotest:{REPO_ID}")

# Top-level "## " headers that can appear in instruction.md, in the order observed
# across all 50 released tasks (an optional "## Background" before "## Question").
# Used only to bound section extraction — see extract_section().
KNOWN_HEADERS = ("Background", "Question", "Data Files", "Required Outputs", "Environment")


def load_env(path: Path = ROOT / ".env") -> None:
    """Populate os.environ from a KEY=VALUE .env file (does not overwrite)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def extract_section(text: str, header: str, stop_headers: tuple[str, ...]) -> str:
    """Return the body of a top-level "## <header>" section, up to the next known header."""
    m = re.search(rf"^## {re.escape(header)}\s*\n", text, re.MULTILINE)
    if not m:
        return ""
    rest = text[m.end() :]
    stops = [sm.start() for h in stop_headers if (sm := re.search(rf"^## {re.escape(h)}\b", rest, re.MULTILINE))]
    end = min(stops) if stops else len(rest)
    return rest[:end].strip()


def parse_max_score(rubric_text: str) -> int:
    m = re.search(r"Total Points:\s*\d+\s*/\s*(\d+)", rubric_text)
    if not m:
        raise ValueError("could not find 'Total Points: X/Y' in rubric.txt")
    return int(m.group(1))


def load_toml_metadata(task_toml_path: Path) -> dict:
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]
    data = tomllib.loads(task_toml_path.read_text())
    return data.get("metadata", {})


def build_problem(task_id: str, task_dir: Path, capsule_dir: Path) -> dict:
    instruction = (task_dir / "instruction.md").read_text()
    rubric_text = (task_dir / "tests" / "rubric.txt").read_text()

    question = extract_section(instruction, "Question", KNOWN_HEADERS)
    data_files = extract_section(instruction, "Data Files", KNOWN_HEADERS)
    background = extract_section(instruction, "Background", KNOWN_HEADERS)
    if not question:
        raise ValueError(f"{task_id}: could not extract '## Question' section")

    protocol_parts = []
    if background:
        protocol_parts.append(f"## Background\n{background}")
    if data_files:
        protocol_parts.append(f"## Data Files\n{data_files}")
    protocol = "\n\n".join(protocol_parts)

    max_score = parse_max_score(rubric_text)
    toml_meta = load_toml_metadata(task_dir / "task.toml")

    src_data_dir = task_dir / "environment" / "data"
    dest_data_dir = capsule_dir / task_id
    if dest_data_dir.exists():
        shutil.rmtree(dest_data_dir)
    try:
        # Hardlink instead of copying bytes when possible — some of these tasks' raw
        # count matrices run into the tens of GB, and capsule contents are never
        # modified in place (hypotest copies them again per-run into a throwaway
        # work_dir), so sharing inodes with the snapshot is safe. Only works when the
        # snapshot and capsule_dir are on the same filesystem (os.link can't cross
        # devices) — default --cache-dir is under this repo's ROOT for that reason;
        # pass --snapshot-dir pointing elsewhere and this silently falls back to a
        # real copy.
        shutil.copytree(src_data_dir, dest_data_dir, copy_function=os.link)
    except OSError:
        # copytree may have partially populated dest_data_dir (e.g. an empty top-level
        # dir from os.makedirs) before the cross-device link error surfaced — clear it
        # so the fallback's own os.makedirs doesn't raise FileExistsError.
        shutil.rmtree(dest_data_dir, ignore_errors=True)
        shutil.copytree(src_data_dir, dest_data_dir)

    return {
        "id": str(uuid.uuid5(UUID_NAMESPACE, task_id)),
        "hypothesis": question,
        "protocol": protocol,
        "answer": None,
        "rubric": rubric_text,
        "max_points": max_score,
        "input_data_path": task_id,
        "task_style": "question",
        "nb_primary_language": "python",
        "metadata": {
            "source": REPO_ID,
            "task_id": task_id,
            "difficulty": toml_meta.get("difficulty"),
            "category": toml_meta.get("category"),
            "task_type": toml_meta.get("task_type"),
            "instruction_md": instruction,
        },
    }


def download_snapshot(dest: Path, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(dest),
        token=token,
        allow_patterns=["da-*/instruction.md", "da-*/task.toml", "da-*/tests/rubric.txt", "da-*/environment/data/**"],
    )
    return Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot-dir", type=Path, default=None, help="Reuse an already-downloaded HF snapshot instead of re-downloading")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "biomnibench-da", help="Where to download the snapshot (if not using --snapshot-dir)")
    ap.add_argument("--out-jsonl", type=Path, default=ROOT / "problems_biomnibench_da.jsonl")
    ap.add_argument("--capsule-dir", type=Path, default=ROOT / "capsules_biomnibench_da")
    ap.add_argument("--limit", type=int, default=None, help="Only convert the first N tasks")
    ap.add_argument("--only", default=None, help="Only tasks whose id contains this substring")
    args = ap.parse_args()

    load_env()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")

    if args.snapshot_dir is not None:
        base = args.snapshot_dir
    else:
        print(f"Downloading {REPO_ID} -> {args.cache_dir} ...")
        base = download_snapshot(args.cache_dir, token)

    task_dirs = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("da-"))
    if args.only:
        task_dirs = [p for p in task_dirs if args.only in p.name]
    if args.limit is not None:
        task_dirs = task_dirs[: args.limit]
    if not task_dirs:
        sys.exit(f"no task folders found under {base}")

    args.capsule_dir.mkdir(parents=True, exist_ok=True)

    problems = []
    for task_dir in task_dirs:
        task_id = task_dir.name
        try:
            problems.append(build_problem(task_id, task_dir, args.capsule_dir))
            print(f"[ok] {task_id}")
        except Exception as e:  # noqa: BLE001 — keep converting the rest, report failures
            print(f"[error] {task_id}: {type(e).__name__}: {e}")

    with args.out_jsonl.open("w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")

    print(f"\nWrote {len(problems)}/{len(task_dirs)} problems -> {args.out_jsonl}")
    print(f"Capsules -> {args.capsule_dir}")


if __name__ == "__main__":
    main()
