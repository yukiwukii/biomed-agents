#!/usr/bin/env python3
"""Build a COMBINED NeMo Gym problem source + deterministic train/eval split for
"train on bixbench-hypothesis, evaluate on hypotest".

Why this is separate from make_splits.py:
    make_splits.py takes ONE problem source and RANDOMLY splits it into disjoint
    train/eval. Here train and eval are two DIFFERENT capsule sets that must be
    served by ONE dataset server (task_idx is positional into a single problem
    list -- Dataset.get_new_env_by_idx). So we concatenate them into one problem
    list and split DETERMINISTICALLY by construction:

        task_idx 0 .. 249  -> bixbench-hypothesis  (train, 250)
        task_idx 250 .. 300 -> hypotest            (eval, 51 judgeable)

    capsule_dir is set to the capsules/ PARENT so a single input_data_path can
    reach into either subdir:
        bixbench:  input_data_path = "bixbench-hypothesis/capsule_<id>"
        hypotest:  input_data_path = "hypotest/CapsuleData-<id>"
    get_new_env_by_idx does `capsule_dir / input_data_path`, so both resolve.

Eval membership: only the 51 hypotest capsules that have rubric/answer metadata
in the HF `EdisonScientific/bixbench_hypothesis` set are usable. The other 14
CapsuleData-* dirs have no rubric anywhere and would return reward 0.0 unjudged,
so they are DROPPED (recorded in the manifest).

Outputs (into rl/data/):
    combined_bixbench_hypotest.jsonl   the 301-problem source (server.train.yaml)
    train_bixbench.jsonl               250 pointers, task_idx 0..249
    eval_hypotest.jsonl                51 pointers,  task_idx 250..300
    manifest_bixbench_hypotest.json    provenance
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datasets import load_dataset  # noqa: E402

from hypotest.env.interpreter_env import ProblemInstance  # noqa: E402

CAPSULES = REPO_ROOT / "capsules"
OUT = Path(__file__).resolve().parent
AGENT_NAME = "hypotest_agent"
HF_EVAL = "EdisonScientific/bixbench_hypothesis"


def gym_line(task_idx: int) -> dict:
    return {
        "task_idx": task_idx,
        "responses_create_params": {"input": []},
        "agent_ref": {"type": "responses_api_agents", "name": AGENT_NAME},
    }


def main() -> None:
    combined: list[dict] = []
    train_ids: list[str] = []
    eval_ids: list[str] = []

    # ---- TRAIN: 250 bixbench-hypothesis (local jsonl, full metadata) ---------
    bb_path = CAPSULES / "bixbench-hypothesis" / "bixbench-hypothesis.jsonl"
    for ln in bb_path.read_text().splitlines():
        if not ln.strip():
            continue
        d = json.loads(ln)
        orig = d["input_data_path"]  # "capsule_<id>"
        assert orig.startswith("capsule_"), f"unexpected bixbench input_data_path: {orig}"
        d["input_data_path"] = f"bixbench-hypothesis/{orig}"
        cap = CAPSULES / d["input_data_path"]
        assert cap.is_dir(), f"missing train capsule: {cap}"
        ProblemInstance.model_validate(d)  # re-validate through the server's schema
        combined.append(d)
        train_ids.append(d["id"])

    # ---- EVAL: 51 hypotest capsules that HAVE HF metadata --------------------
    ds = load_dataset(HF_EVAL, split="train")
    for row in ds:
        d = dict(row)
        cid = d["id"]
        d["input_data_path"] = f"hypotest/CapsuleData-{cid}"
        cap = CAPSULES / d["input_data_path"]
        assert cap.is_dir(), f"missing eval capsule: {cap}"
        ProblemInstance.model_validate(d)
        combined.append(d)
        eval_ids.append(cid)

    # ---- disjointness (belt and braces; we already checked ids don't overlap)
    overlap = set(train_ids) & set(eval_ids)
    assert not overlap, f"train/eval overlap: {len(overlap)}"

    n_train, n_eval = len(train_ids), len(eval_ids)
    assert n_train == 250 and n_eval == 51, f"counts changed: train={n_train} eval={n_eval}"

    # ---- write combined source (task_idx = position in this file) ------------
    # In sources/ (NOT rl/data/ top level): it is ~690 KB and read by the env
    # pod directly from the PVC via server.train.yaml's absolute problem_jsonl
    # path -- it must NOT be swept into the hypotest-rl-splits configmap by
    # workloads.sh (globs rl/data/*.jsonl), whose kubectl-apply annotation caps
    # at 256 KB. Only the small pointer files below belong in that configmap.
    src_dir = OUT / "sources"
    src_dir.mkdir(exist_ok=True)
    src = src_dir / "combined_bixbench_hypotest.jsonl"
    with src.open("w") as f:
        for d in combined:
            f.write(json.dumps(d) + "\n")

    # ---- deterministic pointer splits ----------------------------------------
    with (OUT / "train_bixbench.jsonl").open("w") as f:
        for i in range(0, n_train):  # 0 .. 249
            f.write(json.dumps(gym_line(i)) + "\n")
    with (OUT / "eval_hypotest.jsonl").open("w") as f:
        for i in range(n_train, n_train + n_eval):  # 250 .. 300
            f.write(json.dumps(gym_line(i)) + "\n")

    manifest = {
        "purpose": "train on bixbench-hypothesis (250), eval on hypotest (51 judgeable)",
        "capsule_dir": str(CAPSULES),
        "problem_jsonl": str(src),
        "agent_name": AGENT_NAME,
        "counts": {"train": n_train, "eval": n_eval, "total": n_train + n_eval},
        "task_idx_ranges": {"train": [0, n_train - 1], "eval": [n_train, n_train + n_eval - 1]},
        "eval_dropped_no_rubric": 65 - n_eval,
        "train_source": str(bb_path),
        "eval_source": HF_EVAL,
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "note": (
            "task_idx is positional into combined_bixbench_hypotest.jsonl as served by "
            "the dataset server. Regenerate if either capsule set or its ordering changes."
        ),
    }
    (OUT / "manifest_bixbench_hypotest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"OK: combined={n_train + n_eval}  train={n_train} (idx 0..{n_train-1})  "
          f"eval={n_eval} (idx {n_train}..{n_train + n_eval - 1})  dropped_no_rubric={65 - n_eval}")
    print(f"    -> {src.name}, train_bixbench.jsonl, eval_hypotest.jsonl, manifest_bixbench_hypotest.json")


if __name__ == "__main__":
    main()
