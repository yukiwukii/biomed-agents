#!/usr/bin/env python3
"""Build NeMo Gym rollout-input JSONL splits for GRPO training on hypotest.

NeMo Gym's training examples are not the problems themselves — they are pointers.
Each line is::

    {
        "task_idx": 12,
        "responses_create_params": {"input": []},
        "agent_ref": {"type": "responses_api_agents", "name": "hypotest_agent"},
    }

``task_idx`` is fed straight to ``AviarySeedSessionRequest.task_idx`` and ends up
at ``Dataset.get_new_env_by_idx(idx)`` on the dataset server, so it indexes *the
running server's problem list*, positionally.

That makes ordering safety-critical: a split built against a different problem
source, a different ``max_problems``, or a reordered JSONL will silently train on
the wrong tasks. So rather than re-deriving the order (which is what
``bbh/scripts/03_make_splits.py`` does, and why it needs a sha256 manifest to
detect drift), this script imports ``hypotest.dataset_server`` and calls the very
same ``DatasetConfig.load_problems()`` the server calls. The ordering cannot
disagree, because it is the same code path.

Usage::

    # from server.yaml (the normal case)
    python rl/data/make_splits.py --server-config server.yaml

    # hold out 20% for validation instead of the default 10%
    python rl/data/make_splits.py --server-config server.yaml --eval-frac 0.2

Writes ``train.jsonl``, ``eval.jsonl``, ``tiny.jsonl`` and ``manifest.json`` into
``--out-dir`` (default: the directory holding this script).

``tiny.jsonl`` is a deterministic handful of training tasks for plumbing smoke
tests (rl/scripts/00_smoke_rollout.sh); it is disjoint from neither split by
design — it is a *subset of train*, meant for "does the loop run at all".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hypotest.dataset_server import DatasetConfig  # noqa: E402

DEFAULT_AGENT_NAME = "hypotest_agent"
DEFAULT_SPLIT_SEED = 20260804
DEFAULT_EVAL_FRAC = 0.1
DEFAULT_N_TINY = 4


def eprint(*a: object) -> None:
    print(*a, file=sys.stderr)


def die(msg: str) -> None:
    eprint(f"\033[31mFATAL: {msg}\033[0m")
    raise SystemExit(1)


def gym_line(task_idx: int, agent_name: str) -> dict[str, Any]:
    """One NeMo Gym training example.

    ``responses_create_params.input`` is empty because the aviary resources
    server seeds the conversation itself: ``/seed_session`` returns the task
    description built by ``InterpreterEnv.reset()``. NeMo RL never constructs a
    prompt for this environment.
    """
    return {
        "task_idx": task_idx,
        "responses_create_params": {"input": []},
        "agent_ref": {"type": "responses_api_agents", "name": agent_name},
    }


def emit_jsonl(path: Path, task_indices: list[int], agent_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ti in task_indices:
            f.write(json.dumps(gym_line(ti, agent_name)) + "\n")
    eprint(f">> wrote {len(task_indices):4d} examples -> {path}")


def load_dataset_config(
    server_config_path: Path,
    capsule_dir: Path | None = None,
    max_problems: int | None = None,
    clear_max_problems: bool = False,
) -> DatasetConfig:
    """Build the DatasetConfig exactly as ``dataset_server.launch_server`` does.

    The overrides exist so the splits can be generated from the *deployment*
    server config (rl/runai/server.train.yaml) while running on a machine where
    that config's container paths do not exist. `capsule_dir` is a pydantic
    DirectoryPath and is validated for existence, so it must be redirected;
    `work_dir`/`save_dir` are created on validation, so they are dropped.

    Only fields that cannot affect problem *ordering* may be overridden here.
    `problem_jsonl` / `hf_dataset` / `max_problems` are what determine order and
    membership, which is why max_problems is an explicit, visible flag rather
    than something quietly adjusted.
    """
    raw = yaml.safe_load(server_config_path.read_text(encoding="utf-8"))
    if "dataset" not in raw:
        die(f"{server_config_path} has no top-level `dataset:` key.")
    ds = dict(raw["dataset"])

    if capsule_dir is not None:
        ds["capsule_dir"] = str(capsule_dir)
    # Never create the deployment's directories on the machine building splits.
    ds.pop("work_dir", None)
    ds.pop("save_dir", None)

    if clear_max_problems:
        ds.pop("max_problems", None)
    if max_problems is not None:
        ds["max_problems"] = max_problems

    return DatasetConfig.model_validate(ds)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--server-config",
        type=Path,
        default=REPO_ROOT / "server.yaml",
        help="server.yaml whose `dataset:` block defines the problem source and ordering.",
    )
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument(
        "--agent-name", default=DEFAULT_AGENT_NAME, help="Must match the agent block name in the Gym config."
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    p.add_argument("--eval-frac", type=float, default=DEFAULT_EVAL_FRAC)
    p.add_argument("--n-tiny", type=int, default=DEFAULT_N_TINY)
    p.add_argument(
        "--capsule-dir",
        type=Path,
        default=None,
        help="Override the config's capsule_dir. Needed when generating splits from a deployment "
        "config (e.g. rl/runai/server.train.yaml) whose container paths do not exist locally.",
    )
    p.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Override the config's max_problems. Changes which problems exist, so it changes task_idx.",
    )
    p.add_argument(
        "--all-problems",
        action="store_true",
        help="Drop max_problems entirely. Use this when the repo server.yaml caps the dataset for local testing "
        "but the deployment does not.",
    )
    args = p.parse_args()

    if not args.server_config.exists():
        die(f"{args.server_config} not found.")
    if not 0.0 <= args.eval_frac < 1.0:
        die(f"--eval-frac must be in [0, 1); got {args.eval_frac}")

    cfg = load_dataset_config(
        args.server_config,
        capsule_dir=args.capsule_dir,
        max_problems=args.max_problems,
        clear_max_problems=args.all_problems,
    )
    problems = cfg.load_problems()
    if not problems:
        die("Dataset loaded zero problems. Check `problem_jsonl` / `hf_dataset` in the server config.")

    n = len(problems)
    source = cfg.hf_dataset or str(cfg.problem_jsonl)
    eprint(f">> problem source: {source}")
    eprint(f">> problems loaded: {n}" + (f"  (max_problems={cfg.max_problems})" if cfg.max_problems else ""))

    ids = [str(prob.id) for prob in problems]
    if len(set(ids)) != n:
        dupes = [i for i in set(ids) if ids.count(i) > 1]
        die(f"duplicate problem ids in the dataset ({len(dupes)}). task_idx would be ambiguous. Sample: {dupes[:3]}")

    # Split on shuffled positions. Held-out eval must be disjoint from train, or
    # the GRPO validation curve measures memorization rather than generalization.
    indices = list(range(n))
    rng = random.Random(args.seed)
    rng.shuffle(indices)

    n_eval = round(n * args.eval_frac)
    if args.eval_frac > 0 and n_eval == 0:
        n_eval = 1
    if n_eval >= n:
        die(f"--eval-frac {args.eval_frac} would leave no training tasks ({n_eval}/{n}).")

    eval_indices = sorted(indices[:n_eval])
    train_indices = sorted(indices[n_eval:])
    tiny_indices = train_indices[: min(args.n_tiny, len(train_indices))]

    overlap = set(train_indices) & set(eval_indices)
    if overlap:
        die(f"train/eval overlap ({len(overlap)} indices) — bug in this script, refusing to write.")

    emit_jsonl(args.out_dir / "train.jsonl", train_indices, args.agent_name)
    emit_jsonl(args.out_dir / "eval.jsonl", eval_indices, args.agent_name)
    emit_jsonl(args.out_dir / "tiny.jsonl", tiny_indices, args.agent_name)

    # The manifest exists so a rollout run can be checked against the split that
    # produced it. `problem_ids` is the authoritative record: if the server's
    # problem source changes, these ids stop matching the task_idx positions and
    # the splits must be regenerated.
    manifest = {
        "split_seed": args.seed,
        "agent_name": args.agent_name,
        "server_config": str(args.server_config),
        "server_config_sha256": hashlib.sha256(args.server_config.read_bytes()).hexdigest(),
        "problem_source": source,
        "max_problems": cfg.max_problems,
        "n_problems": n,
        "eval_frac": args.eval_frac,
        "counts": {"train": len(train_indices), "eval": len(eval_indices), "tiny": len(tiny_indices)},
        "overlap_train_eval": 0,
        "problem_ids": {
            "train": [ids[i] for i in train_indices],
            "eval": [ids[i] for i in eval_indices],
            "tiny": [ids[i] for i in tiny_indices],
        },
        "note": (
            "task_idx indexes the running dataset server's problem list positionally "
            "(Dataset.get_new_env_by_idx). Regenerate these splits whenever the server's "
            "dataset source, max_problems, or problem ordering changes."
        ),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    eprint(f">> wrote {manifest_path}")

    print(f"OK: train={len(train_indices)} eval={len(eval_indices)} tiny={len(tiny_indices)}; zero train/eval overlap")


if __name__ == "__main__":
    main()
