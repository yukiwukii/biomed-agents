"""Compare pass@k and avg@k between a benchmark run and the forks taken from it.

This is the headline fork metric: how much of the original failure was
recoverable once the agent was rewound to its first wrong step and handed the
judge's feedback.

Usage:
    python scripts/fork/compare_rewards.py \\
        --rewards  benchmark_results/rewards.json \\
        --fork-summary forks/fork_summary.json

``--rewards`` is the ``rewards.json`` written by ``benchmark_agent.py``; it is a
flat map of ``task_<i>_rep<j>`` to score. ``--fork-summary`` is the
``fork_summary.json`` written by ``fork_trajectory.py``. A trajectory with
several forks contributes the mean of their ``new_score``.

k is inferred from the data (the number of replications per task), so this works
for any pass@k, not just pass@3.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_rewards(rewards_path: Path) -> dict[str, float]:
    with open(rewards_path) as f:
        return json.load(f)


def _replications_per_task(rewards: dict[str, float]) -> int:
    """The k in pass@k: the most common number of replications per task.

    Reported rather than assumed, because a run whose replications partly failed
    would otherwise be labelled with a k it does not have.
    """
    tasks: dict[str, int] = defaultdict(int)
    for traj_id in rewards:
        tasks[traj_id.rsplit("_rep", 1)[0]] += 1
    if not tasks:
        return 0
    return max(set(tasks.values()), key=list(tasks.values()).count)


def compute_metrics(rewards: dict[str, float]) -> tuple[float, float]:
    """Return (pass@3, avg@3) for a flat rewards dict keyed by task_N_repM."""
    tasks: dict[str, list[float]] = defaultdict(list)
    for traj_id, score in rewards.items():
        task_id = traj_id.rsplit("_rep", 1)[0]
        tasks[task_id].append(score)

    n = len(tasks)
    pass3 = sum(1 for reps in tasks.values() if max(reps) == 1.0) / n
    avg3 = sum(sum(reps) / len(reps) for reps in tasks.values()) / n
    return pass3, avg3


def merge_fork_rewards(
    orig_rewards: dict[str, float],
    fork_summary_path: Path,
) -> dict[str, float]:
    """Build a merged rewards dict using new_score for forked trajectories."""
    with open(fork_summary_path) as f:
        fork_data = json.load(f)

    # Collect new_scores per trajectory (multiple forks possible)
    fork_scores: dict[str, list[float]] = defaultdict(list)
    for entry in fork_data["forked"]:
        fork_scores[entry["source_traj_id"]].append(entry["new_score"])

    merged = dict(orig_rewards)
    for traj_id, scores in fork_scores.items():
        merged[traj_id] = sum(scores) / len(scores)

    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--rewards",
        type=Path,
        required=True,
        help="rewards.json from the benchmark run that was forked",
    )
    ap.add_argument(
        "--fork-summary",
        type=Path,
        default=None,
        help="fork_summary.json from the fork run. Omit to report the baseline only.",
    )
    args = ap.parse_args()

    if not args.rewards.is_file():
        raise SystemExit(f"no rewards.json at {args.rewards}")

    orig_rewards = load_rewards(args.rewards)
    orig_pass, orig_avg = compute_metrics(orig_rewards)
    k = _replications_per_task(orig_rewards)

    print("=== Original ===")
    print(f"  pass@{k}: {orig_pass:.4f}")
    print(f"  avg@{k}:  {orig_avg:.4f}")

    if args.fork_summary is None:
        return
    if not args.fork_summary.is_file():
        raise SystemExit(f"no fork_summary.json at {args.fork_summary}")

    merged = merge_fork_rewards(orig_rewards, args.fork_summary)
    fork_pass, fork_avg = compute_metrics(merged)
    n = len({key.rsplit("_rep", 1)[0] for key in merged})

    print("\n=== After forking ===")
    print(f"  pass@{k}: {fork_pass:.4f}  ({round(fork_pass * n)}/{n}, Δ{fork_pass - orig_pass:+.4f})")
    print(f"  avg@{k}:  {fork_avg:.4f}  (Δ{fork_avg - orig_avg:+.4f})")


if __name__ == "__main__":
    main()
