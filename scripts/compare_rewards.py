"""Compare pass@3 and avg@3 between original rewards and forked results."""

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def load_rewards(rewards_path: Path) -> dict[str, float]:
    with open(rewards_path) as f:
        return json.load(f)


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
    orig_rewards = load_rewards(REPO_ROOT / "benchmark_results_hypotest" / "rewards.json")
    fork_summary_path = REPO_ROOT / "forks-w" / "fork_summary.json"

    orig_pass3, orig_avg3 = compute_metrics(orig_rewards)

    print("=== Original ===")
    print(f"  pass@3: {orig_pass3:.4f}")
    print(f"  avg@3:  {orig_avg3:.4f}")

    if fork_summary_path.exists():
        merged = merge_fork_rewards(orig_rewards, fork_summary_path)
        fork_pass3, fork_avg3 = compute_metrics(merged)
        n = len({k.rsplit("_rep", 1)[0] for k in merged})

        n_pass_orig = round(orig_pass3 * n)
        n_pass_fork = round(fork_pass3 * n)

        print("\n=== After Forking ===")
        print(f"  pass@3: {fork_pass3:.4f}  ({n_pass_fork}/{n}, Δ{fork_pass3 - orig_pass3:+.4f})")
        print(f"  avg@3:  {fork_avg3:.4f}  (Δ{fork_avg3 - orig_avg3:+.4f})")
    else:
        print(f"\nNo fork_summary.json found at {fork_summary_path}")


if __name__ == "__main__":
    main()
