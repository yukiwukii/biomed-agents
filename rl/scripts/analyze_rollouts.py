#!/usr/bin/env python3
r"""Smoke-gate analysis of a NeMo Gym rollout JSONL.

This is the gate that decides whether GRPO can start, so it reports the four
things that actually block training — not a generic metrics dump:

1. **Reward distribution.** If every reward is 0.0 the environment is not
   scoring and GRPO has nothing to learn from. If every reward is identical,
   group-relative advantage is exactly zero and training is a no-op regardless
   of what the loss curve does.

2. **Submission rate.** An episode that never calls ``submit_answer`` is scored
   0.0 without a judge call (``interpreter_env.py:1638``). Those are
   indistinguishable from genuine failures and drag the whole group's baseline
   down, so a high truncation rate must be fixed before training, not after.

3. **Sequence length.** NeMo RL trains on the *entire* concatenated multi-turn
   sequence, so ``policy.max_total_sequence_length`` must cover the p99 episode,
   not the median. This is the single number most likely to OOM a 27B run.

4. **Turn count.** Tells you which lever to pull if (3) is too large.

Usage::

    python rl/scripts/analyze_rollouts.py results/smoke/rollouts.jsonl
    python rl/scripts/analyze_rollouts.py results/smoke/rollouts.jsonl \\
        --tokenizer Qwen/Qwen3.6-27B --max-seq-len 32768

Without ``--tokenizer`` the token counts are a chars/4 estimate — fine for a
first look, but re-run with the real tokenizer before setting
``max_total_sequence_length``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Rows carry the reward under "reward"; task identity under "_ng_task_index"
# (falling back to "task_idx"); and the trajectory under "response"."output".
# Mirrors bbh/scripts/lib/rollout_utils.py so the two stay comparable.


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def iter_response_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    resp = record.get("response")
    if isinstance(resp, dict):
        items = resp.get("output") or resp.get("items")
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)]
    if isinstance(resp, list):
        return [x for x in resp if isinstance(x, dict)]
    return []


def task_index(record: dict[str, Any], default: int = -1) -> int:
    return int(record.get("_ng_task_index", record.get("task_idx", default)))


def has_submit_answer(record: dict[str, Any]) -> bool:
    return any(
        it.get("type") == "function_call" and it.get("name") == "submit_answer" for it in iter_response_items(record)
    )


def reward_value(record: dict[str, Any]) -> float | None:
    reward = record.get("reward")
    if reward is None:
        return None
    try:
        return float(reward)
    except (TypeError, ValueError):
        return None


def count_turns(record: dict[str, Any]) -> int:
    """Assistant turns = function calls + assistant messages."""
    return sum(1 for it in iter_response_items(record) if it.get("type") in {"function_call", "message"})


def episode_text(record: dict[str, Any]) -> str:
    """Everything the policy will see concatenated, for a length estimate.

    Includes the seeded task description (the ``input`` side) as well as the
    generated output — NeMo RL trains on the whole sequence, so measuring only
    the completion would understate it badly.
    """
    parts: list[str] = []
    resp = record.get("response")
    if isinstance(resp, dict):
        raw_input = resp.get("input")
        if isinstance(raw_input, list):
            parts.append(json.dumps(raw_input, ensure_ascii=False))
    params = record.get("responses_create_params")
    if isinstance(params, dict) and isinstance(params.get("input"), list):
        parts.append(json.dumps(params["input"], ensure_ascii=False))
    parts.append(json.dumps(iter_response_items(record), ensure_ascii=False))
    return "\n".join(parts)


def make_token_counter(tokenizer_name: str | None):
    """Return (fn, label). Falls back to a chars/4 estimate."""
    if not tokenizer_name:
        return (lambda s: len(s) // 4), "estimated (chars/4)"
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            f"WARNING: transformers not installed; ignoring --tokenizer {tokenizer_name} and estimating.",
            file=sys.stderr,
        )
        return (lambda s: len(s) // 4), "estimated (chars/4)"
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return (lambda s: len(tok(s, add_special_tokens=False)["input_ids"])), f"exact ({tokenizer_name})"


def pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile. Small n here, so no interpolation games."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q / 100.0 * len(ordered) + 0.5) - 1))
    return ordered[idx]


def histogram(values: list[float], bins: int = 10, width: int = 40) -> list[str]:
    if not values:
        return ["  (no data)"]
    lo, hi = min(values), max(values)
    if hi == lo:
        return [f"  all {len(values)} values == {lo:g}"]
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int((v - lo) / step))] += 1
    peak = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(width * c / peak)
        lines.append(f"  [{lo + i * step:9.4g}, {lo + (i + 1) * step:9.4g})  {c:4d}  {bar}")
    return lines


def summarize(values: list[float], label: str, unit: str = "") -> None:
    if not values:
        print(f"{label:24s} (none)")
        return
    suffix = f" {unit}" if unit else ""
    print(
        f"{label:24s} n={len(values):<5d} mean={statistics.mean(values):9.4g}{suffix}  "
        f"min={min(values):9.4g}  p50={pct(values, 50):9.4g}  "
        f"p95={pct(values, 95):9.4g}  p99={pct(values, 99):9.4g}  max={max(values):9.4g}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rollouts_jsonl", type=Path)
    p.add_argument(
        "--tokenizer",
        default=None,
        help="HF id for exact token counts, e.g. Qwen/Qwen3.6-27B. Omit for a chars/4 estimate.",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Proposed policy.max_total_sequence_length; reports the fraction of episodes that would overflow it.",
    )
    args = p.parse_args()

    if not args.rollouts_jsonl.exists():
        print(f"FATAL: {args.rollouts_jsonl} not found", file=sys.stderr)
        raise SystemExit(1)

    records = load_jsonl(args.rollouts_jsonl)
    if not records:
        print("FATAL: no rollouts in file.", file=sys.stderr)
        raise SystemExit(1)

    count_tokens, token_label = make_token_counter(args.tokenizer)

    rewards: list[float] = []
    by_task: dict[int, list[float]] = defaultdict(list)
    submitted, truncated = 0, 0
    submitted_rewards: list[float] = []
    truncated_rewards: list[float] = []
    turn_counts: list[float] = []
    token_counts: list[float] = []

    for rec in records:
        r = reward_value(rec)
        if r is not None:
            rewards.append(r)
            by_task[task_index(rec)].append(r)
        if has_submit_answer(rec):
            submitted += 1
            if r is not None:
                submitted_rewards.append(r)
        else:
            truncated += 1
            if r is not None:
                truncated_rewards.append(r)
        turn_counts.append(float(count_turns(rec)))
        token_counts.append(float(count_tokens(episode_text(rec))))

    n = len(records)
    print("=" * 92)
    print(f"GRPO SMOKE GATE  —  {args.rollouts_jsonl}")
    print(f"rollouts: {n}    unique tasks: {len(by_task)}    token counts: {token_label}")
    print("=" * 92)

    # ---- 1. reward -------------------------------------------------------
    print("\n[1] REWARD  (hypotest reward is terminal-only, normalized to [0,1])")
    summarize(rewards, "  reward")
    if rewards:
        n_zero = sum(1 for r in rewards if r == 0.0)
        n_full = sum(1 for r in rewards if r == 1.0)
        print(f"  zero reward:           {n_zero}/{len(rewards)} ({100 * n_zero / len(rewards):.1f}%)")
        print(f"  full marks:            {n_full}/{len(rewards)} ({100 * n_full / len(rewards):.1f}%)")
        pass1 = sum(1 for rs in by_task.values() if any(r == 1.0 for r in rs)) / len(by_task)
        print(f"  pass@1 (any full):     {pass1:.3f}")
        print("\n  distribution:")
        for line in histogram(rewards):
            print(line)

        # The check that matters most for GRPO and that a plain mean hides.
        degenerate = [t for t, rs in by_task.items() if len(rs) > 1 and len(set(rs)) == 1]
        if degenerate:
            print(
                f"\n  !! {len(degenerate)}/{len(by_task)} tasks have IDENTICAL reward across all "
                f"repeats. GRPO advantage is exactly 0 for those groups — they contribute no "
                f"gradient. Tasks: {sorted(degenerate)[:8]}"
            )
        if rewards and len(set(rewards)) == 1:
            print("\n  !! ALL rewards identical. GRPO cannot learn from this. Check the judge is running.")

    # ---- 2. submission ---------------------------------------------------
    print("\n[2] SUBMISSION / TRUNCATION")
    print(f"  submitted:             {submitted}/{n} ({100 * submitted / n:.1f}%)")
    print(f"  no submit_answer:      {truncated}/{n} ({100 * truncated / n:.1f}%)")
    if truncated_rewards:
        summarize(truncated_rewards, "  reward | truncated")
    if submitted_rewards:
        summarize(submitted_rewards, "  reward | submitted")
    if truncated and truncated / n > 0.15:
        print(
            f"\n  !! {100 * truncated / n:.0f}% of episodes never submitted. These score 0.0 with no "
            f"judge call and are indistinguishable from real failures. Raise AGENT_MAX_STEPS, or "
            f"force a submit_answer on the final step, before training."
        )

    # ---- 3. sequence length ---------------------------------------------
    print("\n[3] SEQUENCE LENGTH  (drives policy.max_total_sequence_length)")
    summarize(token_counts, "  tokens/episode", "tok")
    print("\n  distribution:")
    for line in histogram(token_counts):
        print(line)
    if args.max_seq_len:
        over = sum(1 for t in token_counts if t > args.max_seq_len)
        print(f"\n  would overflow max_total_sequence_length={args.max_seq_len}: {over}/{n} ({100 * over / n:.1f}%)")
        if over:
            print(
                "  !! Overflowing episodes are truncated or dropped. Lower AGENT_MAX_STEPS, set "
                "collapse_old_env_states: true, or raise max_total_sequence_length (and enable "
                "context parallelism + sequence packing to pay for it)."
            )
        headroom = pct(token_counts, 99)
        print(f"  p99 is {headroom:.0f} tok — a safe max_total_sequence_length is ~{int(headroom * 1.2)} (p99 x 1.2).")

    # ---- 4. turns --------------------------------------------------------
    print("\n[4] TURNS PER EPISODE")
    summarize(turn_counts, "  assistant turns")

    # ---- notes -----------------------------------------------------------
    print("\n[note] score_info is expected to be absent in remote mode: Gym's")
    print("       score_info_from_env (app.py:141) reads env.state, which a")
    print("       TaskEnvironmentClient proxy does not have. The reward field is")
    print("       still authoritative; full judge detail is written server-side to")
    print("       save_dir/<problem_id>-iterN/score_info.json.")
    print("=" * 92)


if __name__ == "__main__":
    main()
