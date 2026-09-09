#!/usr/bin/env python
"""Attribute a training episode's tokens to what actually produced them.

WHAT QUESTION THIS ANSWERS
--------------------------
A training episode was observed at 62,871 tokens while its notebook accounted for
only ~23,000. The remaining ~40,000 was never identified, and it is what sizes
``grad_input`` and OOMs the run:

    grad_input = longest_episode_in_microbatch x (vocab / TP) x dtype_bytes

The leading hypothesis was that thinking dominates (``max_new_tokens`` is 4,090
per turn), but nothing on disk could confirm it -- the training path writes no
``rollouts.jsonl``, and upstream's own dump runs after the training step that has
never completed. ``HYPOTEST_TRAJ_DUMP`` (see ``rl/scripts/nemo_rl_setup.sh``)
writes before that step; this reads what it produces.

INPUT
-----
A directory of ``traj_step{N}.jsonl``, one JSON object per episode::

    {"idx", "input_length", "total_message_tokens", "n_messages",
     "reward", "messages": [{"role", "n_tokens", "content"}, ...]}

``input_length`` is authoritative: it is the real trained sequence length, taken
straight from ``batched_message_log_to_flat_message``.

USAGE
-----
    .venv/bin/python rl/scripts/analyze_traj_dump.py rl/results/traj/<tag>
    .venv/bin/python rl/scripts/analyze_traj_dump.py <dir> --top 5   # worst episodes
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

# <think>...</think> is how this model family emits reasoning, and the served
# endpoints run no reasoning parser during training, so it stays inline in the
# assistant content rather than arriving as a separate field.
THINK = re.compile(r"<think>(.*?)</think>", re.S)


def classify(msg: dict) -> str:
    """Bucket a message by what actually generated its tokens."""
    role = (msg.get("role") or "unknown").lower()
    content = msg.get("content") or ""
    if role == "assistant":
        return "assistant:thinking" if THINK.search(content) else "assistant:action"
    if role in {"tool", "function", "function_call_output"}:
        return "tool_result"
    if role == "system":
        return "system"
    if role == "user":
        return "user/task"
    return role


def split_assistant(msg: dict) -> tuple[int, int]:
    """-> (thinking_chars, other_chars) for one assistant message."""
    content = msg.get("content") or ""
    think = sum(len(m) for m in THINK.findall(content))
    return think, max(0, len(content) - think)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=pathlib.Path, help="directory of traj_step*.jsonl, or a single file")
    ap.add_argument("--top", type=int, default=3, help="show this many longest episodes in detail")
    args = ap.parse_args()

    files = (
        sorted(args.path.glob("traj_step*.jsonl"))
        if args.path.is_dir()
        else [args.path]
    )
    if not files:
        print(f"no traj_step*.jsonl under {args.path}", file=sys.stderr)
        return 1

    episodes = []
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                ep = json.loads(line)
                ep["_file"] = f.name
                episodes.append(ep)

    if not episodes:
        print("dump files exist but contain no episodes", file=sys.stderr)
        return 1

    lengths = sorted(e["input_length"] for e in episodes)
    n = len(lengths)
    print(f"{n} episodes from {len(files)} step file(s)\n")
    print("TRAINED SEQUENCE LENGTH (input_length -- this is what sizes grad_input)")
    print(f"  min {lengths[0]:,}   median {lengths[n // 2]:,}   max {lengths[-1]:,}")

    # grad_input for the worst episode, which is the one that decides the run
    vocab, tp = 248320, 4
    for dt, name in ((2, "bf16"), (4, "fp32")):
        gib = lengths[-1] * (vocab // tp) * dt / 1024**3
        print(f"  -> grad_input at TP={tp}, {name}: {gib:.2f} GiB")
    print()

    # ---- where the tokens go, aggregated -----------------------------------
    buckets: Counter[str] = Counter()
    think_chars = other_chars = 0
    for ep in episodes:
        for m in ep["messages"]:
            buckets[classify(m)] += m["n_tokens"]
            if (m.get("role") or "").lower() == "assistant":
                t, o = split_assistant(m)
                think_chars += t
                other_chars += o

    total = sum(buckets.values()) or 1
    print("TOKEN ATTRIBUTION (all episodes)")
    for k, v in buckets.most_common():
        bar = "#" * int(40 * v / total)
        print(f"  {k:<20} {v:>10,}  {v / total * 100:5.1f}%  {bar}")

    if think_chars + other_chars:
        share = think_chars / (think_chars + other_chars) * 100
        print(f"\n  within assistant messages: {share:.1f}% of characters are inside <think>")
        print("  (chars, not tokens -- a proxy, but the ratio is the point)")

    # ---- the episodes that actually decide the allocation -------------------
    print(f"\nLONGEST {args.top} EPISODES")
    for ep in sorted(episodes, key=lambda e: -e["input_length"])[: args.top]:
        b: Counter[str] = Counter()
        for m in ep["messages"]:
            b[classify(m)] += m["n_tokens"]
        print(
            f"\n  {ep['_file']} idx={ep['idx']}  input_length={ep['input_length']:,}  "
            f"messages={ep['n_messages']}  reward={ep.get('reward')}"
        )
        for k, v in b.most_common():
            print(f"      {k:<20} {v:>9,}  {v / max(1, sum(b.values())) * 100:5.1f}%")
        biggest = max(ep["messages"], key=lambda m: m["n_tokens"])
        print(
            f"      largest single message: {biggest['n_tokens']:,} tok "
            f"({biggest.get('role')})"
        )

    print(
        "\nHOW TO USE THIS: per-step cost = input_length / n_messages. Size\n"
        "AGENT_MAX_STEPS so per_step_cost x steps stays under the ceiling that\n"
        "fits memory, rather than letting max_total_sequence_length truncate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
