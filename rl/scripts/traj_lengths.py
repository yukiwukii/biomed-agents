#!/usr/bin/env python
"""Trajectory length distributions, comparable across both rollout paths.

WHAT QUESTION THIS ANSWERS
--------------------------
``max_total_sequence_length`` in the GRPO config must cover the *whole* episode:
every message the policy ever sees plus everything it generates. Because the
transcript is append-only, that is the token count of the union of all messages
in the trajectory -- which is what this script measures, per trajectory, so the
cap can be chosen against a real distribution instead of a guess.

TWO SOURCES, ONE NUMBER
-----------------------
    benchmark_agent.py   -> trajectories.pkl  (ldp Trajectory objects)
    ng_collect_rollouts  -> rollouts.jsonl    (OpenAI Responses-API items)

Both are reduced to the same quantity. Pass either; pass both to compare.

THREE MEASUREMENT TRAPS, ALL PREVIOUSLY HIT
-------------------------------------------
1. **Images.** Tool results embed plots as ``data:image/png;base64,...``. A vision
   model pays roughly ``w*h/750`` tokens, so images are decoded for their true
   pixel dimensions and priced that way; the base64 blob itself is excluded.

   .. warning::
      **Pixel pricing is correct for the benchmark endpoints and WRONG for the
      GRPO policy.** The training policy is served ``language_model_only: true``
      -- no vision tower -- so vLLM tokenizes the base64 as TEXT. One matplotlib
      figure measured 101,573 tokens that way, 56x its 1,824-token pixel price.
      Following the pixel figure hid that bug for four runs.

      This only bites on files written before ``cfg.STRIP_IMAGES`` (default true)
      landed: fresh rollouts contain no images at all, so ``images:`` should
      report 0 and the distinction is moot. If it reports non-zero on a file you
      just collected, STRIP_IMAGES is not in effect in the env pod -- stop and
      fix that before trusting any length in the output.
2. **Double counting.** ``observation[i] == next_observation[i-1]`` in ldp, so
   summing both counts every tool result twice. Only ``steps[0].observation``
   plus each step's ``next_observation`` is taken.
3. **The code cells.** A ``ToolRequestMessage`` keeps its code in ``tool_calls``,
   not in ``.content``. Counting content alone silently drops every code cell.

THE SEED IS MISSING FROM THE GYM PATH
-------------------------------------
``ng_collect_rollouts`` does not persist the system prompt, task text or tool
schema (``responses_create_params.instructions`` is None and ``input`` is ``[]``).
So a pkl converted by ``rollouts_to_pkl.py`` starts mid-conversation and
UNDER-states the real context. ``--seed-tokens N`` adds a flat N to every
trajectory from such a source; the benchmark's own seeds measure 1,769-2,273
tokens (median 1,871), so ``--seed-tokens 1900`` is the like-for-like default.

Prefer ``usage.input_tokens`` where available. Reading ``rollouts.jsonl``
directly reports it: it is the final turn's full prompt as counted by the *real*
Qwen tokenizer, and it already includes the seed, the tool schema and the chat
template. The cl100k figures are estimates that exist only so the two sources can
be compared on equal terms.

    .venv/bin/python rl/scripts/traj_lengths.py \
        archive/sonnet-judge/benchmark_results-hypotest-wo-protocol/trajectories.pkl \
        rl/results/smoke/rollouts.jsonl --seed-tokens 1900
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import pickle
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tiktoken
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    sys.exit(f"need tiktoken + pillow: {exc}\nTry .venv/bin/python")

ENC = tiktoken.get_encoding("cl100k_base")
PIXELS_PER_TOKEN = 750
DATA_URI = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,([A-Za-z0-9+/=\s]+)")


def _image_tokens(b64: str) -> int:
    """Price one embedded image by its pixels, not by its base64 length."""
    try:
        raw = base64.b64decode(re.sub(r"\s", "", b64), validate=False)
        with Image.open(io.BytesIO(raw)) as im:
            w, h = im.size
    except (binascii.Error, OSError, ValueError):
        # Undecodable: fall back to the text estimate rather than dropping it.
        return len(ENC.encode(b64)) // 4
    return max(1, (w * h) // PIXELS_PER_TOKEN)


def count_tokens(text: str) -> tuple[int, int, int]:
    """-> (total, text_tokens, image_tokens). Image blobs are excised, then priced."""
    if not text:
        return 0, 0, 0
    img = 0
    if "base64," in text:
        for m in DATA_URI.finditer(text):
            img += _image_tokens(m.group(1))
        text = DATA_URI.sub("", text)
    txt = len(ENC.encode(text))
    return txt + img, txt, img


def _msg_text(msg: Any) -> str:
    """Everything the model pays for in one message: content AND any tool call."""
    parts = [str(getattr(msg, "content", "") or "")]
    for call in getattr(msg, "tool_calls", None) or []:
        fn = getattr(call, "function", None)
        if fn is not None:
            parts.append(str(getattr(fn, "name", "") or ""))
            args = getattr(fn, "arguments", "")
            parts.append(args if isinstance(args, str) else json.dumps(args))
    return "\n".join(p for p in parts if p)


@dataclass
class TrajStat:
    traj_id: str
    steps: int
    total: int = 0
    seed: int = 0
    thinking_and_code: int = 0  # what the policy generated
    tool_results: int = 0  # what the environment returned
    images: int = 0
    reward: float = 0.0
    usage_input: int | None = None  # real-tokenizer final context, when recorded
    usage_total: int | None = None


def stats_from_pkl(path: Path, seed_tokens: int) -> list[TrajStat]:
    trajectories = pickle.loads(path.read_bytes())
    out: list[TrajStat] = []
    for tr in trajectories:
        s = TrajStat(traj_id=tr.traj_id, steps=len(tr.steps))
        if not tr.steps:
            out.append(s)
            continue
        for msg in tr.steps[0].observation:  # the seed, if this source has one
            t, _, i = count_tokens(_msg_text(msg))
            s.seed += t
            s.images += i
        for st in tr.steps:
            if st.action is not None:
                t, _, i = count_tokens(_msg_text(st.action.value))
                s.thinking_and_code += t
                s.images += i
            for msg in st.next_observation:  # == observation[i+1]; counted once
                t, _, i = count_tokens(_msg_text(msg))
                s.tool_results += t
                s.images += i
        if s.seed == 0:
            s.seed = seed_tokens  # Gym path: seed was never persisted
        s.total = s.seed + s.thinking_and_code + s.tool_results
        s.reward = float(tr.steps[-1].reward or 0.0)
        out.append(s)
    return out


def _item_text(item: dict[str, Any]) -> str:
    for key in ("text", "content", "summary", "output", "arguments"):
        v = item.get(key)
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            parts = []
            for e in v:
                if isinstance(e, str):
                    parts.append(e)
                elif isinstance(e, dict):
                    for k2 in ("text", "content", "output"):
                        if isinstance(e.get(k2), str):
                            parts.append(e[k2])
                            break
            if parts:
                return "".join(parts)
    return ""


def stats_from_jsonl(path: Path, seed_tokens: int) -> list[TrajStat]:
    out: list[TrajStat] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        resp = rec.get("response") or {}
        items = resp.get("output") or []
        ti, ri = rec.get("_ng_task_index", 0), rec.get("_ng_rollout_index", 0)
        s = TrajStat(
            traj_id=f"task_{ti}_rep{ri}",
            steps=sum(1 for it in items if it.get("type") == "function_call"),
            seed=seed_tokens,
            reward=float(rec.get("reward") or 0.0),
        )
        for it in items:
            t, _, i = count_tokens(_item_text(it))
            s.images += i
            if it.get("type") == "function_call_output":
                s.tool_results += t
            else:  # reasoning / message / function_call arguments
                s.thinking_and_code += t
        s.total = s.seed + s.thinking_and_code + s.tool_results
        usage = resp.get("usage") or {}
        s.usage_input = usage.get("input_tokens")
        s.usage_total = usage.get("total_tokens")
        out.append(s)
    return out


@dataclass
class Summary:
    label: str
    stats: list[TrajStat] = field(default_factory=list)

    def _d(self, values: list[int]) -> dict[str, float]:
        v = sorted(values)
        n = len(v)
        return {
            "n": n,
            "min": v[0],
            "median": statistics.median(v),
            "mean": statistics.mean(v),
            "p90": v[min(n - 1, int(0.90 * n))],
            "p95": v[min(n - 1, int(0.95 * n))],
            "p99": v[min(n - 1, int(0.99 * n))],
            "max": v[-1],
        }

    def report(self) -> None:
        st = self.stats
        if not st:
            print(f"\n{self.label}: no trajectories")
            return
        d = self._d([s.total for s in st])
        print(f"\n{'=' * 78}\n{self.label}  (n={d['n']})\n{'=' * 78}")
        print("total sequence length, cl100k, images priced by pixels:")
        for k in ("min", "median", "mean", "p90", "p95", "p99", "max"):
            print(f"    {k:<7}{d[k]:>12,.0f}")
        steps = self._d([s.steps for s in st])
        print(
            f"\nsteps/episode:  min {steps['min']:.0f}  median {steps['median']:.0f}"
            f"  mean {steps['mean']:.1f}  max {steps['max']:.0f}"
        )
        gen = sum(s.thinking_and_code for s in st)
        tool = sum(s.tool_results for s in st)
        seed = sum(s.seed for s in st)
        tot = gen + tool + seed
        print(
            f"composition:    seed {seed / tot:>5.1%}   generated {gen / tot:>5.1%}   tool results {tool / tot:>5.1%}"
        )
        print(f"images:         {sum(s.images for s in st):,} tokens  ({sum(s.images for s in st) / tot:.1%})")
        per_step = [s.total / s.steps for s in st if s.steps]
        if per_step:
            print(f"per step:       median {statistics.median(per_step):,.0f} tokens")

        real = [s.usage_input for s in st if s.usage_input]
        if real:
            r = self._d(real)
            print(
                "\nusage.input_tokens -- REAL Qwen tokenizer, includes seed +"
                " tool schema + template.\nThis is the number to size the cap"
                f" against:\n    (n={r['n']})"
            )
            for k in ("min", "median", "mean", "p90", "p95", "p99", "max"):
                print(f"    {k:<7}{r[k]:>12,.0f}")
            # Pair PER TRAJECTORY. Dividing median(real) by median(total) compares
            # two different subsets whenever usage is missing on some rows, and
            # silently reports a ratio for trajectories that were never measured.
            paired = [s.total / s.usage_input for s in st if s.usage_input and s.total]
            r_med = statistics.median(paired)
            print(
                f"\n    cl100k / usage.input, paired per trajectory (n={len(paired)}"
                f" of {len(st)}): {min(paired):.3f}-{max(paired):.3f}, median {r_med:.3f}"
            )
            print(
                f"    -> a cl100k figure UNDER-states the real count; multiply by ~{1 / r_med:.2f} before sizing a cap."
            )

    def cap_table(self) -> None:
        """What fraction of episodes a given cap would truncate."""
        totals = sorted(s.total for s in self.stats)
        if not totals:
            return
        n = len(totals)
        print(f"\ntruncation rate by cap -- {self.label}:")
        print(f"    {'cap':>10}  {'kept':>7}  {'truncated':>10}")
        for cap in (16384, 24576, 32768, 40960, 49152, 65536, 98304, 131072):
            kept = sum(1 for t in totals if t <= cap)
            print(f"    {cap:>10,}  {kept / n:>6.1%}  {1 - kept / n:>9.1%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", type=Path, help=".pkl and/or rollouts .jsonl")
    ap.add_argument(
        "--seed-tokens",
        type=int,
        default=1900,
        help="flat seed added to sources that did not persist one (default 1900)",
    )
    ap.add_argument("--caps", action="store_true", help="print truncation rate by cap")
    ap.add_argument("--per-traj", action="store_true", help="print every trajectory")
    args = ap.parse_args()

    summaries = []
    for src in args.sources:
        if not src.exists():
            print(f"skip (missing): {src}")
            continue
        stats = (
            stats_from_jsonl(src, args.seed_tokens) if src.suffix == ".jsonl" else stats_from_pkl(src, args.seed_tokens)
        )
        s = Summary(label=str(src), stats=stats)
        summaries.append(s)
        s.report()
        if args.caps:
            s.cap_table()
        if args.per_traj:
            print(f"\n    {'traj_id':<20}{'steps':>6}{'total':>10}{'reward':>8}{'usage_in':>10}")
            for t in sorted(stats, key=lambda x: -x.total):
                u = f"{t.usage_input:,}" if t.usage_input else "-"
                print(f"    {t.traj_id:<20}{t.steps:>6}{t.total:>10,}{t.reward:>8.3f}{u:>10}")

    if len(summaries) > 1:
        print(f"\n{'=' * 78}\nCOMPARISON (cl100k, like-for-like)\n{'=' * 78}")
        print(f"{'source':<44}{'n':>5}{'median':>10}{'mean':>10}{'max':>10}")
        for s in summaries:
            if not s.stats:
                continue
            tt = [x.total for x in s.stats]
            name = s.label if len(s.label) <= 42 else "..." + s.label[-39:]
            print(f"{name:<44}{len(tt):>5}{statistics.median(tt):>10,.0f}{statistics.mean(tt):>10,.0f}{max(tt):>10,}")
        base, *rest = [s for s in summaries if s.stats]
        bm = statistics.median([x.total for x in base.stats])
        for s in rest:
            m = statistics.median([x.total for x in s.stats])
            verdict = "about the same" if 0.9 <= m / bm <= 1.1 else ("LONGER" if m > bm else "SHORTER")
            print(f"\n{s.label}\n    median is {m / bm:.2f}x the first source -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
