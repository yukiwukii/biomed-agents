#!/usr/bin/env python
"""Validate a custom_parallel_plan against a checkpoint's real tensor names.

Offline. No GPU, no cluster. Run this BEFORE submitting -- the 2026-08-06 run
burned 5 H100s for 90 seconds to discover that the auto-selected plan matched
one key out of 1199 because the paths were rooted at `model.` and this model
puts everything under `model.language_model.`.

    .venv/bin/python rl/scripts/check_tp_plan.py \
        --plan rl/configs/qwen3_tp_plan.py \
        --model Qwen/Qwen3.6-27B
"""

from __future__ import annotations

import argparse
import collections
import fnmatch
import glob
import json
import os
import re
import struct
import sys

GB = 1e9


def load_plan_keys(path: str) -> list[str]:
    """Read the plan's dict keys without importing torch."""
    src = open(path, encoding="utf-8").read()
    lm = re.search(r'^_LM\s*=\s*["\'](.+?)["\']', src, re.MULTILINE)
    prefix = lm.group(1) if lm else ""
    return [
        raw.replace("{_LM}", prefix)
        for raw in re.findall(r'^\s*f?["\'](.+?)["\']\s*:\s*(?:Colwise|Rowwise)', src, re.MULTILINE)
    ]


def checkpoint_tensors(model_dir: str) -> dict[str, int]:
    """Map tensor name -> bytes, straight from the safetensors headers."""
    out: dict[str, int] = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            numel = 1
            for d in v["shape"]:
                numel *= d
            out[k] = numel * 2  # bf16
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--model-dir", required=True, help="snapshot dir with *.safetensors")
    args = ap.parse_args()

    keys = load_plan_keys(args.plan)
    tensors = checkpoint_tensors(args.model_dir)
    if not tensors:
        print(f"no safetensors under {args.model_dir}", file=sys.stderr)
        return 2

    # A plan key names a MODULE; checkpoint entries are that module's .weight/.bias.
    modules = {n.rsplit(".", 1)[0] for n in tensors}
    size: collections.Counter[str] = collections.Counter()
    for n, b in tensors.items():
        size[n.rsplit(".", 1)[0]] += b

    matched: dict[str, list[str]] = {}
    for k in keys:
        matched[k] = sorted(m for m in modules if fnmatch.fnmatch(m, k))

    total = sum(tensors.values())
    covered = set()
    print(f"plan keys: {len(keys)}   checkpoint modules: {len(modules)}\n")
    bad = False
    for k in keys:
        ms = matched[k]
        gb = sum(size[m] for m in ms) / GB
        flag = "  <-- MATCHES NOTHING" if not ms else ""
        print(f"  {len(ms):>4} modules  {gb:>6.2f} GB   {k}{flag}")
        if not ms:
            bad = True
        covered |= set(ms)

    sharded = sum(size[m] for m in covered)
    print(f"\n  sharded   {sharded / GB:>7.2f} GB  ({100 * sharded / total:.0f}%)")
    print(f"  replicated{(total - sharded) / GB:>7.2f} GB")
    print(f"  TOTAL     {total / GB:>7.2f} GB")
    for tp in (2, 4):
        per = (sharded / tp + (total - sharded)) / GB
        print(f"    TP={tp}: {per:6.2f} GB/GPU weights")

    # Things that must NOT be swept in.
    print()
    for label, pat in (("linear_attn", "*linear_attn*"), ("visual", "model.visual*"), ("mtp", "mtp*")):
        leaked = sorted(m for m in covered if fnmatch.fnmatch(m, pat))
        print(f"  {label:<12} in plan: {len(leaked)}" + ("  <-- UNEXPECTED" if leaked else "  ok"))
        if leaked:
            bad = True

    print("\nRESULT:", "FAIL" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
