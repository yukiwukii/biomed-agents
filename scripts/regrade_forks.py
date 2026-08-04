#!/usr/bin/env python3
"""Re-grade the per-fork mini-runs under ``./forks`` with a different rubric model.

Each ``forks/<fork_name>/`` is a self-contained run: one ``trajectories.pkl``
(a single trajectory) plus one ``<uuid>-iterN/score_info.json`` holding the saved
rubric prompt and original score. Unlike the top-level run there is exactly one
score_info per fork, so the trajectory ↔ prompt pairing is 1:1 — no answer
matching needed. Grading reuses ``regrade.py``'s ``make_model`` / ``grade_one`` /
``compute_score`` so the logic stays identical to the main run.

With ``--with-prior-judgment`` the new judge is additionally shown, as a
reference point, this same model's evaluation of the *pre-fork* (parent)
notebook — the notebook identified by ``fork_info.json``'s ``source_traj_id``.
That parent evaluation comes from the main ``regrade.py`` run's
``judge_output.regrade.<model>.json`` (so the reference judge and the fork judge
are the same model). It is a different, earlier notebook than the forked one
being graded; it is framed as context only. Run the main regrade first.

Usage:
    conda run -n bixbench python3 scripts/regrade_forks.py \
        --model anthropic/claude-sonnet-4-6 --temperature 0 --write

    # feed this model's own evaluation of the pre-fork parent as a reference point
    # (requires the main regrade's judge_output.regrade.<model>.json to exist):
    conda run -n bixbench python3 scripts/regrade_forks.py \
        --model anthropic/claude-sonnet-4-6 --temperature 0 --write --with-prior-judgment
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
from pathlib import Path
from statistics import mean, median

import re

from regrade import compute_score, grade_one, make_model

ROOT = Path(__file__).resolve().parent.parent


def prior_judgment_block(parent_id: str, parent: dict) -> str:
    """Reference-only section appended to the rubric prompt.

    ``parent`` is the evaluation of the *pre-fork* (parent) notebook — a DIFFERENT,
    earlier notebook than the forked one being graded — so it is framed as context,
    not as an evaluation of the current submission. The new judge is told to form
    its own independent assessment of the forked notebook."""
    lines = [
        "\n\n<reference-evaluation>",
        "This submission is a FORK: it was branched from an earlier notebook and then "
        "continued differently. Below is a previous evaluator's assessment of that "
        "ORIGINAL, PRE-FORK notebook — a DIFFERENT notebook than the one you are grading "
        "now. It is provided ONLY as a point of reference for context on the task and the "
        "shared early steps. Evaluate the notebook above on its own merits; the forked "
        "notebook may have fixed, changed, or introduced issues relative to the original, "
        "so do not assume these scores carry over.",
        f"Reference (pre-fork) notebook id: {parent_id}",
        f"Reference (pre-fork) total: {parent['raw_score']} / {parent['max_score']}",
        "Per-criterion (pre-fork notebook):",
    ]
    for i, c in enumerate(parent.get("criteria") or [], 1):
        lines.append(f"{i}. {c.get('criterion', '')}")
        lines.append(f"   pre-fork score: {c.get('score')}")
        lines.append(f"   pre-fork justification: {c.get('justification', '')}")
    lines.append("</reference-evaluation>")
    lines.append("\nNow produce your own evaluation of the forked notebook in the required JSON format.")
    return "\n".join(lines)


def discover(forks_dir: Path):
    """Yield (fork_name, pkl_path, score_info_dict, source_traj_id) for each gradable fork."""
    for d in sorted(p for p in forks_dir.iterdir() if p.is_dir()):
        pkl = d / "trajectories.pkl"
        sis = list(d.glob("*/score_info.json"))
        fi = d / "fork_info.json"
        if not pkl.exists() or not sis:
            continue
        info = json.loads(sis[0].read_text())
        # [PATCH 24] No saved prompt = deterministically scored; nothing for an LLM to re-grade.
        if "prompt" not in info:
            continue
        source = json.loads(fi.read_text()).get("source_traj_id") if fi.exists() else None
        yield d.name, pkl, info, source


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--forks", type=Path, default=ROOT / "forks")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--write", action="store_true", help="Write combined rewards json (off by default)")
    ap.add_argument(
        "--with-prior-judgment",
        action="store_true",
        help="Show the new judge this model's own evaluation of the PARENT (pre-fork) notebook "
        "as a reference point. Sourced from the main regrade's judge_output json. Outputs get a "
        "'.prior' tag so they don't clobber the blind regrade.",
    )
    ap.add_argument("--parent-judgment", type=Path, default=None,
                    help="Main-regrade judge_output json keyed by parent traj_id "
                    "(default: benchmark_results/judge_output.regrade.<model>.json)")
    args = ap.parse_args()

    normalize = not args.no_normalize
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    forks = list(discover(args.forks))

    all_dirs = [p for p in args.forks.iterdir() if p.is_dir()]
    skipped = len(all_dirs) - len(forks)

    # Map each parent trajectory id -> this model's evaluation of the pre-fork notebook,
    # taken from the main regrade's judge_output json (same judge model as this run).
    # This is the "previous judge" reference the fork judge is shown.
    parent_judgment: dict[str, dict] = {}
    if args.with_prior_judgment:
        pj_path = args.parent_judgment or (ROOT / "benchmark_results" / f"judge_output.regrade.{safe}.json")
        if not pj_path.exists():
            raise SystemExit(
                f"--with-prior-judgment needs the main regrade's judge output at {pj_path} "
                "(run scripts/regrade.py with the same --model first)."
            )
        parent_judgment = json.loads(pj_path.read_text())
        have = sum(1 for _n, _p, _d, src in forks if src in parent_judgment)
        print(f"Parent (pre-fork) evaluations from {pj_path.name}: available for {have}/{len(forks)} forks")

    model = make_model(args.model, args.api_base, args.api_key, args.temperature)
    sem = asyncio.Semaphore(args.concurrency)

    async def run(prompt: str):
        async with sem:
            return await grade_one(model, prompt)

    mode = "with prior judgment (pre-fork reference)" if args.with_prior_judgment else "blind"
    print(f"\nRe-grading {len(forks)} forks with: {args.model}  [{mode}]", end="")
    print(f"  ({skipped} skipped — no pkl/score_info)" if skipped else "")

    def build_prompt(d: dict, source: str | None) -> str:
        parent = parent_judgment.get(source) if args.with_prior_judgment else None
        if parent is None:
            return d["prompt"]
        return d["prompt"] + prior_judgment_block(source, parent)

    rubrics = await asyncio.gather(*(run(build_prompt(d, src)) for _n, _p, d, src in forks))

    print(f"{'fork':32s} {'old':>12s}  {'new':>12s}")
    print("-" * 62)

    new_rewards: dict[str, float] = {}
    judge_output: dict[str, dict] = {}
    old_list, new_list = [], []
    for (name, _pkl, d, src), rubric in zip(forks, rubrics, strict=True):
        max_score = d["max_score"]
        new_raw, new_score = compute_score(rubric, max_score, normalize)
        old_score, old_raw = d["score"], d["raw_score"]
        new_rewards[name] = new_score
        parent = parent_judgment.get(src) if args.with_prior_judgment else None
        judge_output[name] = {
            "model": args.model,
            "max_score": max_score,
            "raw_score": new_raw,
            "score": new_score,
            "old_raw_score": old_raw,
            "old_score": old_score,
            "criteria": [c.model_dump() for c in rubric.criteria],
            "old_criteria": d.get("criteria"),
            "parent_id": src,
            "parent_reference_used": parent is not None,
            "parent_criteria": parent.get("criteria") if parent else None,
            "parent_score": parent.get("score") if parent else None,
        }
        old_list.append(old_score)
        new_list.append(new_score)
        delta = "  ↑" if new_score > old_score else ("  ↓" if new_score < old_score else "  =")
        print(f"{name:32s} {old_score:5.2f} ({old_raw:2}/{max_score:2})  {new_score:5.2f} ({new_raw:2}/{max_score:2}){delta}")

    up = sum(1 for a, b in zip(old_list, new_list) if b > a)
    dn = sum(1 for a, b in zip(old_list, new_list) if b < a)
    eq = len(forks) - up - dn
    print("-" * 62)
    print(f"old mean {mean(old_list):.4f}  median {median(old_list):.4f}")
    print(f"new mean {mean(new_list):.4f}  median {median(new_list):.4f}   mean delta {mean(b - a for a, b in zip(old_list, new_list)):+.4f}")
    print(f"direction: up {up} ({up/len(forks):.0%})  down {dn} ({dn/len(forks):.0%})  same {eq} ({eq/len(forks):.0%})")

    if not args.write:
        print("\n(dry run — pass --write to save combined rewards json)")
        return

    tag = ".prior" if args.with_prior_judgment else ""
    out = args.forks / f"rewards.regrade{tag}.{safe}.json"
    out.write_text(json.dumps(new_rewards, indent=2))

    judge_path = args.forks / f"judge_output.regrade{tag}.{safe}.json"
    judge_path.write_text(json.dumps(judge_output, indent=2))

    # Merge all fork trajectories into one pkl, patching each terminal-step reward.
    merged = []
    for name, pkl, _d, _src in forks:
        for t in pickle.loads(pkl.read_bytes()):
            if t.steps:
                t.steps[-1].reward = new_rewards.get(t.traj_id, new_rewards.get(name))
            merged.append(t)
    pkl_path = args.forks / f"trajectories.regrade{tag}.{safe}.pkl"
    pkl_path.write_bytes(pickle.dumps(merged))

    print(f"\nWrote {out}")
    print(f"Wrote {judge_path}")
    print(f"Wrote {pkl_path}")


if __name__ == "__main__":
    asyncio.run(main())
