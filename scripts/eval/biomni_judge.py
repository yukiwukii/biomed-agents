#!/usr/bin/env python3
"""Grade solutions with BiomniBench-DA's *original* judging method.

This is an **additive** grader — it does not touch hypotest's built-in scoring
(`InterpreterEnv._score_solution`). It faithfully reproduces the judge shipped in
the BiomniBench-DA dataset (`da-*/tests/llm_judge.py`), whose key property is:

    The LLM only chooses a level (A / B / C) per rubric criterion. It never does
    arithmetic. Python maps each chosen letter to the criterion's rubric-defined
    point value (`Levels: A=X B=Y C=0`), sums to a 0-100 score, and clamps.

This removes judge arithmetic noise, unlike hypotest's default judge (which asks
the model to emit an integer score per criterion directly).

Two input modes, so the same judge works for both hypotest outputs and native
BiomniBench-DA outputs:

  1. hypotest results (default): re-grade every ``<results>/*/score_info.json``.
     hypotest saves the fully-rendered grading prompt under the ``"prompt"`` key,
     which embeds ``<rubric>``, ``<notebook>``, and ``<proposed-solution>``. We
     extract those three, feed the notebook as the analysis "trace" and the
     proposed solution as the "answer", and grade with the biomni method. No
     agent re-run and no environment needed.

  2. native BiomniBench-DA (``--rubric/--trace/--answer`` files): grade one task's
     ``tests/rubric.txt`` against the agent-written ``trace.md`` + ``answer.txt``,
     exactly like the original ``llm_judge.py`` (but with a configurable model).

Usage:
    # Re-grade a hypotest run with the biomni method (dry run — prints a table):
    .venv/bin/python scripts/eval/biomni_judge.py --model openai/gpt-5 --results results/

    # Local/vLLM OpenAI-compatible endpoint, and write per-dir + summary outputs:
    .venv/bin/python scripts/eval/biomni_judge.py \
        --model openai/Qwen/Qwen3.6-27B --api-base http://localhost:8000/v1 \
        --api-key none --results results/ --write

    # Native BiomniBench-DA single task:
    .venv/bin/python scripts/eval/biomni_judge.py --model openai/gpt-5 \
        --rubric da-1-3/tests/rubric.txt --trace trace.md --answer answer.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from lmi import LiteLLMModel

ROOT = Path(__file__).resolve().parents[2]

# Import the shared judging core from the package so this script and
# InterpreterEnv's live grading path stay in lockstep (single source of truth).
sys.path.insert(0, str(ROOT / "src"))
from hypotest.env.biomni_judge import (  # noqa: E402
    build_judge_prompt,
    parse_rubric_levels,  # noqa: F401  (re-exported for convenience / callers)
    score_from_response,
)


def load_env(path: Path = ROOT / ".env") -> None:
    """Populate os.environ from a KEY=VALUE .env file (does not overwrite existing keys)."""
    import os

    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


async def grade(model: LiteLLMModel, rubric: str, trace: str, answer: str) -> tuple[int, dict, str]:
    """Run the biomni judge once and return (total_score/100, criteria, reasoning)."""
    if not trace and not answer:
        return 0, {}, "No trace or answer provided."
    prompt = build_judge_prompt(rubric, trace, answer)
    resp = await model.call_single(prompt, timeout=3 * 60)
    if not resp.text:
        raise ValueError("No response from judge model")
    return score_from_response(resp.text, rubric)


# ── extraction of (rubric, notebook, answer) from a hypotest saved prompt ─────


def _section(prompt: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_from_score_info(prompt: str) -> tuple[str, str, str]:
    """Pull (rubric, notebook, proposed_solution) out of a saved hypotest grading prompt."""
    return _section(prompt, "rubric"), _section(prompt, "notebook"), _section(prompt, "proposed-solution")


# ── model construction (mirrors regrade.py) ──────────────────────────────────


def make_model(name: str, api_base: str | None, api_key: str | None, temperature: float | None) -> LiteLLMModel:
    if api_base:
        params: dict = {"model": name, "api_base": api_base, "api_key": api_key or "none"}
        if temperature is not None:
            params["temperature"] = temperature
        return LiteLLMModel(name=name, config={"model_list": [{"model_name": name, "litellm_params": params}]})
    return LiteLLMModel(name=name, config={"temperature": temperature} if temperature is not None else {})


# ── modes ────────────────────────────────────────────────────────────────────


async def run_native(args: argparse.Namespace, model: LiteLLMModel) -> None:
    rubric = Path(args.rubric).read_text()
    trace = Path(args.trace).read_text() if args.trace and Path(args.trace).exists() else ""
    answer = Path(args.answer).read_text() if args.answer and Path(args.answer).exists() else ""
    total, criteria, reasoning = await grade(model, rubric, trace, answer)
    out = {"total_score": total, "score": total / 100, "criteria": criteria, "reasoning": reasoning}
    print(json.dumps(out, indent=2))
    if args.write:
        dest = Path(args.write if isinstance(args.write, str) else "biomni_evaluation.json")
        dest.write_text(json.dumps(out, indent=2))
        print(f"\nWrote {dest}")


async def run_results(args: argparse.Namespace, model: LiteLLMModel) -> None:
    score_infos = sorted(args.results.glob("*/score_info.json"))
    if not score_infos:
        raise SystemExit(f"no */score_info.json under {args.results}")

    sem = asyncio.Semaphore(args.concurrency)

    async def grade_path(p: Path):
        d = json.loads(p.read_text())
        # [PATCH 24] Deterministically-scored runs save no prompt — nothing to re-grade.
        rubric, notebook, answer = extract_from_score_info(d.get("prompt", ""))
        if not rubric:
            return p, d, None
        async with sem:
            total, criteria, reasoning = await grade(model, rubric, notebook, answer)
        return p, d, {"total_score": total, "score": total / 100, "criteria": criteria, "reasoning": reasoning}

    results = await asyncio.gather(*(grade_path(p) for p in score_infos))

    print(f"\nBiomni-method re-grade of {len(results)} solutions with: {args.model}")
    print(f"{'run_id':40s} {'hypotest':>16s}  {'biomni':>14s}")
    print("-" * 76)
    summary: dict[str, dict] = {}
    for p, d, res in results:
        run_id = p.parent.name
        old_raw, old_max, old_score = d.get("raw_score"), d.get("max_score"), d.get("score")
        if res is None:
            print(f"{run_id[:40]:40s} {'—':>16s}  {'(no rubric)':>14s}")
            continue
        new_total, new_score = res["total_score"], res["score"]
        delta = "↑" if new_score > (old_score or 0) else ("↓" if new_score < (old_score or 0) else "=")
        old_str = f"{old_score:5.2f} ({old_raw}/{old_max})" if old_score is not None else "—"
        print(f"{run_id[:40]:40s} {old_str:>16s}  {new_score:5.2f} ({new_total:3d}/100) {delta}")
        summary[run_id] = {
            "biomni_total_score": new_total,
            "biomni_score": new_score,
            "hypotest_raw_score": old_raw,
            "hypotest_max_score": old_max,
            "hypotest_score": old_score,
            "criteria": res["criteria"],
            "reasoning": res["reasoning"],
        }
        if args.write:
            (p.parent / "biomni_score.json").write_text(json.dumps(summary[run_id], indent=2))

    scored = [v["biomni_score"] for v in summary.values()]
    if scored:
        print("-" * 76)
        print(f"mean biomni score: {sum(scored) / len(scored):.3f}  over {len(scored)} graded")

    if args.write:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
        out = args.results.parent / f"biomni_scores.{safe}.json"
        out.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote per-dir biomni_score.json and summary -> {out}")
    else:
        print("\n(dry run — pass --write to save biomni_score.json per dir + a summary)")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Judge model (litellm name, e.g. openai/gpt-5)")
    ap.add_argument("--api-base", default=None, help="OpenAI-compatible endpoint base URL (local/vLLM)")
    ap.add_argument("--api-key", default=None, help="API key for --api-base (default: 'none')")
    ap.add_argument("--temperature", type=float, default=None, help="Sampling temperature for the judge")
    ap.add_argument("--concurrency", type=int, default=4, help="Max concurrent judge calls (results mode)")
    ap.add_argument("--write", nargs="?", const=True, default=False, help="Write outputs (off by default)")
    # results mode (default)
    ap.add_argument("--results", type=Path, default=ROOT / "results", help="Dir of <run_id>/score_info.json")
    # native mode
    ap.add_argument("--rubric", default=None, help="Native mode: path to tests/rubric.txt")
    ap.add_argument("--trace", default=None, help="Native mode: path to trace.md")
    ap.add_argument("--answer", default=None, help="Native mode: path to answer.txt")
    args = ap.parse_args()

    load_env()
    model = make_model(args.model, args.api_base, args.api_key, args.temperature)
    if args.rubric:
        await run_native(args, model)
    else:
        await run_results(args, model)


if __name__ == "__main__":
    asyncio.run(main())
