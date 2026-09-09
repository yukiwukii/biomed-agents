#!/usr/bin/env python3
"""Re-grade benchmarked trajectories with a *different* rubric model.

Grading in hypotest is a single LLM call over a self-contained prompt
(`RUBRIC_SCORE_PROMPT` rendered with rubric + notebook + final answer). That
rendered prompt is saved verbatim to ``results/<run_id>/score_info.json`` under
the ``"prompt"`` key, so re-grading needs no agent re-run and no environment —
just call a new model on the saved prompt and recompute the score the same way
``InterpreterEnv._score_solution`` does (``raw_score / max_score``, clamped).

Each trajectory in ``trajectories.pkl`` is matched to its ``score_info.json`` by
the agent's submitted answer (the ``iterN`` dir suffix does NOT line up with the
``repN`` trajectory order because replications run concurrently).

Usage:
    conda run -n bixbench python3 scripts/eval/regrade.py \
        --model anthropic/claude-opus-4-8

    # OpenAI-compatible / local vLLM endpoint:
    conda run -n bixbench python3 scripts/eval/regrade.py \
        --model openai/Qwen/Qwen3.6-27B \
        --api-base http://localhost:8000/v1 --api-key none

    # also write outputs (new rewards json + patched pkl):
    conda run -n bixbench python3 scripts/eval/regrade.py --model ... --write

Nothing is written unless ``--write`` is passed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import operator
import pickle
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from lmi import LiteLLMModel
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]


# Mirror of hypotest.env.interpreter_env.{StepEvidence,CriterionScore,RubricScore}.
# Redefined locally so the script stays lightweight and avoids the env module's
# heavy (docker/jupyter) import side effects. Keep in sync if the schema changes.
class StepEvidence(BaseModel):
    step: int
    note: str
    correct: bool


class CriterionScore(BaseModel):
    criterion: str
    score: int
    justification: str
    relevant_steps: list[StepEvidence] = []
    # first_wrong_step disabled for now — see the note above _RUBRIC_SCORE_PROMPT_TAIL in prompts.py.
    # first_wrong_step: int | None = None


class RubricScore(BaseModel):
    criteria: list[CriterionScore]


# ── trajectory ↔ score_info matching ─────────────────────────────────────────


def unwrap(a):
    """Unwrap a possibly multiply-JSON-wrapped submit_answer argument to plain text."""
    for _ in range(3):
        if isinstance(a, str):
            s = a.strip()
            if s.startswith("{") and "answer" in s[:15]:
                try:
                    a = json.loads(s)["answer"]
                    continue
                except Exception:
                    break
        elif isinstance(a, dict) and "answer" in a:
            a = a["answer"]
            continue
        break
    return a if isinstance(a, str) else json.dumps(a)


def final_answer(traj) -> str:
    for s in reversed(traj.steps):
        inner = getattr(s.action, "value", s.action)
        for tc in getattr(inner, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            if (getattr(fn, "name", "") or "") == "submit_answer":
                return unwrap(getattr(fn, "arguments", "{}"))
    return ""


def proposed_solution(prompt: str) -> str:
    m = re.search(r"<proposed-solution>\n(.*?)\n</proposed-solution>", prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def rubric_text(prompt: str) -> str:
    """Recover the rubric from a saved prompt, for the full-marks gate in derive_first_wrong_step."""
    m = re.search(r"<rubric>\n(.*?)\n</rubric>", prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def build_mapping(trajs, results_dir: Path):
    """Return list of (traj, run_dir_name, score_info_dict_or_None, match_ratio).

    A trajectory whose agent never submitted an answer (empty ``final_answer``)
    cannot be matched to a saved prompt — its best SequenceMatcher ratio against
    every real solution is 0.0, so it would otherwise mis-match an arbitrary
    prompt for a *different* task. Such entries get ``(t, "(no answer)", None, 0.0)``
    and are skipped during grading (forced to score 0).
    """
    infos = []
    skipped_no_prompt = 0
    for p in results_dir.glob("*/score_info.json"):
        d = json.loads(p.read_text())
        # [PATCH 24] Deterministically-scored runs (grading_method "bioagent") save no prompt —
        # there is no LLM judgment to replay, so they are not re-gradable.
        if "prompt" not in d:
            skipped_no_prompt += 1
            continue
        infos.append((p.parent.name, d, norm(proposed_solution(d["prompt"]))))
    if skipped_no_prompt:
        print(f"Skipped {skipped_no_prompt} deterministically-scored run(s) — no judge prompt to re-grade.")

    mapping = []
    for t in trajs:
        ans = norm(final_answer(t))
        if not ans:
            mapping.append((t, "(no answer)", None, 0.0))
            continue
        ratio, name, d = max(
            ((SequenceMatcher(None, ans, sol).ratio(), name, d) for name, d, sol in infos),
            key=operator.itemgetter(0),
        )
        mapping.append((t, name, d, ratio))
    return mapping


# ── scoring (mirrors InterpreterEnv._score_solution) ─────────────────────────


# Mirror of hypotest.env.interpreter_env.{parse_criterion_max_scores,derive_first_wrong_step}
# — the judge no longer emits this field, so it is computed from the relevant_steps it did
# emit. Keep in sync if the derivation changes.
_HYPOTEST_CRITERION_POINTS = re.compile(
    r"^\s*(?:\d+\.\s*\(\s*(\d+)\s*points?\s*\)|[-*]\s*(\d+)\s*points?\s*:)", re.MULTILINE
)
# `-?` is required: BiomniBench-DA rubrics end with a penalty criterion whose levels are
# negative (`Levels: A=0 B=-5 C=-10`). See biomni_judge.parse_rubric_levels.
_BIOMNI_LEVELS = re.compile(r"Levels:\s*((?:[A-Z]=-?\d+\s*)+)")


def parse_criterion_max_scores(rubric: str) -> list[int]:
    """Per-criterion maximum points, in rubric order; ``[]`` if the rubric isn't parseable.

    Covers both hypotest criterion layouts: numbered ``N. (X points) …`` and bixbench's
    ``* X points: …`` bullets.
    """
    levels = [max(int(v) for v in re.findall(r"[A-Z]=(-?\d+)", m.group(1))) for m in _BIOMNI_LEVELS.finditer(rubric)]
    return levels or [int(m.group(1) or m.group(2)) for m in _HYPOTEST_CRITERION_POINTS.finditer(rubric)]


def derive_first_wrong_step(criteria: list[dict], rubric: str | None = None) -> list[dict]:
    """Inject ``"first_wrong_step"`` — the earliest relevant step with ``"correct": false``.

    A criterion awarded full marks gets ``None`` when the rubric's per-criterion maximum
    is parseable; otherwise the gate is skipped.
    """
    max_scores = parse_criterion_max_scores(rubric) if rubric else []
    for i, c in enumerate(criteria):
        max_pts = max_scores[i] if i < len(max_scores) else None
        score = c.get("score")
        if max_pts is not None and isinstance(score, (int, float)) and not isinstance(score, bool) and score >= max_pts:
            c["first_wrong_step"] = None
            continue
        wrong = [
            s.get("step")
            for s in c.get("relevant_steps") or []
            if isinstance(s, dict) and not s.get("correct", True) and isinstance(s.get("step"), int)
        ]
        c["first_wrong_step"] = min(wrong) if wrong else None
    return criteria


def compute_score(rubric: RubricScore, max_score: int, normalize: bool) -> tuple[int, float]:
    raw = sum(c.score for c in rubric.criteria)
    score = raw / max_score if normalize else raw
    score = max(0.0, min(1.0 if normalize else float(max_score), score))
    return raw, score


async def grade_one(model: LiteLLMModel, prompt: str) -> RubricScore:
    resp = await model.call_single(prompt, output_type=RubricScore, timeout=3 * 60)
    if not resp.text:
        raise ValueError("No response from rubric model")
    json_start = resp.text.index("{")
    data = json.loads(resp.text[json_start:])
    # Some models wrap the structured output in a tool-call-style {"parameters": {...}}
    # envelope; unwrap it so the top-level object carries "criteria".
    if "criteria" not in data and isinstance(data.get("parameters"), dict):
        data = data["parameters"]
    return RubricScore.model_validate(data)


def make_model(
    name: str,
    api_base: str | None,
    api_key: str | None,
    temperature: float | None,
    reasoning_effort: str | None,
) -> LiteLLMModel:
    # Mirror the benchmark judge (dataset_server.py): reasoning_effort is passed
    # through litellm_params with drop_params=True so models that don't support
    # it silently ignore it.
    params: dict[str, Any] = {"model": name, "drop_params": True}
    if api_base:
        params["api_base"] = api_base
        params["api_key"] = api_key or "none"
    if temperature is not None:
        params["temperature"] = temperature
    if reasoning_effort is not None:
        params["reasoning_effort"] = reasoning_effort
    config = {"model_list": [{"model_name": name, "litellm_params": params}]}
    return LiteLLMModel(name=name, config=config)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="New rubric model (litellm name, e.g. anthropic/claude-opus-4-8)")
    ap.add_argument("--api-base", default=None, help="OpenAI-compatible endpoint base URL (for local/vLLM models)")
    ap.add_argument("--api-key", default=None, help="API key for --api-base (default: 'none')")
    ap.add_argument("--pkl", type=Path, default=ROOT / "benchmark_results/trajectories.pkl")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--concurrency", type=int, default=4, help="Max concurrent rubric calls")
    ap.add_argument("--temperature", type=float, default=None, help="Sampling temperature for the rubric model")
    ap.add_argument(
        "--reasoning-effort",
        default="high",
        help="Reasoning effort for the rubric model, matching the benchmark judge (default: high)",
    )
    ap.add_argument(
        "--no-normalize",
        action="store_true",
        help="Report raw_score instead of raw_score/max_score (matches normalize_reward=False)",
    )
    ap.add_argument("--min-match", type=float, default=0.95, help="Warn if best answer match ratio is below this")
    ap.add_argument("--write", action="store_true", help="Write new rewards json + patched pkl (off by default)")
    args = ap.parse_args()

    normalize = not args.no_normalize

    trajs = pickle.loads(args.pkl.read_bytes())
    mapping = build_mapping(trajs, args.results)

    model = make_model(args.model, args.api_base, args.api_key, args.temperature, args.reasoning_effort)
    sem = asyncio.Semaphore(args.concurrency)

    async def run(prompt: str):
        async with sem:
            return await grade_one(model, prompt)

    # Only call the rubric model for trajectories that submitted an answer (d is
    # not None). Trajectories with no answer are forced to score 0 below.
    gradable = [(i, d) for i, (_, _, d, _) in enumerate(mapping) if d is not None]
    rubrics = await asyncio.gather(*(run(d["prompt"]) for _, d in gradable))
    rubric_by_idx = {i: r for (i, _), r in zip(gradable, rubrics, strict=True)}

    n_skipped = len(mapping) - len(gradable)
    print(f"\nRe-grading {len(gradable)} trajectories with: {args.model}", end="")
    print(f"  ({n_skipped} skipped — no submitted answer)" if n_skipped else "")
    print(f"{'trajectory':14s} {'run_id':18s} {'match':>6s}  {'old':>12s}  {'new':>12s}")
    print("-" * 72)

    new_rewards: dict[str, float] = {}
    judge_output: dict[str, dict] = {}
    for i, (t, name, d, ratio) in enumerate(mapping):
        if d is None:
            new_rewards[t.traj_id] = 0.0
            print(f"{t.traj_id:14s} {name:18s} {ratio:6.3f}  {'—':>12s}  {0.0:5.2f} (no answer)  ·")
            continue
        max_score = d["max_score"]
        rubric = rubric_by_idx[i]
        new_raw, new_score = compute_score(rubric, max_score, normalize)
        old_score = d["score"]
        old_raw = d["raw_score"]
        new_rewards[t.traj_id] = new_score
        judge_output[t.traj_id] = {
            "run_id": name,
            "model": args.model,
            "match_ratio": ratio,
            "max_score": max_score,
            "raw_score": new_raw,
            "score": new_score,
            "old_raw_score": old_raw,
            "old_score": old_score,
            "criteria": derive_first_wrong_step([c.model_dump() for c in rubric.criteria], rubric_text(d["prompt"])),
            "old_criteria": d.get("criteria"),
        }
        flag = "  <-- low match!" if ratio < args.min_match else ""
        delta = "  ↑" if new_score > old_score else ("  ↓" if new_score < old_score else "  =")
        print(
            f"{t.traj_id:14s} {name[:8]:18s} {ratio:6.3f}  "
            f"{old_score:5.2f} ({old_raw:2}/{max_score:2})  "
            f"{new_score:5.2f} ({new_raw:2}/{max_score:2}){delta}{flag}"
        )

    if not args.write:
        print("\n(dry run — pass --write to save new rewards json and patched pkl)")
        return

    out_dir = args.pkl.parent
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    rewards_path = out_dir / f"rewards.regrade.{safe_model}.json"
    rewards_path.write_text(json.dumps(new_rewards, indent=2))

    judge_path = out_dir / f"judge_output.regrade.{safe_model}.json"
    judge_path.write_text(json.dumps(judge_output, indent=2))

    # Patch terminal-step reward on a copy of the trajectories and re-pickle.
    for t, *_ in mapping:
        if t.steps:
            t.steps[-1].reward = new_rewards[t.traj_id]
    pkl_path = out_dir / f"trajectories.regrade.{safe_model}.pkl"
    pkl_path.write_bytes(pickle.dumps(trajs))

    print(f"\nWrote {rewards_path}")
    print(f"Wrote {judge_path}")
    print(f"Wrote {pkl_path}")


if __name__ == "__main__":
    asyncio.run(main())
