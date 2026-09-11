#!/usr/bin/env python3
"""Map each trajectory in trajectories.pkl to its results/<run_id>/score_info.json
by matching the agent's submitted answer against the rendered rubric prompt.
"""

import json
import pickle
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
trajs = pickle.load(open(ROOT / "benchmark_results/trajectories.pkl", "rb"))


def unwrap(a):
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


def final_answer(t):
    for s in reversed(t.steps):
        inner = getattr(s.action, "value", s.action)
        for tc in getattr(inner, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            if (getattr(fn, "name", "") or "") == "submit_answer":
                return unwrap(getattr(fn, "arguments", "{}"))
    return ""


from difflib import SequenceMatcher


def proposed_solution(prompt: str) -> str:
    m = re.search(r"<proposed-solution>\n(.*?)\n</proposed-solution>", prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


infos = []
for p in (ROOT / "results").glob("*/score_info.json"):
    d = json.loads(p.read_text())
    infos.append((p.parent.name, d, norm(proposed_solution(d["prompt"]))))

for t in trajs:
    ans = norm(final_answer(t))
    ranked = sorted(
        ((SequenceMatcher(None, ans, sol).ratio(), name, d) for name, d, sol in infos),
        key=lambda x: -x[0],
    )
    ratio, name, d = ranked[0]
    print(
        f"{t.traj_id:14s} traj_reward={t.steps[-1].reward}  ->  {name}"
        f"   score_info={d['score']} ({d['raw_score']}/{d['max_score']})   match={ratio:.3f}"
    )
