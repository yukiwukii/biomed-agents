"""Convert proposer hypotheses_generated.json -> problems.jsonl (ProblemInstance schema).

One JSONL line per (capsule, hypothesis-rubric). Uses task_style="question" so the
open-ended hypotest judge grades against the rubric with no accept/reject framing.

Usage:
    .venv/bin/python proposer/proposer_to_problems.py \
        proposer/preview/hypotheses_generated.json problems_proposer.jsonl \
        --capsule-dir capsules
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

# Fixed namespace so re-runs produce stable per-hypothesis UUIDs.
_NS = uuid.UUID("6f1a0000-0000-4000-8000-000000000001")


def format_rubric(hypothesis: str, criteria: list[dict]) -> tuple[str, int]:
    """Render structured criteria into a free-text rubric string + total points.

    Deliberately does NOT use the biomni 'Criterion N: ... Levels: A=X' format, so the
    'auto' judge detector never misclassifies it as a biomni A/B/C rubric.
    """
    lines = [f"RUBRIC for hypothesis: {hypothesis}", ""]
    total = 0
    for i, c in enumerate(criteria, 1):
        pts = int(c.get("points", 1))
        total += pts
        cats = ", ".join(c.get("categories", []))
        cat_suffix = f"  [{cats}]" if cats else ""
        lines.append(f"{i}. ({pts} point{'s' if pts != 1 else ''}) {c['criterion']}{cat_suffix}")
    lines += ["", f"Total possible points: {total}"]
    return "\n".join(lines), total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--capsule-dir", type=Path, default=Path("capsules"))
    ap.add_argument(
        "--max-per-capsule",
        type=int,
        default=None,
        help="Keep at most N hypotheses per capsule (e.g. 1 for one-per-capsule). Default: all.",
    )
    args = ap.parse_args()

    capsules = json.loads(args.input.read_text())
    rows: list[dict] = []
    skipped_no_data = 0

    for cap in capsules:
        cid = cap["capsule_id"]
        if cap.get("status") != "ok" or not cap.get("rubrics"):
            continue
        data_dir = f"CapsuleData-{cid}"
        if not (args.capsule_dir / data_dir).is_dir():
            skipped_no_data += 1
            continue
        kept = 0
        for j, rub in enumerate(cap["rubrics"]):
            if args.max_per_capsule is not None and kept >= args.max_per_capsule:
                break
            criteria = rub.get("criteria") or []
            if not criteria:
                continue
            kept += 1
            hyp = rub["hypothesis"]
            rubric_text, total = format_rubric(hyp, criteria)
            pid = uuid.uuid5(_NS, f"{cid}:{j}")
            rows.append(
                {
                    "id": str(pid),
                    "hypothesis": hyp,
                    "protocol": "",
                    "answer": None,
                    "rubric": rubric_text,
                    "max_points": total,
                    "input_data_path": data_dir,
                    "task_style": "question",
                    "nb_primary_language": "python",
                    "metadata": {
                        "source": "proposer/hypotheses_generated",
                        "capsule_id": cid,
                        "hypothesis_index": j,
                        "rubric_mode": cap.get("rubric_mode"),
                        "expert_hypothesis": cap.get("expert_hypothesis"),
                    },
                }
            )

    with args.output.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_caps = len({r["metadata"]["capsule_id"] for r in rows})
    print(f"Wrote {len(rows)} problems from {n_caps} capsules -> {args.output}")
    if skipped_no_data:
        print(f"Skipped {skipped_no_data} capsules with no matching capsule data dir under {args.capsule_dir}/")


if __name__ == "__main__":
    main()
