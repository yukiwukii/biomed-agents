"""Build ./original_rubrics.json from the EdisonScientific/bixbench_hypothesis
HF dataset: one entry per capsule with its expert hypothesis and ground-truth
rubric, each rubric criterion classified into one or more of six grading
dimensions via an LLM call (one call per rubric, all its criteria classified together).

Usage:
    .venv/bin/python scripts/classify_original_rubrics.py
"""
import argparse
import asyncio
import json
import re
from pathlib import Path

from datasets import load_dataset
from lmi import LiteLLMModel

from rubric_classifier import DEFAULT_MODEL, classify_many, load_env

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = "EdisonScientific/bixbench_hypothesis"
DEFAULT_OUT = ROOT / "original_rubrics.json"

BULLET_RE = re.compile(r"^\*\s*(\d+)\s*points?:\s*(.*)", re.DOTALL)


def parse_rubric(rubric_text: str) -> list[dict]:
    """Split the dataset's `* N point(s): ...` rubric string into criterion dicts."""
    criteria = []
    for line in rubric_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = BULLET_RE.match(line)
        if not m:
            raise ValueError(f"unparsable rubric bullet: {line!r}")
        criteria.append({"points": int(m.group(1)), "criterion": m.group(2).strip()})
    return criteria


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    load_env()
    # High reasoning effort so classification reasons through multi-label criteria.
    # reasoning_effort MUST live inside litellm_params (like server.yaml): lmi only
    # forwards a whitelist (model/n/temperature/max_tokens) from top-level config, so a
    # bare {"reasoning_effort": ...} is silently dropped and no thinking happens.
    model = LiteLLMModel(
        name=args.model,
        config={
            "model_list": [
                {
                    "model_name": args.model,
                    "litellm_params": {"model": args.model, "reasoning_effort": "high"},
                }
            ],
        },
    )

    ds = load_dataset(args.dataset, split="train")

    out = []
    jobs: list[tuple[str, list[str]]] = []
    targets: list[list[dict]] = []
    for row in ds:
        criteria = parse_rubric(row["rubric"])
        out.append({
            "capsule_id": row["id"],
            "expert_hypothesis": row["hypothesis"],
            "criteria": criteria,
        })
        jobs.append((row["hypothesis"], [c["criterion"] for c in criteria]))
        targets.append(criteria)

    print(f"classifying {len(jobs)} rubrics ({sum(len(c) for c in targets)} criteria) with {args.model} ...")
    results = await classify_many(model, jobs, concurrency=args.concurrency)

    total = 0
    for entry, criteria, (categories, reasoning) in zip(out, targets, results, strict=True):
        for c, cats in zip(criteria, categories, strict=True):
            c["categories"] = cats
            total += 1
        entry["categories_reasoning"] = reasoning

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"wrote {len(out)} capsules, {total} classified criteria to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
