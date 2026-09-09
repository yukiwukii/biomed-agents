"""Classify each rubric criterion in proposer/preview/hypotheses_generated.json
into one or more of six grading dimensions via an LLM call, adding a `categories`
list field in place. One model call per rubric (all its criteria classified together).

Usage:
    .venv/bin/python proposer/classify_rubric_criteria.py
"""
import argparse
import asyncio
import json
from pathlib import Path

from lmi import LiteLLMModel

from rubric_classifier import DEFAULT_MODEL, classify_many, load_env

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "proposer" / "agent" / "hypotheses_generated.json"
# DEFAULT_PATH = ROOT / "proposer"/ "preview" / "test.json"


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=Path, default=DEFAULT_PATH)
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

    with open(args.path) as f:
        data = json.load(f)

    # Collect one job per rubric (hypothesis + its criteria texts), keeping a handle back
    # to the actual criterion dicts (to write `categories` in place) and to the rubric dict
    # (to write the call's `categories_reasoning`, since one call classifies one rubric).
    jobs: list[tuple[str, list[str]]] = []
    targets: list[list[dict]] = []
    rubric_targets: list[dict] = []
    for item in data:
        rubrics = item.get("rubrics")
        if not rubrics:
            continue
        for rub in rubrics:
            criteria = rub["criteria"]
            jobs.append((rub["hypothesis"], [c["criterion"] for c in criteria]))
            targets.append(criteria)
            rubric_targets.append(rub)

    print(f"classifying {len(jobs)} rubrics ({sum(len(c) for c in targets)} criteria) with {args.model} ...")
    results = await classify_many(model, jobs, concurrency=args.concurrency)

    total = 0
    for criteria, rub, (categories, reasoning) in zip(targets, rubric_targets, results, strict=True):
        for c, cats in zip(criteria, categories, strict=True):
            c["categories"] = cats
            total += 1
        rub["categories_reasoning"] = reasoning

    with open(args.path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"done, classified {total} criteria in {args.path}")


if __name__ == "__main__":
    asyncio.run(main())
