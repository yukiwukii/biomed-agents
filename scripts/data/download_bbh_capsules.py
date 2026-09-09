"""Download the 250 nvidia BBH-Train capsule data folders into capsules/bixbench-hypothesis/.

Downloads only `capsules/**` from nvidia/Nemotron-RL-bixbench_hypothesis, flattens the
nested `capsules/` level so each capsule lands at
capsules/bixbench-hypothesis/capsule_<uuid>/, and rewrites the jsonl's input_data_path
from `capsule_<uuid>.zip` to the on-disk directory name `capsule_<uuid>`.
"""

import json
import os
import shutil

from huggingface_hub import snapshot_download

REPO = "nvidia/Nemotron-RL-bixbench_hypothesis"
# Repo root, derived from this file's location (scripts/data/ -> ../..). Override
# with HYPOTEST_ROOT to write the capsules somewhere else.
ROOT = os.environ.get("HYPOTEST_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "capsules", "bixbench-hypothesis")
JSONL = os.path.join(OUT, "bixbench-hypothesis.jsonl")

os.makedirs(OUT, exist_ok=True)

print(f"[1/3] snapshot_download capsules/** -> {OUT}", flush=True)
snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    local_dir=OUT,
    allow_patterns=["capsules/**"],
    max_workers=8,
)

# Flatten OUT/capsules/capsule_* -> OUT/capsule_*
nested = os.path.join(OUT, "capsules")
if os.path.isdir(nested):
    print("[2/3] flattening nested capsules/ dir", flush=True)
    for name in os.listdir(nested):
        src = os.path.join(nested, name)
        dst = os.path.join(OUT, name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
    os.rmdir(nested)

# Rewrite input_data_path to the on-disk directory name (drop .zip)
print("[3/3] rewriting input_data_path in jsonl", flush=True)
rows = [json.loads(line) for line in open(JSONL)]
missing = 0
for r in rows:
    cap = r["input_data_path"]
    if cap.endswith(".zip"):
        cap = cap[:-4]
    r["input_data_path"] = cap
    if not os.path.isdir(os.path.join(OUT, cap)):
        missing += 1
with open(JSONL, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

n_dirs = len([d for d in os.listdir(OUT) if d.startswith("capsule_")])
print(f"DONE: {n_dirs} capsule dirs on disk; {missing} jsonl records with no matching dir", flush=True)
