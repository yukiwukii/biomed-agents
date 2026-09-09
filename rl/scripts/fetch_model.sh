#!/usr/bin/env bash
# rl/scripts/fetch_model.sh
# =============================================================================
# Pre-download a model into a world-READABLE HF cache on the scratch PVC, so
# training pods stop re-downloading it on every run.
#
#   ./rl/scripts/fetch_model.sh Qwen/Qwen3.5-9B
#   ./rl/scripts/fetch_model.sh Qwen/Qwen3.6-35B-A3B
#   ./rl/scripts/fetch_model.sh            # defaults to the 9B
#
# WHY THIS EXISTS
# ---------------
# POD_HF_HOME was a pod-local emptyDir because the PVC's hf_cache/ is mode 0770
# and the trainer, running root-squashed as `nobody`, cannot write it. That is
# correct but wasteful: 19.3 GB re-downloaded per 9B run, 71.9 GB per 35B run.
#
# The fix is the same one used for the interpreter (see make_rl_runtime.sh):
# the *user* writes it once from this host, `chmod a+rX` makes it readable by
# the squashed pod, and the pod runs with HF_HUB_OFFLINE=1 so it only ever
# reads. No pod needs the user's identity and no existing permission changes --
# this is a new directory.
#
# Re-run to add a model or refresh a revision. Existing entries are reused, so
# a second run is nearly free.
# =============================================================================
set -euo pipefail

REPO="${1:-Qwen/Qwen3.5-9B}"
SHARED="${HF_SHARED_CACHE:-/mlbio_scratch/wangsaja/hf_shared}"
PY="${PY:-/mlbio_scratch/wangsaja/hypotest/.venv/bin/python}"

log() { printf '\033[36m[fetch_model]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[31m[fetch_model] FATAL:\033[0m %s\n' "$*" >&2; exit 1; }

[ -x "${PY}" ] || die "python not found at ${PY}"
mkdir -p "${SHARED}/hub"

log "downloading ${REPO} -> ${SHARED}/hub"
HF_HUB_DISABLE_XET=1 "${PY}" - "$REPO" "$SHARED" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, shared = sys.argv[1], sys.argv[2]
p = snapshot_download(
    repo_id=repo,
    cache_dir=f"{shared}/hub",
    # Weights + config + tokenizer only. No .pt/.bin duplicates of safetensors,
    # which on these repos would double the footprint for nothing.
    allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model"],
    max_workers=8,
)
print(p)
PY

log "chmod a+rX (this is what lets the root-squashed pod read it)"
chmod -R a+rX "${SHARED}"

blocked=$(find "${SHARED}" -type d ! -perm -o+x 2>/dev/null | wc -l)
[ "${blocked}" = 0 ] || die "${blocked} dirs are still not world-traversable."

log "OK -> ${SHARED}  ($(du -sh "${SHARED}" 2>/dev/null | cut -f1))"
log "workloads.sh picks this up automatically; it sets HF_HUB_OFFLINE=1 so the"
log "pod never attempts a write into the read-only cache."
