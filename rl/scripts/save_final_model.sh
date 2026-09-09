#!/usr/bin/env bash
# rl/scripts/save_final_model.sh
# =============================================================================
# Copy a training checkpoint out of the rotating checkpoint directory into a
# stable, named location that nothing will delete.
#
#   ./rl/scripts/save_final_model.sh                    # latest step -> rl/models/<timestamp>
#   ./rl/scripts/save_final_model.sh --name qwen9b-v1   # name it
#   ./rl/scripts/save_final_model.sh --step 9           # a specific step
#
# WHY THIS EXISTS
# ---------------
# `checkpointing.keep_top_k: 2` means the trainer DELETES older checkpoints as
# it writes new ones. That is correct for disk hygiene and wrong for "keep the
# model I just trained" -- the run's own output is a rotating buffer, not an
# archive. This copies one out from under the rotation.
#
# It also copies rather than moves: a run that is still going must keep its
# checkpoint history intact for resume-after-preemption.
#
# WHAT YOU GET
# ------------
# With lora_cfg enabled, NeMo RL writes a PEFT adapter (is_peft is set
# automatically, automodel/checkpoint.py:239) and, with save_consolidated: true,
# the HF metadata alongside it. The result loads without NeMo RL:
#
#     from peft import PeftModel
#     from transformers import AutoModelForCausalLM
#     base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B")
#     model = PeftModel.from_pretrained(base, "<this directory>/policy/weights/model")
#
# VERIFIED against step_3 of the 2026-08-11 preflight: adapter_config.json
# (peft_type LORA, r 64, alpha 128, base Qwen/Qwen3.5-9B, task CAUSAL_LM,
# 32 target modules) + adapter_model.safetensors, 64 tensors / 15,728,640
# params fp32 (~63 MB). It is an ADAPTER, not a standalone model, and still
# needs the base checkpoint.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_ROOT="${HYPOTEST_CKPT_DIR:-${REPO_ROOT}/rl/results/ckpt/grpo-hypotest-9b}"
DEST_ROOT="${HYPOTEST_MODEL_DIR:-${REPO_ROOT}/rl/models}"

grn() { printf '\033[32m%s\033[0m\n' "$*"; }
red() { printf '\033[31m%s\033[0m\n' "$*" >&2; }
die() { red "FATAL: $*"; exit 1; }

NAME=""
STEP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    -h|--help) sed -n '2,35p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ -d "${CKPT_ROOT}" ] || die "no checkpoint directory at ${CKPT_ROOT}
Either the run has not saved yet (save_period: 3, so first save is step 3), or
HYPOTEST_CKPT_DIR points elsewhere. Check with:
  kubectl -n \$NS exec hypotest-env-0-0 -- ls -la ${CKPT_ROOT}"

# NeMo RL lays checkpoints out as step_N/ under the checkpoint root.
if [ -n "${STEP}" ]; then
  SRC="${CKPT_ROOT}/step_${STEP}"
  [ -d "${SRC}" ] || die "step_${STEP} not found. Available: $(ls "${CKPT_ROOT}" | tr '\n' ' ')"
else
  # Highest step number, numerically -- NOT lexically ("step_9" > "step_10" in
  # a plain sort, which would silently save an older model.
  SRC="$(find "${CKPT_ROOT}" -maxdepth 1 -type d -name 'step_*' \
         | sed 's/.*step_//' | sort -n | tail -1 | sed "s|^|${CKPT_ROOT}/step_|")"
  [ -n "${SRC}" ] && [ -d "${SRC}" ] || die "no step_* directories under ${CKPT_ROOT}"
fi

STEP_NUM="$(basename "${SRC}" | sed 's/step_//')"
[ -n "${NAME}" ] || NAME="qwen3.5-9b-grpo-step${STEP_NUM}-$(date +%Y%m%d-%H%M%S)"
DEST="${DEST_ROOT}/${NAME}"

[ -e "${DEST}" ] && die "${DEST} already exists -- refusing to overwrite. Pass a different --name."

grn ">> source:      ${SRC}  ($(du -sh "${SRC}" 2>/dev/null | cut -f1))"
grn ">> destination: ${DEST}"

mkdir -p "${DEST_ROOT}"
# -a preserves timestamps; copy to a temp name first so an interrupted copy
# never looks like a complete saved model.
cp -a "${SRC}" "${DEST}.partial"
mv "${DEST}.partial" "${DEST}"

# Record what this actually is. A bare adapter directory six months from now is
# unidentifiable otherwise.
cat > "${DEST}/PROVENANCE.txt" <<EOF
Qwen3.5-9B + LoRA, GRPO (hypotest)
saved:        $(date -u +%Y-%m-%dT%H:%M:%SZ)
source:       ${SRC}
step:         ${STEP_NUM}
base model:   Qwen/Qwen3.5-9B
adapter:      LoRA r=64 alpha=128 on *language_model*{q,k,v,o}_proj
              8 full-attention layers x 4 projections = 32 modules, 15,728,640 params fp32
config:       rl/configs/grpo_hypotest_27b.yaml
load with:
  from peft import PeftModel
  from transformers import AutoModelForCausalLM
  base  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B")
  model = PeftModel.from_pretrained(base, "${DEST}/policy/weights/model")
NOTE: this is an ADAPTER (15.7M params, ~63 MB), not a standalone model.
EOF

grn ">> saved. contents:"
find "${DEST}" -maxdepth 2 | head -20 | sed 's/^/     /'
grn ">> total: $(du -sh "${DEST}" | cut -f1)"
