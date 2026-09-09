#!/usr/bin/env bash
# rl/scripts/00_smoke_rollout.sh
# =============================================================================
# GATE 3: collect rollouts through the full NeMo Gym stack, with NO training.
#
# This is the cheap gate. It exercises everything that GRPO depends on —
# ng_run's server orchestration, the aviary agent loop, the remote resources
# server, the reward round-trip — without loading a trainer, an optimizer, or a
# reference policy. If this does not produce non-zero rewards, GRPO will not
# either; it will just take an hour longer to tell you.
#
#   bash rl/scripts/00_smoke_rollout.sh
#
# Env:
#   HYPOTEST_SERVER_URL   required   http://hypotest-env:8008
#   HYPOTEST_API_KEY      required   matches server.yaml
#   NEMO_GYM_ROOT         default /opt/Gym
#   SMOKE_INPUT           default /config/rl/data/tiny.jsonl
#   SMOKE_OUT             default results/smoke/rollouts.jsonl
#   SMOKE_REPEATS         default 4    repeats per task — must be >1 or the
#                                      reward-variance check in [1] is vacuous
#   SMOKE_LIMIT           default 4    tasks
#   AGENT_MAX_STEPS       default 20
#   BBH_POLICY_MODEL      default Qwen/Qwen3.5-9B
# =============================================================================
set -euo pipefail

grn() { printf '\033[32m%s\033[0m\n' "$*"; }
red() { printf '\033[31m%s\033[0m\n' "$*" >&2; }
die() { red "FATAL: $*"; exit 1; }
need() { [ -n "${!1:-}" ] || die "$1 is not set."; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="${HYPOTEST_CONFIG_DIR:-${REPO_ROOT}/rl/configs}"
NEMO_GYM_ROOT="${NEMO_GYM_ROOT:-/opt/Gym}"

SMOKE_INPUT="${SMOKE_INPUT:-${REPO_ROOT}/rl/data/tiny.jsonl}"
SMOKE_OUT="${SMOKE_OUT:-${REPO_ROOT}/results/smoke/rollouts.jsonl}"
SMOKE_REPEATS="${SMOKE_REPEATS:-4}"
SMOKE_LIMIT="${SMOKE_LIMIT:-4}"
SERVER_WARMUP_SECS="${SERVER_WARMUP_SECS:-1800}"

export AGENT_MAX_STEPS="${AGENT_MAX_STEPS:-20}"
export BBH_POLICY_MODEL="${BBH_POLICY_MODEL:-Qwen/Qwen3.5-9B}"
export HYPOTEST_GYM_CONFIG="${HYPOTEST_GYM_CONFIG:-${CONFIG_DIR}/gym_hypotest_remote.yaml}"

need HYPOTEST_SERVER_URL
need HYPOTEST_API_KEY
[ -d "${NEMO_GYM_ROOT}" ] || die "NEMO_GYM_ROOT='${NEMO_GYM_ROOT}' is not a directory."
[ -f "${SMOKE_INPUT}" ] || die "${SMOKE_INPUT} missing. Run rl/data/make_splits.py."
[ -f "${HYPOTEST_GYM_CONFIG}" ] || die "${HYPOTEST_GYM_CONFIG} missing."
[ -f "${CONFIG_DIR}/gym_policy_smoke.yaml" ] || die "${CONFIG_DIR}/gym_policy_smoke.yaml missing."
command -v ng_run >/dev/null 2>&1 || die "ng_run not on PATH. Activate the NeMo Gym venv."
command -v ng_collect_rollouts >/dev/null 2>&1 || die "ng_collect_rollouts not on PATH."

mkdir -p "$(dirname "${SMOKE_OUT}")"
NG_LOG="$(dirname "${SMOKE_OUT}")/ng_run.log"
: > "${NG_LOG}"

# ---------------------------------------------------------------------------
# GATE 2 inline: fail fast on an unreachable env server.
#
# Worth doing explicitly because the failure mode otherwise is silent — the
# agent loop catches the connection error, ends the episode, and reports
# reward 0.0. You would see "the model is bad" instead of "nothing was running".
# ---------------------------------------------------------------------------
grn ">> Checking the dataset server at ${HYPOTEST_SERVER_URL}"
INFO="$(curl -fsS -m 15 -H "X-API-Key: ${HYPOTEST_API_KEY}" "${HYPOTEST_SERVER_URL}/info" 2>/dev/null)" \
  || die "GET ${HYPOTEST_SERVER_URL}/info failed. Is Workload A running, and does HYPOTEST_API_KEY match server.yaml?"
grn "   /info -> ${INFO}"
DATASET_SIZE="$(printf '%s' "${INFO}" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("dataset_size", 0))')"
[ "${DATASET_SIZE}" -gt 0 ] || die "server reports dataset_size=0 — check capsule_dir / problem source in server.yaml."

# task_idx is positional into the server's problem list, so an index past the
# end is a silent mis-train, not an error. Check it here.
MAX_IDX="$(python3 -c "
import json,sys
idxs=[json.loads(l)['task_idx'] for l in open('${SMOKE_INPUT}') if l.strip()]
print(max(idxs) if idxs else -1)")"
[ "${MAX_IDX}" -lt "${DATASET_SIZE}" ] \
  || die "${SMOKE_INPUT} references task_idx=${MAX_IDX} but the server only has ${DATASET_SIZE} problems. Regenerate splits (rl/data/make_splits.py) against the server's current config."
grn "   split max task_idx=${MAX_IDX} < dataset_size=${DATASET_SIZE}  OK"

# ---------------------------------------------------------------------------
# Launch the three servers.
#
# Two config files: the remote env wiring, plus a STANDALONE policy. During
# training NeMo RL supplies the policy instead (vllm_model_for_training.yaml) —
# gym_policy_smoke.yaml is only ever used here.
# ---------------------------------------------------------------------------
CFG="[${HYPOTEST_GYM_CONFIG},${CONFIG_DIR}/gym_policy_smoke.yaml]"
grn ">> Launching NeMo Gym servers (config_paths=${CFG})"
grn "   log: ${NG_LOG}"

( cd "${NEMO_GYM_ROOT}" && exec ng_run "+config_paths=${CFG}" ) 2>&1 | tee -a "${NG_LOG}" &
TEE_PID=$!

resolve_ng_pid() { pgrep -f '[n]g_run' | head -1 || true; }
NG_PID="$(resolve_ng_pid)"

shutdown_ng() {
  [ "${NG_SHUTDOWN_DONE:-0}" = 1 ] && return 0
  NG_SHUTDOWN_DONE=1
  local pid; pid="$(resolve_ng_pid)"
  if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
    grn ">> Shutting down ng_run (pid ${pid})"
    # SIGINT reaches RunHelper.shutdown(); SIGTERM is intercepted by Ray and is noisy.
    kill -INT "${pid}" 2>/dev/null || true
    local waited=0
    while kill -0 "${pid}" 2>/dev/null && [ "${waited}" -lt 180 ]; do sleep 2; waited=$((waited + 2)); done
    kill -0 "${pid}" 2>/dev/null && { red ">> ng_run did not exit in 180s; SIGKILL"; kill -9 "${pid}" 2>/dev/null || true; }
  fi
  command -v ray >/dev/null 2>&1 && ray stop --force >/dev/null 2>&1 || true
}
trap shutdown_ng EXIT

grn ">> Waiting for 3/3 servers (timeout ${SERVER_WARMUP_SECS}s; the 9B vLLM load dominates)"
elapsed=0
while :; do
  NG_PID="$(resolve_ng_pid)"
  if [ -z "${NG_PID}" ]; then
    red ">> ng_run exited before the servers were ready. Tail of ${NG_LOG}:"
    tail -n 80 "${NG_LOG}" >&2 || true
    die "ng_run died early — see ${NG_LOG}"
  fi
  grep -qE 'All 3 / 3 servers ready' "${NG_LOG}" 2>/dev/null && { grn ">> All 3 servers up"; break; }
  [ "${elapsed}" -ge "${SERVER_WARMUP_SECS}" ] && {
    tail -n 80 "${NG_LOG}" >&2 || true
    die "servers not ready after ${SERVER_WARMUP_SECS}s — see ${NG_LOG}"
  }
  sleep 10; elapsed=$((elapsed + 10))
done

# ---------------------------------------------------------------------------
# Collect. num_repeats > 1 is required, not cosmetic: reward variance *within a
# task group* is what GRPO's advantage is computed from, and a single rollout
# per task cannot show whether that variance exists.
# ---------------------------------------------------------------------------
grn ">> Collecting ${SMOKE_LIMIT} tasks x ${SMOKE_REPEATS} repeats -> ${SMOKE_OUT}"
( cd "${NEMO_GYM_ROOT}" && ng_collect_rollouts \
    +agent_name=hypotest_agent \
    "+input_jsonl_fpath=${SMOKE_INPUT}" \
    "+output_jsonl_fpath=${SMOKE_OUT}" \
    "+limit=${SMOKE_LIMIT}" \
    "+num_repeats=${SMOKE_REPEATS}" \
    "+num_samples_in_parallel=${SMOKE_PARALLEL:-${SMOKE_LIMIT}}" ) \
  || die "ng_collect_rollouts failed — see output above and ${NG_LOG}"

shutdown_ng

grn ""
grn ">> Analysis"
python3 "${REPO_ROOT}/rl/scripts/analyze_rollouts.py" "${SMOKE_OUT}" \
  --tokenizer "${BBH_POLICY_MODEL}" \
  --max-seq-len "${SMOKE_MAX_SEQ_LEN:-32768}"

grn ""
grn "Rollouts:  ${SMOKE_OUT}"
grn "ng_run log: ${NG_LOG}"
grn "Judge detail is server-side, under the env pod's save_dir."
