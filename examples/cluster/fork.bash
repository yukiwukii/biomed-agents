#!/usr/bin/env bash
#
# ─────────────────────────────────────────────────────────────────────────────
# SITE-SPECIFIC REFERENCE, NOT A SUPPORTED ENTRY POINT.
#
# This is the orchestration we actually used, kept because it records real
# operating knowledge (see the api_base/model sed guard below, which exists to
# stop a misconfigured run from silently billing a hosted API instead of using
# the GPU you just paid for). It assumes a run:ai cluster, a `csub.py`
# submission helper that does NOT ship with this repo, a shared filesystem
# mounted at the same path in every pod, and two prebuilt images.
#
# REMOTE_DIR, BIXBENCH_IMAGE and MLO_IMAGE have no defaults and must be set.
# NODE_TYPE (default h100) and POD_HOME (default REMOTE_DIR) are optional.
#
# The portable entry point is the Python script directly:
#   python scripts/fork/fork_trajectory.py --pkl ... --results ... --out-dir ...
# ─────────────────────────────────────────────────────────────────────────────
#
# One-shot orchestrator for forking benchmarked trajectories on run:ai.
#
# Submits TWO jobs:
#   1. policy   (mlo-base image, 1x H100) -> serves vLLM (${POLICY_MODEL})
#   2. sandbox  (bixbench image, no GPU)  -> runs scripts/fork/fork_trajectory.py
#
# Unlike launch_all.bash, the sandbox does NOT start the dataset server:
# fork_trajectory.py imports Dataset directly and builds the env in-process.
# It still needs the vLLM policy (live policy calls) and the rubric model
# (claude-sonnet-4-6 per server.fork.yaml, via .env API keys), and it reads a
# PRIOR run's artifacts (trajectories.pkl + <id>/score_info.json).
#
# Each trajectory is forked exactly ONCE, so a run costs one rollout per forkable
# trajectory. (An earlier revision chained forks — round N+1 forking round N's
# notebook — but that machinery has been removed; archive/ holds runs made under it.)
#
# Run this from your LOCAL machine (where `runai` is available), NOT inside a pod.
# The EXIT trap deletes both jobs on success, failure, or Ctrl-C.
#
# Usage:
#   ./fork.bash                       # fork ALL trajectories in the pkl (default)
#   ./fork.bash TRAJ_ID               # fork only TRAJ_ID, e.g. task_1_rep0
#   PKL=... RESULTS=... OUT_DIR=... ./fork.bash   # fork a different prior run (see below)
#   NUM_PARALLEL=4 ./fork.bash        # cap concurrent forks (default 8)
#
# Defaults fork archive/sonnet-judge's hypotest-wo-protocol run: 153 trajectories,
# 143 of them forkable. See the PKL / RESULTS / OUT_DIR block below.
#
set -uo pipefail

# ----------------------------------------------------------------------------- config
# ═══ WHICH RUN TO FORK — the three paths you are most likely to change ═══════
#   PKL      the trajectories to fork
#   RESULTS  the <id>/score_info.json dirs the fork points are read from
#   OUT_DIR  where the forks are written (keep the trailing slash)
PKL="${PKL:-archive/sonnet-judge/benchmark_results-hypotest-wo-protocol/trajectories.pkl}"
RESULTS="${RESULTS:-archive/sonnet-judge/results-hypotest-wo-protocol}"
OUT_DIR="${OUT_DIR:-archive/sonnet-judge/forks-driving-hypotest-wo-protocol/}"
#
# Each is independently overridable — e.g. to fork archive/trial/ (the 5-trajectory run):
#   PKL=archive/trial/benchmark_results/trajectories.pkl \
#     RESULTS=archive/trial/results OUT_DIR=archive/trial/forks/ ./fork.bash
#
# Why three paths and not one shared RUN_DIR prefix: archive/sonnet-judge/ holds one *pair*
# of directories per benchmark variant (-hypotest-wo-protocol, -hypotest-w-protocol,
# -biomni-w-protocol) rather than the plain benchmark_results/ + results/ layout that
# archive/trial/ uses, so the variant name has to appear in each path anyway — a RUN_DIR
# knob would only ever be half of a working override. Keep OUT_DIR inside the forked run's
# own directory so a run's forks stay with the run; that also keeps --skip-existing honest,
# since two runs can produce the same traj_ids and a shared output dir would make one skip
# the other's forks. And forks-driving- rather than plain forks-: forks-hypotest-wo-protocol/
# already holds 139 older single-fork runs under the same task_N_repM-fork_cellK/ naming, which
# a new run would silently overwrite (as would its fork_summary.json rollup).
#
# The wo-protocol defaults match server.fork.yaml's `include_protocol: false`, and were
# verified against these artifacts: 153 trajectories (51 tasks x 3 reps), 153 graded runs,
# every trajectory matching its score_info at ratio 1.000 with no collisions, and 143 of
# them forkable — the other 10 have no criterion with a first_wrong_step, and SkipFork
# cleanly rather than failing.
# ═════════════════════════════════════════════════════════════════════════════

: "${REMOTE_DIR:?set REMOTE_DIR to this repo path as seen from inside the pods}"

: "${BIXBENCH_IMAGE:?set BIXBENCH_IMAGE to an image built from this repo Dockerfile}"
: "${MLO_IMAGE:?set MLO_IMAGE to an image that can serve vLLM}"

POLICY_JOB="policy-fork"
SANDBOX_JOB="sandbox-fork"

VLLM_PORT=8000
# Served by the policy pod AND selected as the agent's llm_model.name in
# benchmark.fork.gen.yaml (see the sed below). Must be one of benchmark.yaml's
# model_list entries, or SimpleAgentConfig cannot resolve it.
POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen3.6-27B}"

POLICY_TIMEOUT="24h"
FORK_TIMEOUT="24h"          # one rollout per forkable trajectory

POLL_INTERVAL=15
POLICY_READY_TIMEOUT=1800   # seconds to wait for vLLM to come up (model load is slow)

# Max trajectories forked concurrently inside the sandbox (asyncio semaphore in
# fork_trajectory.py). Kept below the benchmark's num_parallel: each fork hits both
# the policy and rubric models on the single vLLM pod.
NUM_PARALLEL="${NUM_PARALLEL:-8}"

# server.fork.yaml (not server.yaml) — it fixes capsule_dir to capsules/hypotest and
# drops max_problems, both of which server.yaml gets wrong. See the header of that file.
SERVER_CONFIG="${SERVER_CONFIG:-server.fork.yaml}"

# Trajectory to fork. Empty (the default) forks EVERY trajectory in the pkl in a
# single sandbox job. Pass an id to fork just one; valid ids depend on the prior
# run — archive/trial's benchmark (num_replications=1) produced task_0 .. task_4.
TRAJ_ID="${1:-}"
if [ -n "${TRAJ_ID}" ]; then
  TRAJ_ARG="--traj-id ${TRAJ_ID}"
else
  TRAJ_ARG=""   # no --traj-id => fork_trajectory.py forks all trajectories
fi

# ----------------------------------------------------------------------------- preflight
# The sandbox mounts the same shared filesystem, so when this machine can see REMOTE_DIR
# we can catch a bad path here rather than after paying for an H100 to spin up.
if [ -d "${REMOTE_DIR}" ]; then
  [ -e "${REMOTE_DIR}/${SERVER_CONFIG}" ] || {
    echo "ERROR: missing ${REMOTE_DIR}/${SERVER_CONFIG} (set SERVER_CONFIG=...)" >&2; exit 1; }
  [ -e "${REMOTE_DIR}/${PKL}" ] || {
    echo "ERROR: missing ${REMOTE_DIR}/${PKL} (set PKL=...)" >&2
    exit 1; }
  if ! compgen -G "${REMOTE_DIR}/${RESULTS}/*/score_info.json" >/dev/null; then
    echo "ERROR: no <id>/score_info.json under ${REMOTE_DIR}/${RESULTS} (set RESULTS=...)" >&2
    echo "       The fork point is read from there." >&2
    exit 1
  fi
  echo "==> Preflight OK"
  echo "    pkl     : ${PKL}"
  echo "    results : ${RESULTS} ($(compgen -G "${REMOTE_DIR}/${RESULTS}/*/score_info.json" | wc -l) graded runs)"
  echo "    out     : ${OUT_DIR}"
fi

# ----------------------------------------------------------------------------- cleanup
cleanup() {
  echo
  kill "${LOGS_PID:-}" 2>/dev/null || true
  echo "==> Cleaning up run:ai jobs..."
  runai delete job "${SANDBOX_JOB}" 2>/dev/null || true
  runai delete job "${POLICY_JOB}"  2>/dev/null || true
}
trap cleanup EXIT

# ----------------------------------------------------------------------------- 1. policy / vLLM
VLLM_CMD="timeout -k 30s ${POLICY_TIMEOUT} env VLLM_USE_DEEP_GEMM=0 vllm serve ${POLICY_MODEL} \
  --gdn-prefill-backend triton \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --port ${VLLM_PORT} --max-num-seqs 64"

echo "==> Submitting policy (vLLM) training job..."
python3.11 csub.py \
  -n "${POLICY_JOB}" \
  --train \
  -g 1 --node-type ${NODE_TYPE:-h100} \
  -i "${MLO_IMAGE}" \
  --exp-folder "${REMOTE_DIR}" \
  --venv .venv_policy \
  -c "${VLLM_CMD}"

# ----------------------------------------------------------------------------- 2. wait for vLLM
echo "==> Waiting for policy pod to start accepting exec..."
until runai exec "${POLICY_JOB}" -- true >/dev/null 2>&1; do
  sleep "${POLL_INTERVAL}"
done

echo "==> Waiting for vLLM to finish loading (/health)..."
waited=0
until runai exec "${POLICY_JOB}" -- bash -lc "curl -sf http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do
  sleep "${POLL_INTERVAL}"
  waited=$((waited + POLL_INTERVAL))
  if [ "${waited}" -ge "${POLICY_READY_TIMEOUT}" ]; then
    echo "ERROR: vLLM did not become healthy within ${POLICY_READY_TIMEOUT}s." >&2
    echo "       Check: runai logs ${POLICY_JOB}" >&2
    exit 1
  fi
done

POLICY_IP="$(runai exec "${POLICY_JOB}" -- bash -lc 'hostname -I' 2>/dev/null | tr -d '\r' | awk '{print $1}')"
if [ -z "${POLICY_IP}" ]; then
  echo "ERROR: could not determine policy pod IP via hostname -I." >&2
  exit 1
fi
POLICY_API_BASE="http://${POLICY_IP}:${VLLM_PORT}/v1"
echo "==> vLLM is up at ${POLICY_API_BASE}"

# ----------------------------------------------------------------------------- 3. sandbox: fork trajectory (NO dataset server)
SANDBOX_CMD=$(cat <<EOF
set -euo pipefail
export HOME=${POD_HOME:-${REMOTE_DIR}}
cd ${REMOTE_DIR}
source .venv/bin/activate
unset PYTHONPATH
set -a && source .env && set +a

# Forking is single-shot: one rollout per trajectory, so there is no rounds knob.
# AGENT_MAX_STEPS — the step budget, replay included — is read from .env by the env itself.
echo "==> single fork per trajectory (AGENT_MAX_STEPS=\${AGENT_MAX_STEPS:-?} steps, replay included)"

# Two rewrites into benchmark.fork.gen.yaml, both required:
#   1. api_base -> the live vLLM pod IP (same as the benchmark flow).
#   2. llm_model.name -> POLICY_MODEL. Without this the agent uses whatever
#      benchmark.yaml selects (currently claude-sonnet-4-6), the H100 pod sits
#      idle, and the fork silently bills Anthropic instead. The 6-space indent
#      anchors it to llm_model.name and not the model_list's model_name keys.
# Written to its own filename (not benchmark.gen.yaml) so this doesn't clobber the
# shared benchmark.gen.yaml a concurrently-running launch_all.bash/benchmark_agent.py
# may still be reading from.
sed -E -e 's|(api_base: *http://)[0-9.]+(:[0-9]+/v1)|\1${POLICY_IP}\2|' \\
       -e 's|^      name: .*|      name: ${POLICY_MODEL}|' \\
       benchmark.yaml > benchmark.fork.gen.yaml
grep -qE "^      name: ${POLICY_MODEL//\//\\/}\$" benchmark.fork.gen.yaml || {
  echo "ERROR: failed to set llm_model.name to ${POLICY_MODEL} in benchmark.fork.gen.yaml" >&2
  exit 1
}
echo "==> policy model: \$(grep -E '^      name: ' benchmark.fork.gen.yaml)"

timeout -k 30s ${FORK_TIMEOUT} python scripts/fork/fork_trajectory.py \\
  ${TRAJ_ARG} \\
  --server-config ${SERVER_CONFIG} \\
  --benchmark-config benchmark.fork.gen.yaml \\
  --pkl ${PKL} \\
  --results ${RESULTS} \\
  --out-dir ${OUT_DIR} \\
  --skip-existing \\
  --num-parallel ${NUM_PARALLEL}
EOF
)

echo "==> Submitting sandbox (fork_trajectory) training job..."
python3.11 csub.py \
  -n "${SANDBOX_JOB}" \
  --train \
  -i "${BIXBENCH_IMAGE}" \
  --skip-secret-sync \
  -c "${SANDBOX_CMD}"

# ----------------------------------------------------------------------------- 4. follow to completion
echo "==> Waiting for sandbox pod to start..."
until runai exec "${SANDBOX_JOB}" -- true >/dev/null 2>&1; do
  sleep "${POLL_INTERVAL}"
done

runai logs "${SANDBOX_JOB}" -f &
LOGS_PID=$!

PATTERN="fork_trajectory.py"

echo "==> Waiting for ${PATTERN} to start..."
while runai exec "${SANDBOX_JOB}" -- true >/dev/null 2>&1 \
   && ! runai exec "${SANDBOX_JOB}" -- pgrep -f "${PATTERN}" >/dev/null 2>&1; do
  sleep "${POLL_INTERVAL}"
done

echo "==> ${PATTERN} is running; waiting for it to finish..."
while runai exec "${SANDBOX_JOB}" -- pgrep -f "${PATTERN}" >/dev/null 2>&1; do
  sleep "${POLL_INTERVAL}"
done

kill "${LOGS_PID:-}" 2>/dev/null || true
echo "==> Fork finished. Output under ${REMOTE_DIR}/${OUT_DIR}"
echo "    rollup    : ${OUT_DIR}fork_summary.json  (forked / skipped / failed)"
echo "    per fork  : ${OUT_DIR}<traj_id>-fork_cell<K>/fork_info.json"
echo "    view      : python scripts/inspect_fork.py ${OUT_DIR} --html forks.html"
echo "    resume    : re-run this script — --skip-existing keeps finished forks"
# cleanup runs on EXIT
