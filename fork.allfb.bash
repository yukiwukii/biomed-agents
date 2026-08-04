#!/usr/bin/env bash
#
# ABLATION orchestrator: fork ONCE per trajectory, injecting the feedback from EVERY
# criterion rather than only the ones driving the fork cell. Sibling of fork.bash, which
# it leaves completely untouched — run either, or both concurrently.
#
# Submits TWO jobs:
#   1. policy   (mlo-base image, 1x H100) -> serves vLLM (${POLICY_MODEL})
#   2. sandbox  (bixbench image, no GPU)  -> runs scripts/fork_trajectory_allfb.py
#
# Unlike launch_all.bash, the sandbox does NOT start the dataset server:
# fork_trajectory.py imports Dataset directly and builds the env in-process.
# It still needs the vLLM policy (live policy calls) and the rubric model
# (claude-sonnet-4-6 per server.fork.yaml, via .env API keys), and it reads a
# PRIOR run's artifacts (trajectories.pkl + <id>/score_info.json).
#
# The ONE difference from fork.bash: the injected note carries every criterion's
# feedback (mean 3.55 bullets per fork vs the baseline's 1.90, measured on the 143
# forkable runs), split into a "Start here" section (the criteria at the fork cell —
# byte-identical to what the baseline injects) and "Also address as you continue" (the
# rest). The fork CELL is unchanged, so this arm is comparable cell-for-cell.
#
# BASELINE ARM — run ./fork.bash against the same PKL/RESULTS. No archived fork run is a
# valid control: every one of them was graded with the judge-side fork anchoring
# (resume_from_step / the resume + prior-score prompt notes) that has since been removed
# from src/, so those grades are not comparable with anything produced now.
#
# Run this from your LOCAL machine (where `runai` is available), NOT inside a pod.
# The EXIT trap deletes both jobs on success, failure, or Ctrl-C.
#
# Usage:
#   ./fork.allfb.bash                 # fork ALL trajectories in the pkl (default)
#   ./fork.allfb.bash TRAJ_ID         # fork only TRAJ_ID, e.g. task_1_rep0
#   PKL=... RESULTS=... OUT_DIR=... ./fork.allfb.bash   # fork a different prior run
#   NUM_PARALLEL=4 ./fork.allfb.bash  # cap concurrent forks (default 8)
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
OUT_DIR="${OUT_DIR:-archive/sonnet-judge/forks-allfb-hypotest-wo-protocol/}"
#
# Each is independently overridable — e.g. to fork archive/trial/ (the 5-trajectory run):
#   PKL=archive/trial/benchmark_results/trajectories.pkl \
#     RESULTS=archive/trial/results OUT_DIR=archive/trial/forks-allfb/ ./fork.allfb.bash
#
# Why three paths and not one shared RUN_DIR prefix: archive/sonnet-judge/ holds one *pair*
# of directories per benchmark variant (-hypotest-wo-protocol, -hypotest-w-protocol,
# -biomni-w-protocol) rather than the plain benchmark_results/ + results/ layout that
# archive/trial/ uses, so the variant name has to appear in each path anyway — a RUN_DIR
# knob would only ever be half of a working override. Keep OUT_DIR inside the forked run's
# own directory so a run's forks stay with the run; that also keeps --skip-existing honest,
# since two runs can produce the same traj_ids and a shared output dir would make one skip
# the other's forks. And forks-allfb- rather than forks-: forks-hypotest-wo-protocol/ holds
# 139 older single forks and seq-forks-hypotest-wo-protocol/ the sequential run — both are
# historical artifacts under the old grading, and neither may be overwritten.
#
# The wo-protocol defaults match server.fork.yaml's `include_protocol: false`, and were
# verified against these artifacts: 153 trajectories (51 tasks x 3 reps), 153 graded runs,
# every trajectory matching its score_info at ratio 1.000 with no collisions, and 143 of
# them forkable — the other 10 have no criterion with a first_wrong_step, and SkipFork
# cleanly rather than failing.
# ═════════════════════════════════════════════════════════════════════════════

REMOTE_DIR="/mlbio_scratch/wangsaja/hypotest"

BIXBENCH_IMAGE="ic-registry.epfl.ch/yuki/bixbench:latest"
MLO_IMAGE="ic-registry.epfl.ch/mlo/mlo-base:uv1"

# Suffixed -allfb so this ablation arm can run alongside fork.bash without either
# EXIT trap deleting the other's jobs (run:ai job names are the only handle here).
POLICY_JOB="policy-fork-allfb"
SANDBOX_JOB="sandbox-fork-allfb"

VLLM_PORT=8000
# Served by the policy pod AND selected as the agent's llm_model.name in
# benchmark.fork.allfb.gen.yaml (see the sed below). Must be one of benchmark.yaml's
# model_list entries, or SimpleAgentConfig cannot resolve it.
POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen3.6-27B}"

POLICY_TIMEOUT="24h"
FORK_TIMEOUT="24h"          # one rollout per trajectory (single fork), so shorter than a chain

POLL_INTERVAL=15
POLICY_READY_TIMEOUT=1800   # seconds to wait for vLLM to come up (model load is slow)

# Max trajectories forked concurrently inside the sandbox (asyncio semaphore in
# fork_trajectory.py). Kept below the benchmark's num_parallel: each fork hits both
# the policy and rubric models on the single vLLM pod.
NUM_PARALLEL="${NUM_PARALLEL:-8}"

# No MAX_ROUNDS knob: forking is single-shot, one rollout per trajectory. AGENT_MAX_STEPS
# — the step budget, replay included — is read from .env by the env itself.

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
  TRAJ_ARG=""   # no --traj-id => fork_trajectory_allfb.py forks all trajectories
fi

# ----------------------------------------------------------------------------- preflight
# The sandbox mounts the same /mlbio_scratch, so when this machine can see REMOTE_DIR
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
  -g 1 --node-type h100 \
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
export HOME=/mlbio_scratch/wangsaja
cd ${REMOTE_DIR}
source .venv/bin/activate
unset PYTHONPATH
set -a && source .env && set +a

# No --max-rounds resolution: forking is single-shot. AGENT_MAX_STEPS — the step budget,
# replay included — is read from .env by the env itself.
echo "==> single fork per trajectory, all-criteria feedback (AGENT_MAX_STEPS=\${AGENT_MAX_STEPS:-?} steps)"

# Two rewrites into benchmark.fork.allfb.gen.yaml, both required:
#   1. api_base -> the live vLLM pod IP (same as the benchmark flow).
#   2. llm_model.name -> POLICY_MODEL. Without this the agent uses whatever
#      benchmark.yaml selects (currently claude-sonnet-4-6), the H100 pod sits
#      idle, and the fork silently bills Anthropic instead. The 6-space indent
#      anchors it to llm_model.name and not the model_list's model_name keys.
# Written to its OWN filename — not benchmark.gen.yaml (which a concurrent
# launch_all.bash/benchmark_agent.py may be reading) and not fork.bash's
# benchmark.fork.gen.yaml. The -allfb job names exist so this arm can run alongside
# fork.bash, but each arm brings up its own vLLM pod at its own IP: sharing one
# generated config would let whichever sed ran last point BOTH arms at one pod.
sed -E -e 's|(api_base: *http://)[0-9.]+(:[0-9]+/v1)|\1${POLICY_IP}\2|' \\
       -e 's|^      name: .*|      name: ${POLICY_MODEL}|' \\
       benchmark.yaml > benchmark.fork.allfb.gen.yaml
grep -qE "^      name: ${POLICY_MODEL//\//\\/}\$" benchmark.fork.allfb.gen.yaml || {
  echo "ERROR: failed to set llm_model.name to ${POLICY_MODEL} in benchmark.fork.allfb.gen.yaml" >&2
  exit 1
}
echo "==> policy model: \$(grep -E '^      name: ' benchmark.fork.allfb.gen.yaml)"

timeout -k 30s ${FORK_TIMEOUT} python scripts/fork_trajectory_allfb.py \\
  ${TRAJ_ARG} \\
  --server-config ${SERVER_CONFIG} \\
  --benchmark-config benchmark.fork.allfb.gen.yaml \\
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

# Must name the variant explicitly: pgrep -f treats this as a regex, and
# "fork_trajectory.py" does NOT match "fork_trajectory_allfb.py" (the `.` cannot span
# "_al"), so the baseline's pattern would wait here forever.
PATTERN="fork_trajectory_allfb.py"

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
echo "    view      : python scripts/inspect_fork.py ${OUT_DIR} --html forks-allfb.html"
echo "    resume    : re-run this script — --skip-existing keeps finished forks"
echo "    control   : ./fork.bash with the same PKL/RESULTS (no archived run is comparable)"
# cleanup runs on EXIT
