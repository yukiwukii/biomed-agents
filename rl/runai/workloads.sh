#!/usr/bin/env bash
# rl/runai/workloads.sh
# =============================================================================
# Submit the GRPO stack to Run:ai using ONLY kubectl -- no runai CLI.
#
# Why kubectl and not the Run:ai CLI: the CLI is not distributed for
# this host (rcpepfl.run.ai/cli/linux returns 502, /v1/k8s/cli/linux returns 404
# even with a valid id-token), and the macOS build obviously will not run here.
# But the account CAN create Run:ai workload CRDs directly:
#
#   kubectl -n runai-mlbio-wangsaja auth can-i create trainingworkloads.run.ai
#   -> yes                    (also interactiveworkloads, distributedworkloads)
#   kubectl ... auth can-i create pods|jobs|services
#   -> no                     (which is why bare Pods/Deployments are not an option)
#
# So everything below is `kubectl apply` of run.ai/v2alpha1 objects. Run:ai's
# controller creates the underlying Pod and, for the env workload, the Service --
# using its own permissions, which is how we get a Service without being allowed
# to create one.
#
# CRD conventions, which are unusual:
#   * every scalar field is wrapped:            image: {value: "repo/img:tag"}
#   * every map field is items-of-wrapped:      environment: {items: {K: {value: "v"}}}
#   * command/arguments are single STRINGS, not lists
#
#   ./rl/runai/workloads.sh env      # environment server + ClusterIP service
#   ./rl/runai/workloads.sh configs  # push rl/configs, rl/scripts, splits
#   ./rl/runai/workloads.sh smoke    # 9B rollouts only, 1 GPU -- measures the
#                                    # real transcript length distribution
#   ./rl/runai/workloads.sh nine     # GRPO on Qwen3.6-27B, 8 GPU (colocated)
#   ./rl/runai/workloads.sh 9b       # GRPO on Qwen3.5-9B, 6 GPU (async: 4 train + 2 vLLM)
#   ./rl/runai/workloads.sh status | logs <job> | clean
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBMIT_ENV="${REPO_ROOT}/rl/runai/submit.env"
[ -f "${SUBMIT_ENV}" ] || { echo "FATAL: ${SUBMIT_ENV} missing (cp submit.env.example submit.env)" >&2; exit 1; }
set -a; . "${SUBMIT_ENV}"; set +a

NS="${K8S_NAMESPACE:?set K8S_NAMESPACE in submit.env}"
REGISTRY="${REGISTRY:?set REGISTRY in submit.env}"
IMAGE_PREFIX="${IMAGE_PREFIX:-}"
ENV_IMAGE="${ENV_IMAGE:-${REGISTRY}/${IMAGE_PREFIX}hypotest-env:latest}"
# TRAIN_MODE=ngc pulls the official NeMo RL image and does the Gym wiring at pod
# start (rl/scripts/nemo_rl_setup.sh) -- no image build anywhere. Verified
# pullable: nvcr.io/nvidia/nemo-rl:v0.6.0 -> HTTP 200 with the ngc-secret creds.
TRAIN_MODE="${TRAIN_MODE:-ngc}"
GYM_SRC="${GYM_SRC:-/mlbio_scratch/wangsaja/bbh-third-party/Gym}"
if [ "${TRAIN_MODE}" = "ngc" ]; then
  TRAIN_IMAGE="${NGC_IMAGE:-nvcr.io/nvidia/nemo-rl:v0.6.0}"
  # `bash <script>`, not the bare path: ConfigMap volumes mount their files
  # 0644, so with_secrets.sh's `exec "$@"` on a bare path dies with
  # "Permission denied". Every script under /config must be invoked via bash.
  TRAIN_WRAP="bash /config/rl/scripts/nemo_rl_setup.sh"
else
  TRAIN_IMAGE="${TRAIN_IMAGE:-${REGISTRY}/${IMAGE_PREFIX}hypotest-grpo:latest}"
  TRAIN_WRAP=""
fi
ENV_JOB="${ENV_JOB:-hypotest-env}"
TRAIN_JOB="${TRAIN_JOB:-hypotest-grpo}"
SCRATCH_PVC="${SCRATCH_PVC:-mlbio-scratch}"
SCRATCH_MOUNT="${SCRATCH_MOUNT:-/mlbio_scratch}"
# The pod's HF cache is pod-local, NOT the PVC's hf_cache: that directory is
# mode 0770 and the trainer must WRITE it, but the pod runs root-squashed (see
# the NFS root_squash note below), so it fails with
#   PermissionError: [Errno 13] .../hf_cache/hub
#
# Deliberately NOT called HF_HOME. submit.env sets HF_HOME for host-side use and
# `set -a; . submit.env` exports it, so a `${HF_HOME:-...}` default here would
# be silently overridden by it -- which is exactly how this bug first shipped.
#
# Costs a re-download per run: 1.75 GB for the 0.8B is free, 19.3 GB for the 9B
# and 71.9 GB for the 35B are not.
#
# So: if a world-readable shared cache exists on the PVC (populate it with
# rl/scripts/fetch_model.sh), use it read-only with HF_HUB_OFFLINE=1 -- the pod
# reads and never writes, which is the only thing root_squash actually forbids.
# Otherwise fall back to the pod-local emptyDir and download every run.
HF_SHARED_CACHE="${HF_SHARED_CACHE:-/mlbio_scratch/wangsaja/hf_shared}"
if [ -d "${HF_SHARED_CACHE}/hub" ]; then
  POD_HF_HOME="${POD_HF_HOME:-${HF_SHARED_CACHE}}"
  HF_OFFLINE="1"
else
  POD_HF_HOME="${POD_HF_HOME:-/data/hf}"
  HF_OFFLINE="0"
fi
HYPOTEST_API_KEY="${HYPOTEST_API_KEY:-hypotest-secret}"
# 25. Was 10 (0.8B exhausted it), then 50 -- but a 50-turn trajectory at 65536
# tokens could not fit through get_logprobs on 2x H100 (76.73/79.18 GiB in use,
# 862 MiB free). 25 halves the trajectory. Keep in step with
# max_total_sequence_length in the GRPO config -- see the note there.
AGENT_MAX_STEPS="${AGENT_MAX_STEPS:-25}"
# ---------------------------------------------------------------------------
# THE STEP BUDGET IS ENFORCED BY THE **ENV** POD, NOT THE TRAINER.
#
#   src/hypotest/env/config.py:6      AGENT_MAX_STEPS = int(os.getenv(...))
#   interpreter_env.py:1268           max_steps: int = cfg.AGENT_MAX_STEPS
#   interpreter_env.py:1458           if self.step_count >= (self.max_steps - 1)
#
# The trainer/smoke pod reads the variable too, but it is the env server that
# actually stops the episode and injects FORCE_MSG. So setting it on only one
# of the two silently does nothing -- on 2026-08-07 a run submitted with
# SMOKE_STEPS=50 produced episodes of exactly 20 steps, because the env pod had
# been started earlier with the submit.env value of 20.
#
# This single value is therefore stamped onto ALL THREE workloads (env, train,
# smoke) so they cannot disagree. Override with SMOKE_STEPS, which is
# deliberately absent from submit.env -- see the note on cmd_smoke27.
#
# The env pod bakes it in at creation, so changing it means RECREATING the env
# workload, not just resubmitting the trainer.
# ---------------------------------------------------------------------------
EFFECTIVE_MAX_STEPS="${SMOKE_STEPS:-${AGENT_MAX_STEPS}}"
GPUS="${GPUS:-4}"
IMAGE_PULL_SECRET="${IMAGE_PULL_SECRET:-}"

# ---------------------------------------------------------------------------
# Node pool. REQUIRED, not cosmetic: this cluster is heterogeneous and the
# `default` pool has no label selector, so an unpinned workload can land on
# anything -- including the 148 V100s. V100 is sm_70 with NO bf16 support, and
# both GRPO configs set precision: bfloat16, so that placement fails.
#
#   pool       GPUs  product
#   h100         80  NVIDIA-H100-80GB-HBM3     <- default here
#   h200         72  NVIDIA-H200
#   a100        232  NVIDIA-A100-SXM4-80GB     (bf16 OK; fine for the 35B too)
#   a100-40g    124  NVIDIA-A100-SXM4-40GB     (too small for the 35B)
#   v100        148  Tesla-V100-SXM2-32GB      (no bf16 -- do not use)
#
# The 0.8B tiny run needs 1.75 GB and would fit anywhere bf16 works; it is
# pinned to the same pool as the real run so the test exercises the real
# hardware path.
# ---------------------------------------------------------------------------
NODE_POOL="${NODE_POOL:-h100}"
CPU_NODE_POOL="${CPU_NODE_POOL:-cpu}"
CLUSTER_DOMAIN="${CLUSTER_DOMAIN:-caas-prod.rcp.epfl.ch}"


# ---------------------------------------------------------------------------
# ENV_MODE=scratch  (default)  -- run the dataset server straight off the PVC,
#                                 with NO custom image to build.
#
# This works because the pieces already line up:
#   host             Ubuntu 22.04.5, glibc 2.35
#   repo Dockerfile  FROM ubuntu:22.04          <- same base, so ABI-compatible
#   .venv            py3.13.10, on the scratch PVC
#   kernel_env_conda py3.12.12,  on the scratch PVC
#   capsules         13.2 GB,    on the scratch PVC
#
# So a stock ubuntu:22.04 container that mounts /mlbio_scratch has everything.
# It skips a ~14 GB conda image build, which is the single slowest step in the
# whole setup, and it guarantees the pod runs the exact interpreter and packages
# you have been testing against on the host.
#
# ENV_MODE=image    -- use the purpose-built hypotest-env image instead. Use
#                      this once the setup is stable: it is reproducible and
#                      does not depend on the PVC's mutable contents.
# ---------------------------------------------------------------------------
ENV_MODE="${ENV_MODE:-scratch}"
HYPOTEST_ROOT="${HYPOTEST_ROOT:-/mlbio_scratch/wangsaja/hypotest}"
KERNEL_ENV_PATH="${KERNEL_ENV_PATH:-/mlbio_scratch/wangsaja/kernel_env_conda}"
# ---------------------------------------------------------------------------
# NFS root_squash: why the pod does NOT use hypotest/.venv directly.
#
# The scratch export squashes root to `nobody`, and containers run as root by
# default, so the pod only ever sees world-readable files. hypotest/.venv is
# world-readable -- but .venv/bin/python is a symlink into
# ~/.uv/cpython-3.13.10-linux-x86_64-gnu/, which is mode 0770. `nobody` cannot
# traverse it, so the pod died with
#
#     .venv/bin/python: Permission denied
#
# RL_RUNTIME is a self-contained, world-readable copy of that interpreter plus
# the venv, built once by rl/scripts/make_rl_runtime.sh. Nothing under
# ~/.uv or hypotest/.venv is modified, and no pod needs the user's identity.
# Re-run that script after changing dependencies -- the copy does not track
# .venv automatically.
# ---------------------------------------------------------------------------
RL_RUNTIME="${RL_RUNTIME:-/mlbio_scratch/wangsaja/rl-runtime}"

# ---------------------------------------------------------------------------
# CA certificates. Stock ubuntu:22.04 has NO ca-certificates package, so every
# TLS client in the env pod starts with an empty trust store. This first showed
# up as an HF download crash from the Rust Xet downloader:
#
#   RuntimeError: Xet Runtime Error: ... "No CA certificates were loaded from
#   the system"
#
# certifi is already in the venv, so point the standard env vars at its bundle
# rather than apt-installing at pod start (which would need apt egress).
# HF_HUB_DISABLE_XET additionally routes downloads through Python's requests
# stack, which uses certifi natively -- belt and braces, because the Xet
# downloader is Rust and does not reliably honour SSL_CERT_FILE.
#
# The judge's Anthropic calls need this too: no trust store, no LiteLLM.
# ---------------------------------------------------------------------------
CA_BUNDLE="${CA_BUNDLE:-$(ls -d "${RL_RUNTIME}"/venv/lib/python*/site-packages/certifi/cacert.pem 2>/dev/null | head -1)}"
[ -n "${CA_BUNDLE}" ] || echo "WARN: no certifi bundle under ${RL_RUNTIME}; TLS in the env pod will fail" >&2
if [ "${ENV_MODE}" = "scratch" ]; then
  ENV_IMAGE="${ENV_IMAGE_SCRATCH:-ubuntu:22.04}"
  ENV_PYTHON="${RL_RUNTIME}/venv/bin/python"
  ENV_SERVER="${HYPOTEST_ROOT}/src/hypotest/dataset_server.py"
else
  ENV_PYTHON="python"
  ENV_SERVER="/app/hypotest/src/hypotest/dataset_server.py"
fi

grn() { printf '\033[32m%s\033[0m\n' "$*"; }
red() { printf '\033[31m%s\033[0m\n' "$*" >&2; }
die() { red "FATAL: $*"; exit 1; }

preflight() {
  command -v kubectl >/dev/null 2>&1 || die "kubectl not found."
  kubectl get ns "${NS}" >/dev/null 2>&1 || die "namespace ${NS} unreachable -- check your kubeconfig."
  kubectl -n "${NS}" get pvc "${SCRATCH_PVC}" >/dev/null 2>&1 || {
    red "FATAL: PVC '${SCRATCH_PVC}' not found in ${NS}. Available:"
    kubectl -n "${NS}" get pvc --no-headers 2>/dev/null | awk '{printf "  %-16s %s %s\n",$1,$4,$6}' >&2
    exit 1
  }
  case "${REGISTRY}" in *CHANGE-ME*)
    # Only the GPU workloads need a built image; the env pod in scratch mode does not.
    # Nothing is built in scratch/ngc mode, so REGISTRY is unused there.
    { [ "${1:-}" = "env" ] && [ "${ENV_MODE}" = "scratch" ]; } || [ "${TRAIN_MODE}" = "ngc" ] \
      || die "REGISTRY still contains CHANGE-ME (see submit.env)." ;;
  esac
  return 0
}

# ---------------------------------------------------------------------------
# Run:ai does NOT name the Service after the workload. An InteractiveWorkload
# called `hypotest-env` with a ClusterIP port gets:
#
#   iw-hypotest-env-0-clusterip        <- iw-<name>-<port index>-clusterip
#
# Guessing `hypotest-env` cost a failed port-forward ("services not found") and
# would have silently zeroed every reward, because the agent catches the
# connection error, ends the episode and reports 0.0 (§6). So look it up.
# ---------------------------------------------------------------------------
env_service_name() {
  kubectl -n "${NS}" get svc -o name 2>/dev/null | sed 's|^service/||' | grep -- "${ENV_JOB}" | head -1
}

# Emit `imagePullSecrets: {value: "name"}` only when one is configured.
pull_secret_block() { [ -n "${IMAGE_PULL_SECRET}" ] && echo "  imagePullSecrets: {value: \"${IMAGE_PULL_SECRET}\"}"; }

# ---- secrets + configmaps --------------------------------------------------
cmd_secrets() {
  preflight
  grn ">> secret/${ENV_JOB}-secrets"
  kubectl create secret generic "${ENV_JOB}-secrets" -n "${NS}" \
    --from-literal=HYPOTEST_API_KEY="${HYPOTEST_API_KEY}" \
    --from-literal=ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
    --from-literal=OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    --from-literal=OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" \
    --from-literal=HF_TOKEN="${HF_TOKEN:-}" \
    --from-literal=WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --dry-run=client -o yaml | kubectl apply -f -
  grn ">> configmap/hypotest-server-config"
  kubectl create configmap hypotest-server-config -n "${NS}" \
    --from-file=server.yaml="${REPO_ROOT}/rl/runai/server.train.yaml" \
    --dry-run=client -o yaml | kubectl apply -f -
}

cmd_configs() {
  preflight
  [ -f "${REPO_ROOT}/rl/data/tiny.jsonl" ] || die "rl/data/tiny.jsonl missing -- run rl/data/make_splits.py first."
  for pair in "hypotest-rl-configs:rl/configs" "hypotest-rl-scripts:rl/scripts"; do
    grn ">> configmap/${pair%%:*} <- ${pair##*:}/"
    kubectl create configmap "${pair%%:*}" --from-file="${REPO_ROOT}/${pair##*:}/" -n "${NS}" \
      --dry-run=client -o yaml | kubectl apply -f -
  done
  grn ">> configmap/hypotest-rl-splits"
  # Every *.jsonl in rl/data, not a hardcoded list. The old version named
  # train/eval/tiny explicitly, so a newly added split was silently absent from
  # the pod and the run died with "FATAL: /config/rl/data/<x>.jsonl missing"
  # only after the 27B vLLM had finished loading (2026-08-07, all5.jsonl).
  # Deliberately *.jsonl and not the whole directory: rl/data also holds
  # make_splits.py and manifest.json, which the pod has no use for.
  local split_args=()
  for f in "${REPO_ROOT}"/rl/data/*.jsonl; do
    [ -e "${f}" ] || continue
    split_args+=(--from-file="$(basename "${f}")=${f}")
  done
  [ "${#split_args[@]}" -gt 0 ] || die "no rl/data/*.jsonl found — run rl/data/make_splits.py"
  grn "   splits: $(printf '%s ' "${split_args[@]}" | sed 's|--from-file=||g; s|=[^ ]*||g')"
  kubectl create configmap hypotest-rl-splits -n "${NS}" \
    "${split_args[@]}" \
    --dry-run=client -o yaml | kubectl apply -f -
}

# ---- environment workload --------------------------------------------------
# InteractiveWorkload, not TrainingWorkload: it is a long-running service the
# trainer connects to, and only Interactive exposes `ports` (whose serviceType
# makes Run:ai create the ClusterIP Service on our behalf).
cmd_env() {
  preflight env; cmd_secrets
  # 2026-08-15: TrainingWorkload, NOT InteractiveWorkload. Interactive workloads
  # hit a hard 12h PROJECT runtime limit (workload-overseer force-suspends them),
  # which killed the env mid-run and crashed training. TrainingWorkloads are not
  # under that cap and still expose a ClusterIP Service via the ports block
  # (tw-hypotest-env-0-clusterip, which env_service_name's grep still matches).
  grn ">> TrainingWorkload/${ENV_JOB} (CPU, ClusterIP :8008, no 12h cap)"
  kubectl apply -f - <<EOF
apiVersion: run.ai/v2alpha1
kind: TrainingWorkload
metadata: {name: ${ENV_JOB}, namespace: ${NS}}
spec:
  name: {value: ${ENV_JOB}}
  image: {value: "${ENV_IMAGE}"}
  imagePullPolicy: {value: Always}
$(pull_secret_block)
  nodePools: {value: "${CPU_NODE_POOL}"}
  cpu: {value: "${ENV_CPU:-8}"}
  memory: {value: "${ENV_MEM:-64G}"}
  command: {value: "bash"}
  arguments: {value: "/config/rl/scripts/with_secrets.sh ${ENV_PYTHON} ${ENV_SERVER} /config/server.yaml"}
  ports:
    items:
      http: {value: {container: 8008, serviceType: ClusterIP}}
  pvcs:
    items:
      scratch: {value: {claimName: ${SCRATCH_PVC}, existingPvc: true, path: ${SCRATCH_MOUNT}}}
  # work_dir stays OFF the NFS PVC: dataset_server.py:128 copytree's a full
  # capsule per rollout, so it is the hottest path in the system.
  emptyDirVolumes:
    items:
      work: {value: {path: /data/work}}
  configMapVolumes:
    items:
      cfg:     {value: {name: hypotest-server-config, mountPath: /config}}
      scripts: {value: {name: hypotest-rl-scripts, mountPath: /config/rl/scripts}}
  secretVolumes:
    items:
      secrets: {value: {name: ${ENV_JOB}-secrets, mountPath: /secrets}}
  environment:
    items:
      KERNEL_ENV_PATH: {value: "${KERNEL_ENV_PATH}"}
      AGENT_MAX_STEPS: {value: "${EFFECTIVE_MAX_STEPS}"}
      # Stripped here, in the ENV pod, because this is what builds the tool result
      # (src/hypotest/env/config.py STRIP_IMAGES). Setting it on the trainer does
      # nothing -- same trap as AGENT_MAX_STEPS above. Changing it requires
      # recreating this pod.
      STRIP_IMAGES: {value: "${STRIP_IMAGES:-true}"}
      # Caps cell OUTPUT (head+tail truncation in src/hypotest/env/config.py /
      # notebook_utils.py). 2026-08-19: 3000 -> 1500. At AGENT_MAX_STEPS=30 the
      # 9B async run OOMs in loss.backward on the tail episodes (exp_011-013 all
      # died at step 1, trainer GPUs pegged 78.8/79.2 GiB). Tool results are the
      # dominant term in episode length (~52% at 20 steps), so NB_OUTPUT_LIMIT is
      # THE source-side lever -- halving it (char cap) roughly halves the
      # tool-result tokens and pulls the projected 30-step mean/hottest of
      # ~50.5k/59.7k tokens down to ~37k/44k, well under the 65536 cap and inside
      # the backward's fitting range. Paired with HYPOTEST_DROP_LONG_TOKENS=60000
      # in cmd_9b as the neutral backstop for any episode that still balloons.
      # (The earlier 1500 was a 27B band-aid; this reinstates it for a DIFFERENT
      # reason -- the async+reference-KL additions of 2026-08-18 ate the 9B's
      # headroom.) Enforced in the ENV pod, so changing it requires restarting the
      # env workload (via 'workloads.sh env') -- the trainer alone will not pick
      # it up.
      # NOTE: this whole heredoc is unquoted, so backticks here would run as
      # commands -- keep comments backtick-free.
      NB_OUTPUT_LIMIT: {value: "${NB_OUTPUT_LIMIT:-1500}"}
      # Reap orphaned environments. Retries abandon an env without closing it
      # (run 12: 12 /start vs 9 /close), and each orphan holds a live Jupyter
      # kernel plus a ~1 GB capsule copy forever -- the env pod hit ~275 GiB
      # against a 64 GB request and was evicted. See dataset_server.sweep_idle_envs.
      #
      # 1800s is deliberately well above cell_execution_timeout (600s): a single
      # legitimate long-running cell leaves its env looking idle, and sweeping a
      # LIVE env kills a running episode. Set ENV_SWEEP_IDLE_SECONDS=0 to disable.
      ENV_SWEEP_IDLE_SECONDS:   {value: "${ENV_SWEEP_IDLE_SECONDS:-1800}"}
      ENV_SWEEP_PERIOD_SECONDS: {value: "${ENV_SWEEP_PERIOD_SECONDS:-300}"}
      # Without this, plain print() from dataset_server.py is block-buffered and
      # never reaches kubectl logs -- even "Starting dataset server" was
      # invisible, which hid every diagnostic this server emits. The trainer pod
      # already sets it (nemo_rl_setup.sh); the env pod never did.
      PYTHONUNBUFFERED: {value: "1"}
      DEPLOYMENT_PROFILE: {value: "standard"}
      USE_DOCKER: {value: "false"}
      # stock ubuntu:22.04 ships NO ca-certificates -- see CA_BUNDLE above.
      SSL_CERT_FILE: {value: "${CA_BUNDLE}"}
      REQUESTS_CA_BUNDLE: {value: "${CA_BUNDLE}"}
      CURL_CA_BUNDLE: {value: "${CA_BUNDLE}"}
      HF_HUB_DISABLE_XET: {value: "1"}
EOF
  grn ""
  grn "Wait for Running, then verify:"
  grn "  kubectl -n ${NS} get pods -w"
  grn "  kubectl -n ${NS} port-forward svc/${ENV_JOB} 8008:8008 &"
  grn "  curl -H 'X-API-Key: ${HYPOTEST_API_KEY}' http://localhost:8008/info"
}

# ---- GPU workloads ---------------------------------------------------------
submit_training() {
  local name="$1" gpus="$2" script="$3" extra_env="$4"
  preflight; cmd_secrets
  local env_svc
  env_svc="$(env_service_name)"
  [ -n "${env_svc}" ] || die "no Service matching '${ENV_JOB}' -- start the env workload first ($0 env) and wait for it to be Running. Submitting now would give every rollout reward 0.0."
  # NOT cluster.local -- this cluster's DNS domain is caas-prod.rcp.epfl.ch
  # (from a pod's /etc/resolv.conf search list). Verified resolvable from the
  # env pod; `cluster.local` gives "Name or service not known".
  local server_url="http://${env_svc}.${NS}.svc.${CLUSTER_DOMAIN}:8008"
  grn "   env server: ${server_url}"
  grn ">> TrainingWorkload/${name} (${gpus} GPU)"
  kubectl apply -f - <<EOF
apiVersion: run.ai/v2alpha1
kind: TrainingWorkload
metadata: {name: ${name}, namespace: ${NS}}
spec:
  name: {value: ${name}}
  image: {value: "${TRAIN_IMAGE}"}
  imagePullPolicy: {value: Always}
$(pull_secret_block)
  gpu: {value: "${gpus}"}
  nodePools: {value: "${NODE_POOL}"}
  cpu: {value: "${TRAIN_CPU:-32}"}
  memory: {value: "${TRAIN_MEM:-384G}"}
  # Ray's object store lives in /dev/shm, which defaults to 64Mi in Kubernetes
  # and would be exhausted immediately.
  largeShm: {value: true}
  backoffLimit: {value: 0}
  command: {value: "bash"}
  arguments: {value: "/config/rl/scripts/with_secrets.sh ${TRAIN_WRAP} bash ${script}"}
  pvcs:
    items:
      scratch: {value: {claimName: ${SCRATCH_PVC}, existingPvc: true, path: ${SCRATCH_MOUNT}}}
  # HF_HOME lives here, not on the PVC -- the pod is root-squashed and hf_cache
  # is 0770, so it cannot write there. See the HF_HOME note at the top.
  emptyDirVolumes:
    items:
      hf: {value: {path: /data/hf}}
  configMapVolumes:
    items:
      cfgs:    {value: {name: hypotest-rl-configs, mountPath: /config/rl/configs}}
      splits:  {value: {name: hypotest-rl-splits,  mountPath: /config/rl/data}}
      scripts: {value: {name: hypotest-rl-scripts, mountPath: /config/rl/scripts}}
  secretVolumes:
    items:
      secrets: {value: {name: ${ENV_JOB}-secrets, mountPath: /secrets}}
  environment:
    items:
      HYPOTEST_SERVER_URL: {value: "${server_url}"}
      HYPOTEST_GYM_CONFIG: {value: "/config/rl/configs/gym_hypotest_remote.yaml"}
      HYPOTEST_CONFIG_DIR: {value: "/config/rl/configs"}
      NEMO_GYM_ROOT:       {value: "/opt/Gym"}
      GYM_SRC:             {value: "${GYM_SRC}"}
      HF_HOME:             {value: "${POD_HF_HOME}"}
      # 1 when HF_HOME is the shared read-only PVC cache: forces local_files_only
      # so nothing ever tries to write into it. A missing model then fails loudly
      # instead of silently re-downloading.
      HF_HUB_OFFLINE:      {value: "${HF_OFFLINE}"}
      HF_HUB_DISABLE_XET:  {value: "1"}
      AGENT_MAX_STEPS:     {value: "${EFFECTIVE_MAX_STEPS}"}
      # The Gym servers default to ONE FastAPI worker
      # (server_utils.py:473, getenv(NEMO_GYM_FASTAPI_NUM_WORKERS, "1")).
      # NeMo RL issues all rollouts concurrently -- run_examples() is called with
      # no semaphore, so nullcontext, then tqdm.as_completed -- but they then
      # queue behind that single worker. Measured: total rollout time was exactly
      # N x per-rollout in every run (4 x 146.85s = 9:47, 4 x 327.86s = 21:51),
      # i.e. perfectly serialised. Real concurrency should make the total
      # approach ONE rollout, not N.
      NEMO_GYM_FASTAPI_NUM_WORKERS: {value: "8"}
      # DO NOT set PYTORCH_ALLOC_CONF=expandable_segments:True here. It is the
      # obvious fix for the 27B optimizer-step OOM (7.65 GiB reserved but
      # unallocated) and the torch error message even suggests it -- but it
      # BREAKS the weight refit into vLLM:
      #
      #   Error in VllmInternalWorkerExtension.update_weights_via_ipc_zmq:
      #   pidfd_getfd: Operation not permitted
      #
      # expandable_segments allocates through CUDA's virtual-memory API, and
      # sharing those allocations with the vLLM process over IPC needs
      # pidfd_getfd(2) -> CAP_SYS_PTRACE, which this pod lacks. It fails at the
      # first weight sync, before any rollout. Isolated across three runs:
      #   no expandable_segments, cpu_offload false -> 0 pidfd, rollouts ran
      #   expandable_segments,    cpu_offload true  -> 2 pidfd, no rollouts
      #   expandable_segments,    cpu_offload false -> 2 pidfd, no rollouts
${extra_env}
EOF
  grn ""
  grn "Follow:  $0 logs ${name}"
}

# Qwen3.5-9B rollouts ONLY -- no trainer, no optimizer. 1 GPU instead of 5.
#
# Purpose: measure how long our transcripts actually are, which no run has ever
# done. Every length we have is an accident (an OOM allocation size, or a vLLM
# 400), so the cap has been sized from two samples and a benchmark that measures
# a different thing. analyze_rollouts.py reports p50/p95/p99/max with the real
# tokenizer and the overflow fraction against a proposed cap.
#
#   SMOKE_MAX_SEQ_LEN 98304, deliberately far above anything observed, so
#   nothing is rejected and we see the true distribution rather than a censored
#   one. At TP=1 that is ~6 GB of KV for one sequence against a ~16 GB budget.
#
#   Sampling defaults to the TRAINING regime (top_p 1.0, top_k null). Re-run
#   with SMOKE_TOP_P=0.95 SMOKE_TOP_K=20 to A/B against the inference regime
#   the standalone benchmark endpoints use -- that pair of runs is what
#   distinguishes "sampling makes our episodes ramble" from "the agent harness
#   differs".
#
# Trajectories land on the PVC (rl/results is mode 1777, so the squashed pod can
# write there). The script's own default would resolve under /config, which is a
# read-only ConfigMap mount, so SMOKE_OUT must be set explicitly.
#
# STEP BUDGET -- use SMOKE_STEPS, not AGENT_MAX_STEPS.
# This file does `set -a; . submit.env; set +a` at line 40, which EXPORTS
# submit.env's AGENT_MAX_STEPS over anything you put on the command line. So
#   AGENT_MAX_STEPS=50 ./workloads.sh smoke      <- silently runs 20
#   SMOKE_STEPS=50     ./workloads.sh smoke      <- works
# SMOKE_STEPS is deliberately absent from submit.env for that reason. This is
# the same shadowing that made a whole session of "50 steps" changes no-ops
# (docs/rl_context.md 0.8). Verify what actually reached the pod with:
#   kubectl -n $NS get trainingworkload.run.ai hypotest-grpo-smoke \
#     -o jsonpath='{.spec.environment.items}' | tr ',' '\n' | grep AGENT_MAX
#
# Defaults to the training budget (20). Set SMOKE_STEPS=50 to match the
# standalone benchmark's budget when comparing trajectory lengths against it.
cmd_smoke() {
  submit_training "${TRAIN_JOB}-smoke" 1 /config/rl/scripts/00_smoke_rollout.sh \
"      SMOKE_INPUT:        {value: \"${SMOKE_INPUT_FILE:-/config/rl/data/all5.jsonl}\"}
      SMOKE_OUT:          {value: \"${SAVE_DIR}/smoke/rollouts.jsonl\"}
      SMOKE_LIMIT:        {value: \"${SMOKE_LIMIT:-5}\"}
      SMOKE_REPEATS:      {value: \"${SMOKE_REPEATS:-1}\"}
      SMOKE_PARALLEL:     {value: \"${SMOKE_PARALLEL:-2}\"}
      SMOKE_MAX_SEQ_LEN:  {value: \"${SMOKE_MAX_SEQ_LEN:-131072}\"}
      SMOKE_TP:           {value: \"1\"}
      AGENT_MAX_STEPS:    {value: \"${EFFECTIVE_MAX_STEPS}\"}
      SMOKE_TOP_P:        {value: \"${SMOKE_TOP_P:-1.0}\"}
      SMOKE_TOP_K:        {value: \"${SMOKE_TOP_K:-null}\"}
      SMOKE_MAX_TOKENS:   {value: \"${SMOKE_MAX_TOKENS:-4090}\"}
      BBH_POLICY_MODEL:   {value: \"Qwen/Qwen3.5-9B\"}"
}

# Qwen3.6-27B GRPO. The only training workload. (Job name still says -9b; the
# model moved 2026-08-12 and the name is wired into the ckpt/log paths.)
#
# 8 GPUs, COLOCATED -- vLLM and the trainer SHARE all 8 rather than splitting
# them. Matches `cluster.gpus_per_node: 8` and
# `generation.colocated.enabled: true` in grpo_hypotest_27b.yaml. These numbers
# MUST agree: submit fewer than the config asks for and the job wedges in
# Pending with no useful error.
#
# WHY 8 AND WHY COLOCATED. The DTensor mesh derives dp = world/(tp*cp*ep), and
# tensor_parallel_size must DIVIDE the trainer rank count. TP=4 is forced (TP=3
# does not divide the 248320 vocab; TP=1 routes logprobs into the unchunked fp32
# full-vocab path that has OOM'd every run; TP=8 would split a 256-dim KV head
# across ranks, since num_key_value_heads is 4). The 27B needs 23.56 GB/GPU of
# weights at TP=4 -- ~15 GB/GPU more than the 9B, against ~9.7 GiB of headroom --
# so it needs dp=2, i.e. EIGHT trainer ranks. Non-colocated 8 would give
# 4 vLLM + 4 trainer, leaving dp=1 and the same OOM. Colocation is what puts all
# 8 ranks on the trainer. See the header of grpo_hypotest_27b.yaml.
#
# TRAJECTORY CAPTURE. Both paths below must point at the PVC, because the pod's
# own filesystem dies with it -- that is why the transcripts from runs 5-9 are
# gone. `logs/grpo-hypotest-9b` in the config is RELATIVE, and 01_grpo.sh cds to
# /opt/nemo-rl before exec, so it silently resolved inside the container.
#
# Do NOT pre-create these directories from the host. rl/results is 1777 so the
# root-squashed pod makes its own; a host-created, wangsaja-owned directory is
# unwritable by the pod.
cmd_nine() {
  submit_training "${TRAIN_JOB}-27b" "${NINE_GPUS:-8}" /config/rl/scripts/01_grpo.sh \
"      HYPOTEST_GRPO_CONFIG: {value: \"/config/rl/configs/grpo_hypotest_27b.yaml\"}
      HYPOTEST_TRAJ_DUMP:   {value: \"${SAVE_DIR}/traj/${TRAJ_TAG:-$(date +%Y%m%d-%H%M%S)}\"}
      HYPOTEST_LOG_DIR:     {value: \"${SAVE_DIR}/logs/grpo-hypotest-27b\"}
      HYPOTEST_BF16_LOGITS: {value: \"${HYPOTEST_BF16_LOGITS:-1}\"}
      # 2026-08-14: memory backstop. Episodes whose trained length exceeds this
      # are excluded from the gradient (message_log truncated to a stub +
      # loss_multiplier 0) instead of OOMing the forward -- see drop_long_patch.py
      # / HYPOTEST_DROP_LONG_PATCH. 37000 sits just below the 37.4k that trained
      # fine on 80 GB (48.5k OOM'd). Reward still counts in the GRPO baseline;
      # only the backward is skipped. Pairs with NB_OUTPUT_LIMIT on the env pod.
      HYPOTEST_DROP_LONG_TOKENS: {value: \"${HYPOTEST_DROP_LONG_TOKENS:-37000}\"}
      # Checkpoints MUST land on the PVC. The config default is relative and
      # 01_grpo.sh cds to /opt/nemo-rl, so unset this and checkpoints die with
      # the pod -- worthless against the preemption they exist to survive.
      #
      # 2026-08-12: -9b -> -27b, AND the old tree was physically moved aside:
      #   rl/results/ckpt  ->  rl/results/ckpt-9b-retired-20260812
      # (437 MB, step_9 + step_12, both intact and still readable as the
      # provenance for rl/models/qwen3.5-9b-grpo-preflight12/.)
      #
      # Those are Qwen3.5-9B r=64 32-module adapters and CANNOT be loaded by the
      # 27B r=32 64-module config. grpo.py:269 get_latest_checkpoint_path()
      # auto-resumes from the newest dir with NO config check, so either alone
      # would have prevented the shape-mismatch crash; both are in place.
      #
      # NOTE ON HOW the move was possible: the pod runs root-squashed, so ckpt/
      # and grpo-hypotest-9b/ are owned by uid nobody, mode 755 -- the host user
      # has
      # (NB: no backticks anywhere in this block. It sits inside a DOUBLE-QUOTED
      # bash string, so backticks are command substitution and would execute.)
      # no write bit on either, and step_9/step_12 therefore CANNOT be renamed
      # individually. What worked was renaming ckpt/ itself, whose parent
      # rl/results IS host-owned (1777). Do NOT chmod anything to get around
      # this. Do NOT pre-create the new directory either -- rl/results is 1777 so
      # the pod makes its own; a host-created, wangsaja-owned directory is
      # unwritable BY the pod.
      HYPOTEST_CKPT_DIR:    {value: \"${SAVE_DIR}/ckpt/grpo-hypotest-27b\"}
      # Distinct W&B run per submission. RUN_TAG labels what this run is for
      # (e.g. RUN_TAG=preflight, RUN_TAG=fullepoch); without it every run
      # reported under one static name and could not be told apart.
      WANDB_RUN_NAME:       {value: \"${RUN_TAG:-run}-$(date +%Y%m%d-%H%M%S)\"}"
}

# 6 GPUs, NON-COLOCATED (4 trainer + 2 vLLM), ASYNC GRPO. Qwen3.5-9B r64/a128.
# Everything model-specific is in grpo_hypotest_9b.yaml; this only sets the GPU
# count and the per-run env. Separate ckpt/log dirs from -27b so auto-resume
# (get_latest_checkpoint_path, no config check) never crosses model sizes.
cmd_9b() {
  submit_training "${TRAIN_JOB}-9b" "${GPUS_9B:-6}" /config/rl/scripts/01_grpo.sh \
"      HYPOTEST_GRPO_CONFIG: {value: \"/config/rl/configs/grpo_hypotest_9b.yaml\"}
      HYPOTEST_TRAJ_DUMP:   {value: \"${SAVE_DIR}/traj/${TRAJ_TAG:-$(date +%Y%m%d-%H%M%S)}\"}
      HYPOTEST_LOG_DIR:     {value: \"${SAVE_DIR}/logs/grpo-hypotest-9b\"}
      HYPOTEST_BF16_LOGITS: {value: \"${HYPOTEST_BF16_LOGITS:-1}\"}
      # 2026-08-19: 70000 -> 60000. The old 70000 was ABOVE the 65536 seq cap, so
      # the drop condition (trained_length > threshold) could never fire and the
      # backstop was effectively disabled -- which is why exp_011-013 CRASHED in
      # loss.backward (backoffLimit 0, no retry) instead of skipping the one long
      # episode. 60000 sits below the 65536 ceiling so it can actually fire: any
      # episode over 60k tokens is excluded from the GRADIENT (message_log
      # truncated to a stub + loss_multiplier 0) while its REWARD still counts in
      # the GRPO baseline -- a NEUTRAL exclusion, unlike lowering the seq cap,
      # which makes vLLM REJECT the episode and return a silent reward 0.0.
      # In normal use it should rarely fire: with NB_OUTPUT_LIMIT=1500 the
      # projected 30-step hottest episode is ~44k, well under 60k. It is the
      # safety net for a hotter-than-projected tail (the 9B's verbosity is
      # unmeasured). NOTE: this is the crash-preventer, not the fit -- if step 1
      # still OOMs, an episode is OOMing BELOW 60k and this threshold must come
      # down (and/or NB_OUTPUT_LIMIT down further).
      HYPOTEST_DROP_LONG_TOKENS: {value: \"${HYPOTEST_DROP_LONG_TOKENS:-60000}\"}
      # SEPARATE from grpo-hypotest-27b (matches the config default). A 27B r=32
      # adapter cannot load into this 9B r=64 config; distinct dirs keep
      # auto-resume from ever crossing them.
      HYPOTEST_CKPT_DIR:    {value: \"${SAVE_DIR}/ckpt/grpo-hypotest-9b\"}
      WANDB_RUN_NAME:       {value: \"${RUN_TAG:-9b-async}-$(date +%Y%m%d-%H%M%S)\"}"
}

cmd_status() {
  preflight
  echo "=== workloads ==="
  kubectl -n "${NS}" get interactiveworkloads,trainingworkloads 2>/dev/null || true
  echo; echo "=== pods ==="
  kubectl -n "${NS}" get pods -o wide 2>/dev/null || true
  echo; echo "=== services ==="
  kubectl -n "${NS}" get svc 2>/dev/null || true
}

cmd_logs() {
  preflight
  local job="${1:-${TRAIN_JOB}-tiny}"
  local pod
  pod=$(kubectl -n "${NS}" get pods -o name 2>/dev/null | grep -- "${job}" | head -1)
  [ -n "${pod}" ] || die "no pod matching '${job}'. Try: $0 status"
  kubectl -n "${NS}" logs -f "${pod}"
}

cmd_clean() {
  preflight
  # The env is a TrainingWorkload since the 12h-cap fix (2026-08-15); delete that.
  # Keep the interactiveworkload delete too, to reap any legacy/stale interactive
  # env. Both --ignore-not-found. Missing the trainingworkload here left a stale
  # env holding its immutable env-vars -- a later `env` then fails to apply with
  # "field is immutable" (e.g. NB_OUTPUT_LIMIT). Deleting is the only way to
  # change an env var on a RunAI workload.
  kubectl -n "${NS}" delete trainingworkload "${ENV_JOB}" --ignore-not-found
  kubectl -n "${NS}" delete interactiveworkload "${ENV_JOB}" --ignore-not-found
  # Sweep by prefix, not a fixed list. The old list was
  #   "${TRAIN_JOB}" "${TRAIN_JOB}-tiny" "${TRAIN_JOB}-smoke"
  # which silently missed -9b and -27b: on 2026-08-06 a Failed 5-GPU
  # hypotest-grpo-27b workload survived `clean` and kept holding its slot until
  # it was deleted by hand. GPUs are shared -- this must not depend on someone
  # remembering to extend a list when a new variant is added.
  local tws
  tws="$(kubectl -n "${NS}" get trainingworkload -o name 2>/dev/null \
         | grep -E "/${TRAIN_JOB}(-[a-z0-9_]+)?$" || true)"
  if [ -n "${tws}" ]; then
    # shellcheck disable=SC2086
    kubectl -n "${NS}" delete ${tws} --ignore-not-found
  fi
  kubectl -n "${NS}" delete configmap hypotest-server-config hypotest-rl-configs \
    hypotest-rl-splits hypotest-rl-scripts --ignore-not-found
  kubectl -n "${NS}" delete secret "${ENV_JOB}-secrets" hypotest-env-secrets \
    hypotest-train-secrets --ignore-not-found
}

case "${1:-}" in
  env) cmd_env ;; configs) cmd_configs ;; secrets) cmd_secrets ;;
  smoke) cmd_smoke ;; nine) cmd_nine ;; 9b) cmd_9b ;;
  status) cmd_status ;; logs) shift; cmd_logs "${@}" ;; clean) cmd_clean ;;
  *) sed -n '30,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
