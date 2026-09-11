#!/usr/bin/env bash
# rl/runai/build.sh
# =============================================================================
# Build the container images INSIDE the cluster with kaniko, submitted as Run:ai
# TrainingWorkloads. No Docker daemon anywhere.
#
#   ./rl/runai/build.sh secret     # one-time: registry credentials
#   ./rl/runai/build.sh nemo-rl    # base image (SLOW: ~1-3h, tens of GB)
#   ./rl/runai/build.sh env        # hypotest-env (optional -- see note below)
#
# There is no `train` target. TRAIN_MODE=ngc pulls the stock NeMo RL image and
# does the Gym wiring at pod start (rl/scripts/nemo_rl_setup.sh), so the trainer
# image never has to be built.
#   ./rl/runai/build.sh status | logs <job> | clean
#
# WHY KANIKO
# ----------
# There is no Docker on the host, and rootless podman is impossible here:
# /etc/subuid has no entry for this user, which needs a sysadmin. But the
# account can create TrainingWorkloads, and kaniko builds images from inside a
# container with no daemon and no privileges. Build context and Dockerfiles come
# off the scratch PVC, which the cluster already mounts.
#
# The NVIDIA base is publicly pullable, verified anonymously:
#   nvcr.io/nvidia/cuda-dl-base:25.05-cuda12.9-devel-ubuntu24.04 -> HTTP 200
# so no NGC account is required.
#
# NOTE ON `env`: with ENV_MODE=scratch (the default in workloads.sh) you do not
# need the hypotest-env image at all -- the env pod runs stock ubuntu:22.04 and
# uses the .venv and kernel_env already on the PVC. Build it only when you want
# a reproducible image that does not depend on mutable PVC contents.
#
# KANIKO vs BUILDKIT
# ------------------
# kaniko does not implement `RUN --mount=type=cache` (or =ssh). All four
# Dockerfiles here use them, so this script generates stripped copies at submit
# time into ${BUILD_DIR}. Dropping a cache mount only makes the build slower --
# it never changes the resulting image.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBMIT_ENV="${REPO_ROOT}/rl/runai/submit.env"
[ -f "${SUBMIT_ENV}" ] || { echo "FATAL: ${SUBMIT_ENV} missing." >&2; exit 1; }
set -a; . "${SUBMIT_ENV}"; set +a

NS="${K8S_NAMESPACE:?}"
REGISTRY="${REGISTRY:?}"
SCRATCH_PVC="${SCRATCH_PVC:-mlbio-scratch}"
SCRATCH_MOUNT="${SCRATCH_MOUNT:-/mlbio_scratch}"
IMAGE_PREFIX="${IMAGE_PREFIX:-}"

# Build context: the PARENT of both this repo and bbh-third-party, because
# Dockerfile.train needs Gym from one and rl/configs from the other.
CTX_ROOT="${CTX_ROOT:-/mlbio_scratch/wangsaja}"
REPO_SUBDIR="${REPO_SUBDIR:-hypotest}"
THIRD_PARTY_SUBDIR="${THIRD_PARTY_SUBDIR:-bbh-third-party}"
BUILD_DIR="${BUILD_DIR:-${CTX_ROOT}/${REPO_SUBDIR}/rl/.kaniko}"

KANIKO_IMAGE="${KANIKO_IMAGE:-gcr.io/kaniko-project/executor:v1.23.2}"
KANIKO_CPU="${KANIKO_CPU:-16}"
KANIKO_MEM="${KANIKO_MEM:-64G}"
DOCKER_SECRET="${DOCKER_SECRET:-kaniko-docker-config}"

grn() { printf '\033[32m%s\033[0m\n' "$*"; }
red() { printf '\033[31m%s\033[0m\n' "$*" >&2; }
die() { red "FATAL: $*"; exit 1; }

preflight() {
  command -v kubectl >/dev/null 2>&1 || die "kubectl not found."
  kubectl get ns "${NS}" >/dev/null 2>&1 || die "namespace ${NS} unreachable."
  case "${REGISTRY}" in *CHANGE-ME*) die "REGISTRY still contains CHANGE-ME." ;; esac
  kubectl -n "${NS}" get secret "${DOCKER_SECRET}" >/dev/null 2>&1 \
    || die "secret/${DOCKER_SECRET} missing. Run: $0 secret"
}

# kaniko reads /kaniko/.docker/config.json. A kubernetes.io/dockerconfigjson
# secret projects its key as `.dockerconfigjson`, which is the wrong filename and
# the Run:ai CRD's secretVolumes has no key-remapping. So this is a GENERIC
# secret whose key is literally `config.json`.
cmd_secret() {
  local src="${DOCKER_CONFIG_JSON:-$HOME/.docker/config.json}"
  [ -f "${src}" ] || die "no docker config at ${src}.
On your Mac:  docker login ic-registry.epfl.ch   (then re-run with
DOCKER_CONFIG_JSON=/path/to/config.json, or copy the file to this host)."
  if grep -q '"credsStore"' "${src}" && ! grep -q '"auth"' "${src}"; then
    red "WARNING: ${src} uses a credential helper (credsStore) and holds no auth blob."
    red "kaniko cannot use a helper. Generate a plain config.json from a Harbor CLI"
    red "secret instead (Harbor UI -> your profile -> CLI secret):"
    red ""
    red "  U=<gaspar-user>; T=<harbor-cli-secret>"
    red "  printf '{\"auths\":{\"ic-registry.epfl.ch\":{\"auth\":\"%s\"}}}' \\"
    red "    \"\$(printf '%s:%s' \"\$U\" \"\$T\" | base64 -w0)\" > /tmp/kaniko-config.json"
    red "  DOCKER_CONFIG_JSON=/tmp/kaniko-config.json $0 secret"
    exit 1
  fi
  grn ">> secret/${DOCKER_SECRET} from ${src}"
  kubectl create secret generic "${DOCKER_SECRET}" -n "${NS}" \
    --from-file=config.json="${src}" --dry-run=client -o yaml | kubectl apply -f -
}

# Strip BuildKit-only syntax kaniko cannot parse.
kanikoize() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "${dst}")"
  sed -E 's/--mount=type=(cache|ssh)[^ ]*//g; s/[[:space:]]+\\$/ \\/' "${src}" > "${dst}"
  local n; n=$(grep -c 'mount=type=' "${dst}" || true)
  [ "${n}" -eq 0 ] || die "still ${n} unsupported mount(s) in ${dst}"
  grn "   kanikoized $(basename "${src}") -> ${dst#${CTX_ROOT}/}"
}

submit_build() {
  local name="$1" dockerfile="$2" context="$3" dest="$4" buildargs="$5"
  preflight
  grn ">> TrainingWorkload/${name}"
  grn "   dockerfile: ${dockerfile}"
  grn "   context:    ${context}"
  grn "   push to:    ${dest}"
  local args="--dockerfile=${dockerfile} --context=dir://${context} --destination=${dest} --verbosity=info --single-snapshot ${buildargs}"
  kubectl apply -f - <<EOF
apiVersion: run.ai/v2alpha1
kind: TrainingWorkload
metadata: {name: ${name}, namespace: ${NS}}
spec:
  name: {value: ${name}}
  image: {value: "${KANIKO_IMAGE}"}
  cpu: {value: "${KANIKO_CPU}"}
  memory: {value: "${KANIKO_MEM}"}
  backoffLimit: {value: 0}
  arguments: {value: "${args}"}
  pvcs:
    items:
      scratch: {value: {claimName: ${SCRATCH_PVC}, existingPvc: true, path: ${SCRATCH_MOUNT}}}
  secretVolumes:
    items:
      docker: {value: {name: ${DOCKER_SECRET}, mountPath: /kaniko/.docker}}
EOF
  grn ""
  grn "Follow:  $0 logs ${name}"
}

cmd_nemo_rl() {
  local rl="${CTX_ROOT}/${THIRD_PARTY_SUBDIR}/RL"
  [ -d "${rl}" ] || die "${rl} not found."
  kanikoize "${rl}/docker/Dockerfile" "${BUILD_DIR}/Dockerfile.nemo-rl"
  red ">> NOTE: this build runs six 'uv sync' passes (torch, vLLM, sglang,"
  red "   megatron-core, automodel) plus a Transformer Engine compile. Expect"
  red "   1-3 hours and tens of GB. It only has to happen once."
  submit_build "build-nemo-rl" "${BUILD_DIR}/Dockerfile.nemo-rl" "${rl}" \
    "${REGISTRY}/${IMAGE_PREFIX}nemo-rl:v0.6.0" "--target=release"
}

cmd_env() {
  local repo="${CTX_ROOT}/${REPO_SUBDIR}"
  kanikoize "${repo}/Dockerfile" "${BUILD_DIR}/Dockerfile.interpreter"
  kanikoize "${repo}/rl/docker/Dockerfile.env" "${BUILD_DIR}/Dockerfile.env"
  red ">> Two sequential builds: interpreter-env (large conda stack), then"
  red "   hypotest-env on top. Only needed for ENV_MODE=image."
  submit_build "build-interpreter-env" "${BUILD_DIR}/Dockerfile.interpreter" "${repo}" \
    "${REGISTRY}/${IMAGE_PREFIX}interpreter-env:latest" ""
  grn ""
  grn "When that finishes, run:  $0 env-stage2"
}

cmd_env_stage2() {
  submit_build "build-hypotest-env" "${BUILD_DIR}/Dockerfile.env" "${CTX_ROOT}/${REPO_SUBDIR}" \
    "${REGISTRY}/${IMAGE_PREFIX}hypotest-env:latest" "--build-arg=BASE_IMAGE=${REGISTRY}/${IMAGE_PREFIX}interpreter-env:latest"
}

cmd_status() { kubectl -n "${NS}" get trainingworkloads,pods 2>/dev/null | grep -E 'build-|NAME' || echo "no build workloads"; }
cmd_logs()   { local p; p=$(kubectl -n "${NS}" get pods -o name | grep -- "${1:?job name}" | head -1); [ -n "$p" ] || die "no pod matching '$1'"; kubectl -n "${NS}" logs -f "$p"; }
cmd_clean()  { for j in build-nemo-rl build-hypotest-grpo build-interpreter-env build-hypotest-env; do kubectl -n "${NS}" delete trainingworkload "$j" --ignore-not-found; done; }

case "${1:-}" in
  secret) cmd_secret ;; nemo-rl) cmd_nemo_rl ;;
  env) cmd_env ;; env-stage2) cmd_env_stage2 ;;
  status) cmd_status ;; logs) shift; cmd_logs "${@}" ;; clean) cmd_clean ;;
  *) sed -n '5,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
