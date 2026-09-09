#!/usr/bin/env bash
# rl/scripts/01_grpo.sh
# =============================================================================
# GATE 4: GRPO training.
#
#   bash rl/scripts/01_grpo.sh                    # full run
#   GRPO_SMOKE=1 bash rl/scripts/01_grpo.sh       # ~20 steps, 2x4 rollouts
#
# Runs NeMo RL's NeMo Gym entrypoint. That entrypoint — not examples/run_grpo.py —
# is the only one that can train a multi-turn agentic environment: run_grpo.py
# goes through max_rollout_turns: 1 single-turn recipes and cannot represent a
# hypotest episode. (bbh/train/configs/*.yaml name those single-turn math
# recipes as their base; that is a bug in bbh's placeholders, not a pattern to
# copy.)
#
# Env:
#   HYPOTEST_SERVER_URL   required   http://hypotest-env:8008
#   HYPOTEST_API_KEY      required   matches server.yaml
#   HYPOTEST_GRPO_CONFIG  default /config/rl/configs/grpo_hypotest_27b.yaml
#   HYPOTEST_GYM_CONFIG   default /config/rl/configs/gym_hypotest_remote.yaml
#   NEMO_RL_ROOT          default /opt/nemo-rl
#   AGENT_MAX_STEPS       default 20
#   GRPO_SMOKE            unset      set to 1 for the cheap 20-step gate
#   WANDB_API_KEY         optional   logging is disabled without it
# =============================================================================
set -euo pipefail

grn() { printf '\033[32m%s\033[0m\n' "$*"; }
red() { printf '\033[31m%s\033[0m\n' "$*" >&2; }
die() { red "FATAL: $*"; exit 1; }
need() { [ -n "${!1:-}" ] || die "$1 is not set."; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="${HYPOTEST_CONFIG_DIR:-${REPO_ROOT}/rl/configs}"
NEMO_RL_ROOT="${NEMO_RL_ROOT:-/opt/nemo-rl}"

export HYPOTEST_GRPO_CONFIG="${HYPOTEST_GRPO_CONFIG:-${CONFIG_DIR}/grpo_hypotest_27b.yaml}"
# Interpolated by the GRPO config's env.nemo_gym.config_paths.
export HYPOTEST_GYM_CONFIG="${HYPOTEST_GYM_CONFIG:-${CONFIG_DIR}/gym_hypotest_remote.yaml}"
export AGENT_MAX_STEPS="${AGENT_MAX_STEPS:-20}"

need HYPOTEST_SERVER_URL
need HYPOTEST_API_KEY
[ -d "${NEMO_RL_ROOT}" ] || die "NEMO_RL_ROOT='${NEMO_RL_ROOT}' is not a directory."
[ -f "${HYPOTEST_GRPO_CONFIG}" ] || die "${HYPOTEST_GRPO_CONFIG} missing."
[ -f "${HYPOTEST_GYM_CONFIG}" ] || die "${HYPOTEST_GYM_CONFIG} missing."

# ---------------------------------------------------------------------------
# The workspace member that the bbh-third-party bundle is missing.
#
# RL/pyproject.toml declares 3rdparty/Gym-workspace/Gym as a uv workspace member
# and maps `nemo_gym` to it, but bbh/scripts/maint/publish_third_party.sh rsyncs
# with --exclude '.git/', so submodule content never reached the bundle. The
# trainer image already fixes this; re-check here so a bind-mounted or
# locally-built tree fails loudly rather than at `uv sync`.
# ---------------------------------------------------------------------------
GYM_MEMBER="${NEMO_RL_ROOT}/3rdparty/Gym-workspace/Gym"
[ -e "${GYM_MEMBER}" ] \
  || die "${GYM_MEMBER} missing. Symlink it at the patched Gym tree (see rl/docker/Dockerfile.train)."

# Same pre-flight as the smoke gate: an unreachable env server does not raise,
# it just yields an entire training run of 0.0 rewards.
grn ">> Checking the dataset server at ${HYPOTEST_SERVER_URL}"
INFO="$(curl -fsS -m 15 -H "X-API-Key: ${HYPOTEST_API_KEY}" "${HYPOTEST_SERVER_URL}/info" 2>/dev/null)" \
  || die "GET ${HYPOTEST_SERVER_URL}/info failed. Is Workload A Ready?"
grn "   /info -> ${INFO}"

OVERRIDES=()

if [ "${GRPO_SMOKE:-0}" = "1" ]; then
  grn ">> GRPO_SMOKE=1 — short gate run, not a real training run"
  OVERRIDES+=(
    "++grpo.num_prompts_per_step=2"
    "++grpo.num_generations_per_prompt=4"
    "++grpo.max_num_steps=20"
    "++grpo.val_period=10"
    "++checkpointing.enabled=false"
    "++logger.wandb.name=smoke-$(date +%Y%m%d-%H%M%S)"
  )
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  red ">> WANDB_API_KEY unset — disabling W&B. You will be flying blind on the reward curve."
  OVERRIDES+=("++logger.wandb_enabled=false")
fi

grn ">> Launching GRPO"
grn "   config:  ${HYPOTEST_GRPO_CONFIG}"
grn "   gym:     ${HYPOTEST_GYM_CONFIG}"
grn "   env url: ${HYPOTEST_SERVER_URL}"
grn "   steps:   AGENT_MAX_STEPS=${AGENT_MAX_STEPS}"

cd "${NEMO_RL_ROOT}"
exec uv run --extra nemo_gym python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${HYPOTEST_GRPO_CONFIG}" \
  "${OVERRIDES[@]}"
