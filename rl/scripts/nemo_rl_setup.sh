#!/usr/bin/env bash
# rl/scripts/nemo_rl_setup.sh
# =============================================================================
# Turn the stock nvcr.io/nvidia/nemo-rl image into a working hypotest trainer,
# at pod start, then exec the real command.
#
#     nemo_rl_setup.sh bash /config/rl/scripts/01_grpo.sh
#
# This replaces rl/docker/Dockerfile.train entirely. Everything that Dockerfile
# added is available without a build:
#
#   COPY bbh-third-party/Gym /opt/Gym   -> the Gym tree is on the scratch PVC
#   COPY rl/configs, rl/data            -> already mounted as ConfigMaps
#   sed the aviary requirements         -> one line, below
#   mkdir + symlink Gym-workspace       -> one line, below
#   uv sync --extra nemo_gym            -> runs here
#
# WHY THE GYM TREE IS COPIED, NOT SYMLINKED FROM THE PVC
# ------------------------------------------------------
# We modify it (the requirements sed, and ng_run writes per-server .venvs into
# the tree at runtime). The PVC copy is the shared bbh-third-party bundle that
# the host and any other pod also read, so mutating it in place would corrupt a
# shared checkout. 58 MB copies in seconds.
#
# WHY THE REQUIREMENTS SED
# ------------------------
# bbh added `-e ../../../hypotest` to Gym/resources_servers/aviary/requirements.txt
# so hypotest_app.py could import the env in-process. We run client_app.py
# (remote mode) instead, and that relative path does not exist here -- leaving it
# in makes ng_run's per-server `uv venv` build fail.
# =============================================================================
set -euo pipefail

GYM_SRC="${GYM_SRC:-/mlbio_scratch/wangsaja/bbh-third-party/Gym}"
GYM_DST="${GYM_DST:-/opt/Gym}"
NEMO_RL_ROOT="${NEMO_RL_ROOT:-/opt/nemo-rl}"
SETUP_MARKER="${GYM_DST}/.hypotest-setup-done"

log() { printf '\033[36m[nemo_rl_setup]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[31m[nemo_rl_setup] FATAL:\033[0m %s\n' "$*" >&2; exit 1; }

if [ -f "${SETUP_MARKER}" ]; then
  log "already set up; skipping"
else
  # ---- preconditions, checked loudly ---------------------------------------
  [ -d "${GYM_SRC}" ] || die "GYM_SRC=${GYM_SRC} not found. Is the scratch PVC mounted?"
  [ -d "${NEMO_RL_ROOT}" ] || die "NEMO_RL_ROOT=${NEMO_RL_ROOT} not found. Is this the NeMo RL image?"
  touch "${NEMO_RL_ROOT}/.writetest" 2>/dev/null \
    || die "${NEMO_RL_ROOT} is not writable. The uv workspace must be modified here; bake this setup into an image instead."
  rm -f "${NEMO_RL_ROOT}/.writetest"

  log "copying Gym  ${GYM_SRC} -> ${GYM_DST}"
  rm -rf "${GYM_DST}"
  cp -a "${GYM_SRC}" "${GYM_DST}"

  log "stripping the in-process hypotest install from the aviary requirements"
  sed -i '/hypotest/d' "${GYM_DST}/resources_servers/aviary/requirements.txt"

  # RL/pyproject.toml declares 3rdparty/Gym-workspace/Gym as a uv workspace
  # member and maps nemo_gym to it, but the directory is a git submodule that
  # never made it into the bbh-third-party bundle (publish_third_party.sh rsyncs
  # with --exclude '.git/'). Without this, `uv sync --extra nemo_gym` cannot
  # resolve. Pointed at the patched tree, which is the one with client_app.py.
  log "restoring the Gym-workspace uv member"
  mkdir -p "${NEMO_RL_ROOT}/3rdparty/Gym-workspace"
  ln -sfn "${GYM_DST}" "${NEMO_RL_ROOT}/3rdparty/Gym-workspace/Gym"

  log "uv sync --extra nemo_gym  (needs PyPI egress; several minutes on a cold cache)"
  cd "${NEMO_RL_ROOT}"
  if ! uv sync --extra nemo_gym --no-dev; then
    die "uv sync failed. Most likely the pod has no PyPI egress -- check with
  kubectl -n \$NS exec <pod> -- curl -sI https://pypi.org/simple/
If egress is blocked, bake this setup into an image (rl/docker/Dockerfile.train)
rather than doing it at pod start."
  fi

  touch "${SETUP_MARKER}"
  log "setup complete"
fi

# ---------------------------------------------------------------------------
# Make policy.dtensor_cfg.custom_parallel_plan work on the V2 worker.
#
# V1 resolves the dotted path (nemo_rl/models/dtensor/parallelize.py:679,
# `get_class(custom_parallel_plan)`), but the automodel path that V2 uses
# forwards the raw config value into FSDP2Config(tp_plan=...) without resolving
# it (nemo_rl/models/automodel/setup.py:434). LoRA forces V2
# (lm_policy.py:141), so without this the documented config key silently hands
# Automodel a string instead of a plan.
#
# Deliberately NOT hydra's get_class, which V1 uses. get_class asserts the
# resolved object is a class and raises ValueError otherwise -- and a parallel
# plan is a dict. (V1's own string path looks broken for that reason:
# parallelize.py:678-689 catches the failure and reports "not valid".) Plain
# importlib has no such assertion and handles both accepted forms: a module
# attribute that is already a dict, or a zero-arg factory returning one.
#
# Idempotent: guarded on the marker comment.
# ---------------------------------------------------------------------------
AUTOMODEL_SETUP="${NEMO_RL_ROOT}/nemo_rl/models/automodel/setup.py"
if [ -f "${AUTOMODEL_SETUP}" ] && ! grep -q HYPOTEST_TP_PLAN_PATCH "${AUTOMODEL_SETUP}"; then
  log "patching automodel/setup.py to resolve custom_parallel_plan (V2 gap)"
  python3 - "${AUTOMODEL_SETUP}" <<'PY' || die "custom_parallel_plan patch failed"
import sys

path = sys.argv[1]
src = open(path).read()
anchor = '    tp_plan = config["dtensor_cfg"].get("custom_parallel_plan", None)\n'
if anchor not in src:
    raise SystemExit(
        f"anchor line not found in {path}; upstream changed. "
        "Re-check nemo_rl/models/automodel/setup.py around the tp_plan assignment."
    )
patched = anchor + (
    "    if isinstance(tp_plan, str):  # HYPOTEST_TP_PLAN_PATCH\n"
    "        import importlib as _il\n"
    "        _mod, _, _attr = tp_plan.rpartition('.')\n"
    "        tp_plan = getattr(_il.import_module(_mod), _attr)\n"
    "        if callable(tp_plan):\n"
    "            tp_plan = tp_plan()\n"
    "        assert isinstance(tp_plan, dict), (\n"
    "            f'custom_parallel_plan resolved to {type(tp_plan).__name__}, expected dict'\n"
    "        )\n"
)
open(path, "w").write(src.replace(anchor, patched, 1))
print("patched", path)
PY
fi

# ---------------------------------------------------------------------------
# Stop get_next_token_logprobs_from_logits from upcasting the WHOLE logits
# tensor to fp32 before it chunks.
#
# model_utils.py:1379 does an unconditional
#     next_token_logits = next_token_logits.to(torch.float32)
# BEFORE branching. On the vocab-parallel / DTensor branches that tensor is then
# handed to ChunkedDistributedLogprob, whose docstring says its entire purpose
# is "logits casting from float16 or bfloat16 -> float32 is performed inside the
# chunk loop to avoid materializing a whole float32 logits tensor". The pre-cast
# defeats the chunking it is about to call.
#
# Measured 2026-08-06, run 2: train() died asking for exactly 9.13 GiB =
# 39,477 tokens x (248320/4 vocab shard) x 4 bytes -- this cast, at TP=4. Only
# the non-parallel `else` branch actually needs a materialised fp32 tensor, so
# the cast moves there.
#
# Idempotent: guarded on the marker comment.
# ---------------------------------------------------------------------------
MODEL_UTILS="${NEMO_RL_ROOT}/nemo_rl/distributed/model_utils.py"
if [ -f "${MODEL_UTILS}" ] && ! grep -q HYPOTEST_FP32_CAST_PATCH "${MODEL_UTILS}"; then
  log "patching model_utils.py to defer the fp32 logits cast (train() OOM)"
  python3 - "${MODEL_UTILS}" <<'PY' || die "fp32 cast patch failed"
import sys

path = sys.argv[1]
src = open(path).read()

# 1. Drop the unconditional pre-cast.
pre = "    next_token_logits = next_token_logits.to(torch.float32)\n\n    if vocab_parallel_group is not None:\n"
if pre not in src:
    raise SystemExit(
        "pre-cast anchor not found; upstream changed "
        "get_next_token_logprobs_from_logits. Re-check model_utils.py:~1379."
    )
src = src.replace(
    pre,
    "    # HYPOTEST_FP32_CAST_PATCH: cast deferred into the non-parallel branch;\n"
    "    # the chunked kernels upcast per chunk and must NOT get a full fp32 tensor.\n"
    "    if vocab_parallel_group is not None:\n",
    1,
)

# 2. Restore it only where a full fp32 tensor is genuinely required.
post = "    else:\n        # Remove last position's logits\n        next_token_logits_wo_last = next_token_logits[:, :-1]\n"
if post not in src:
    raise SystemExit("non-parallel branch anchor not found; upstream changed model_utils.py.")
src = src.replace(
    post,
    "    else:\n"
    "        next_token_logits = next_token_logits.to(torch.float32)  # HYPOTEST_FP32_CAST_PATCH\n"
    "        # Remove last position's logits\n"
    "        next_token_logits_wo_last = next_token_logits[:, :-1]\n",
    1,
)

open(path, "w").write(src)
print("patched", path)
PY
fi

# ---------------------------------------------------------------------------
# Pass logprob_chunk_size through on the TRAINING path too.
#
# The same config key is wired on one call site and dropped on the other:
#
#   automodel/train.py:685   get_logprobs_from_vocab_parallel_logits(
#                              ..., chunk_size=self.logprob_chunk_size, ...)   <- get_logprobs
#   model_utils.py:1398      get_logprobs_from_vocab_parallel_logits(
#                              next_token_logits, input_ids,
#                              seq_index=..., sampling_params=...)             <- train(), NO chunk_size
#
# So `policy.logprob_chunk_size: 1024` is honoured during get_logprobs and
# silently ignored during train(). chunk_size=None then selects DistributedLogprob
# (model_utils.py:917) instead of ChunkedDistributedLogprob, which materialises a
# full fp32 [B, S, vocab/TP] intermediate inside _compute_distributed_log_softmax.
#
# Measured 2026-08-06, run 3: exactly 8.00 GiB = 34,592 tokens x (248320/4) x 4,
# with 6.02 GiB free. Chunked at 1024 the same intermediate is 0.24 GiB.
#
# `get_next_token_logprobs_from_logits` has no access to the policy config
# (it would have to be threaded through LossPostProcessor -> prepare_loss_input),
# so the chunk size comes from the environment, defaulting to the same 1024 the
# config asks for. Override with HYPOTEST_LOGPROB_CHUNK_SIZE.
#
# Idempotent: guarded on the marker comment.
# ---------------------------------------------------------------------------
if [ -f "${MODEL_UTILS}" ] && ! grep -q HYPOTEST_TRAIN_CHUNK_PATCH "${MODEL_UTILS}"; then
  log "patching model_utils.py to chunk logprobs on the train() path too"
  python3 - "${MODEL_UTILS}" <<'PY' || die "train-path chunk patch failed"
import sys

path = sys.argv[1]
src = open(path).read()
anchor = (
    "    elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):\n"
    "        logprobs = get_logprobs_from_vocab_parallel_logits(\n"
    "            next_token_logits,\n"
    "            input_ids,\n"
    "            seq_index=seq_index,\n"
    "            sampling_params=sampling_params,\n"
    "        )\n"
)
if anchor not in src:
    raise SystemExit(
        "DTensor branch anchor not found in get_next_token_logprobs_from_logits; "
        "upstream changed model_utils.py:~1398."
    )
replacement = (
    "    elif isinstance(next_token_logits, torch.distributed.tensor.DTensor):\n"
    "        import os as _os  # HYPOTEST_TRAIN_CHUNK_PATCH\n"
    "        logprobs = get_logprobs_from_vocab_parallel_logits(\n"
    "            next_token_logits,\n"
    "            input_ids,\n"
    "            seq_index=seq_index,\n"
    "            chunk_size=int(_os.environ.get('HYPOTEST_LOGPROB_CHUNK_SIZE', '1024')),\n"
    "            sampling_params=sampling_params,\n"
    "        )\n"
)
open(path, "w").write(src.replace(anchor, replacement, 1))
print("patched", path)
PY
fi

# ---------------------------------------------------------------------------
# Halve the grad_input buffer in ChunkedDistributedLogprob.backward.
#
# THIS IS NOT AN UPSTREAM BUG. The three patches above each fix something that
# was plainly wired wrong; this one does not. Upstream's code is numerically
# correct and the fp32 buffer is a deliberate precision choice. State that
# clearly before changing it.
#
# model_utils.py:~329 preallocates the whole gradient, for the whole sequence:
#
#     grad_input = torch.zeros_like(vocab_parallel_logits, dtype=torch.float32)
#
# then the chunk loop performs TWO operations *on that buffer*:
#
#     grad_input_chunk.copy_(is_chosen.float().sub_(softmax_output))
#     grad_input_chunk.mul_(grad_output[...].unsqueeze(dim=-1))     <- in-buffer
#
# So the buffer's dtype governs the ARITHMETIC of the multiply, not merely the
# storage. fp32 there means the multiply runs in fp32 and the value is rounded
# exactly once, by autograd, on return. Simply flipping the dtype to bf16 is
# therefore NOT equivalent -- measured on a [1, 4096, 2048] case, it changes
# 2,097,311 of 8,388,608 elements, max |diff| 1.56e-2. Do not do that.
#
# The fix is to reorder, not to re-type: fold the multiply into the fp32
# temporary that already exists per chunk, so the narrowing happens once, on
# write, and the destination can be bf16. Same single rounding, bit-identical
# output (0 differing elements of 8,388,608, verified), half the buffer.
#
# Measured 2026-08-10, run 5: train() died in backward asking for exactly
# 9.00 GiB = 38,912 tokens x (248320/4 vocab shard) x 4 bytes, with 6.73 GiB
# free. At the input dtype (bf16) the same buffer is 4.50 GiB and fits.
#
# The per-chunk fp32 temporary is 1024 tokens wide and costs nothing.
#
# Idempotent: guarded on the marker comment.
# ---------------------------------------------------------------------------
if [ -f "${MODEL_UTILS}" ] && ! grep -q HYPOTEST_GRAD_BUFFER_PATCH "${MODEL_UTILS}"; then
  log "patching model_utils.py to narrow the backward grad_input buffer"
  python3 - "${MODEL_UTILS}" <<'PY' || die "grad buffer patch failed"
import sys

path = sys.argv[1]
src = open(path).read()

# 1. Allocate at the input dtype instead of fp32.
alloc = (
    "        grad_input: torch.Tensor = torch.zeros_like(\n"
    "            vocab_parallel_logits, dtype=torch.float32\n"
    "        )\n"
)
if alloc not in src:
    raise SystemExit(
        "grad_input allocation anchor not found; upstream changed "
        "ChunkedDistributedLogprob.backward. Re-check model_utils.py:~329."
    )
src = src.replace(
    alloc,
    "        # HYPOTEST_GRAD_BUFFER_PATCH: buffer at the input dtype, not fp32.\n"
    "        # Safe ONLY together with the fused multiply below, which keeps the\n"
    "        # arithmetic in the per-chunk fp32 temporary so there is still exactly\n"
    "        # one rounding. Narrowing this alone would change the gradient.\n"
    "        grad_input: torch.Tensor = torch.zeros_like(\n"
    "            vocab_parallel_logits, dtype=vocab_parallel_logits.dtype\n"
    "        )\n",
    1,
)

# 2. Fuse the multiply into the fp32 temporary, so the copy is the only narrowing.
loop = (
    "            grad_input_chunk.copy_(\n"
    "                is_chosen.float().sub_(softmax_output)\n"
    "            )  # inplace copy\n"
    "            grad_input_chunk.mul_(\n"
    "                grad_output[:, chunk_start:chunk_end].unsqueeze(dim=-1)\n"
    "            )\n"
)
if loop not in src:
    raise SystemExit(
        "copy_/mul_ anchor not found in ChunkedDistributedLogprob.backward; "
        "upstream changed the chunk loop. Re-check model_utils.py:~354."
    )
src = src.replace(
    loop,
    "            # HYPOTEST_GRAD_BUFFER_PATCH: multiply in the fp32 temporary,\n"
    "            # then narrow once on write. Was: copy_ then mul_ in-buffer.\n"
    "            grad_input_chunk.copy_(\n"
    "                is_chosen.float()\n"
    "                .sub_(softmax_output)\n"
    "                .mul_(grad_output[:, chunk_start:chunk_end].unsqueeze(dim=-1))\n"
    "            )\n",
    1,
)

open(path, "w").write(src)
print("patched", path)
PY
fi

# ---------------------------------------------------------------------------
# Make FSDP2's output_dtype follow policy.precision (the fp32-logits root cause).
#
# `policy.precision: "bfloat16"` never reached the logits: MixedPrecisionPolicy
# pins output_dtype=torch.float32 (automodel/setup.py:~443, no config knob), and
# output_dtype is what FSDP casts module OUTPUTS to. So lm_head emitted fp32, and
# grad_input cost 65536 x 62,080 x 4 = 15.16 GiB instead of x2 = 7.58 GiB.
#
# This is what makes the grad-buffer patch below actually do something: that one
# is dtype-FOLLOWING, so it halves a bf16 input and does nothing for an fp32 one.
# The two are a pair -- neither is sufficient alone.
#
# reduce_dtype stays fp32, so gradient reduction is unchanged, and the chunked
# logprob kernels upcast per chunk by design. The unaudited exposure is any other
# consumer of module outputs that assumed fp32.
#
# Gated on HYPOTEST_BF16_LOGITS: unset = upstream behaviour.
#
# Idempotent: guarded on the marker comment.
# ---------------------------------------------------------------------------
if [ -f "${AUTOMODEL_SETUP}" ] && ! grep -q HYPOTEST_BF16_LOGITS_PATCH "${AUTOMODEL_SETUP}"; then
  log "patching automodel/setup.py so output_dtype follows policy.precision"
  python3 /config/rl/scripts/bf16_logits_patch.py "${AUTOMODEL_SETUP}" || die "bf16 logits patch failed"
fi

# ---------------------------------------------------------------------------
# Resolve the chat_template PATH into its CONTENT.
#
# OpenAIServingChat wants the template CONTENT. vLLM's own api_server resolves a
# path first via load_chat_template(); this code path splats
# http_server_serving_chat_kwargs straight into the constructor
# (vllm_worker_async.py:~494) and never does. So the PATH became the template,
# every render produced that same constant string, and run 11 died with
#   AssertionError: Found possibly non-monotonically increasing trajectory!
#   Template repr (detokenized): '/config/rl/configs/qwen3_5_retain_thinking.jinja'
#
# Hidden for eleven runs: that assert is behind
# `if not model_prefix_token_ids: return`, so it is only reachable on a SECOND
# agent turn -- which never happened until the reasoning parser was fixed.
#
# The template matters because Qwen3.5 has no `preserve_thinking` flag (Qwen3.6
# does). Verified locally on a 3-turn conversation ending in FORCE_MSG:
#   stock Qwen3.5 template : keeps 1 of 3 thinking blocks -> prefix too short
#   qwen3_5_retain_thinking: keeps all 3 + generation prompt -> monotonic
#
# Idempotent: guarded on the marker comment.
# ---------------------------------------------------------------------------
VLLM_WORKER="${NEMO_RL_ROOT}/nemo_rl/models/generation/vllm/vllm_worker_async.py"
if [ -f "${VLLM_WORKER}" ] && ! grep -q HYPOTEST_CHAT_TEMPLATE_PATCH "${VLLM_WORKER}"; then
  log "patching vllm_worker_async.py to resolve the chat_template path"
  python3 /config/rl/scripts/chat_template_patch.py "${VLLM_WORKER}" || die "chat template patch failed"
fi

# ---------------------------------------------------------------------------
# DIAGNOSTIC: dump trajectories BEFORE the training step.
#
# THIS IS TEMPORARY. Remove it once the token budget question is settled.
#
# Upstream already logs everything we want -- token_ids, input_lengths, content,
# masks -- via log_batched_dict_as_jsonl("train_data_step{N}.jsonl") at
# grpo.py:~2068. The problem is purely one of ORDER:
#
#     grpo.py:1805   print("Preparing for training...")   <- every run dies here
#     grpo.py:1812   policy.train(...)                    <- the OOM
#     grpo.py:2068   log_batched_dict_as_jsonl(...)       <- never reached
#
# So a run that OOMs in backward writes no trajectories at all, which is why the
# transcripts from runs 5-9 are gone. This injects the same dump at the point the
# data is first available (grpo.py:~1700, right after the message log is
# flattened), roughly 100 lines before the OOM.
#
# What it records per episode:
#   input_length          the REAL trained sequence length -- the number that
#                         sizes grad_input, and the whole reason for the run
#   messages[].n_tokens   per-message token counts, which is what attributes the
#                         total to thinking vs tool output vs notebook
#   messages[].content    full text, so nothing has to be inferred
#
# Inert unless HYPOTEST_TRAJ_DUMP is set to a directory. Wrapped in try/except:
# a diagnostic must never be able to kill the run it is observing.
#
# NOTE on `env.should_log_nemo_gym_responses`: that flag is INVERTED from what
# its name suggests. Upstream guards the train_data dump with
# `if not _should_log_nemo_gym_responses(...)`, so setting it TRUE (as the config
# does) SKIPS the built-in dump. This patch does not depend on it either way.
#
# Idempotent: guarded on the marker comment.
# ---------------------------------------------------------------------------
GRPO_ALGO="${NEMO_RL_ROOT}/nemo_rl/algorithms/grpo.py"
if [ -f "${GRPO_ALGO}" ] && ! grep -q HYPOTEST_TRAJ_DUMP_PATCH "${GRPO_ALGO}"; then
  log "patching grpo.py to dump trajectories before the training step (diagnostic)"
  python3 /config/rl/scripts/traj_dump_patch.py "${GRPO_ALGO}" || die "traj dump patch failed"
fi

# ---------------------------------------------------------------------------
# HYPOTEST_DROP_LONG_PATCH: exclude over-long episodes from the gradient so the
# 27B trainer forward cannot OOM on a single long episode (80 GB H100). Truncates
# their message_log to a stub + zeroes loss_multiplier, right before the flatten.
# Inert unless HYPOTEST_DROP_LONG_TOKENS>0 is set on the trainer pod. See
# rl/scripts/drop_long_patch.py. Idempotent: guarded on the marker comment.
# ---------------------------------------------------------------------------
if [ -f "${GRPO_ALGO}" ] && ! grep -q HYPOTEST_DROP_LONG_PATCH "${GRPO_ALGO}"; then
  log "patching grpo.py to drop over-long episodes from the gradient (memory backstop)"
  python3 /config/rl/scripts/drop_long_patch.py "${GRPO_ALGO}" || die "drop-long patch failed"
fi

# ---------------------------------------------------------------------------
# HYPOTEST_GROUP_KEY_PATCH: fix the GRPO advantage-grouping key for this
# multi-turn env. NeMo RL groups a prompt's generations by torch.unique() over
# ALL user/system messages in the trajectory (_extract_prompt_only_messages).
# This env records per-turn observations, env-state, and wall-clock time warnings
# as USER-role messages, which diverge per rollout -> every generation gets a
# unique key -> groups of size 1 -> leave-one-out baseline == reward ->
# advantage == 0 for the whole batch (zero gradient -- the real cause of the flat
# eval, confirmed on a traj dump: 3x4 prompts seen as 12 groups of 1). The fix
# keys grouping on the INITIAL prompt only (messages before the first assistant
# turn), which is identical across a prompt's generations. See group_key_patch.py.
# Idempotent: guarded on the marker.
# ---------------------------------------------------------------------------
if [ -f "${GRPO_ALGO}" ] && ! grep -q HYPOTEST_GROUP_KEY_PATCH "${GRPO_ALGO}"; then
  log "patching grpo.py: fix GRPO grouping key (advantages were all zero)"
  python3 /config/rl/scripts/group_key_patch.py "${GRPO_ALGO}" || die "group-key patch failed"
fi

export NEMO_GYM_ROOT="${GYM_DST}"
export PYTHONUNBUFFERED=1
# rl/configs is mounted as a ConfigMap at /config/rl/configs. Putting /config on
# the path makes the parallel plan importable as
# rl.configs.qwen3_tp_plan.custom_parallel_plan (PEP 420 namespace packages --
# no __init__.py needed, which matters because ConfigMap mounts are read-only).
export PYTHONPATH="/config${PYTHONPATH:+:${PYTHONPATH}}"

[ "$#" -gt 0 ] || { log "no command given; exiting"; exit 2; }
log "exec: $*"
exec "$@"
