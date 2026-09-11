# rl/configs/qwen3_tp_plan.py
# =============================================================================
# Tensor-parallel plan for Qwen/Qwen3.5-9B.
#
# This is upstream's `examples/custom_parallel/custom_parallel.py` with two
# changes. Nothing here is novel; it is a prefix edit.
#
#   1. Every path is re-rooted at `model.language_model.` instead of `model.`.
#      Qwen3.6 is multimodal-wrapped: the safetensors index has 850 tensors
#      under `model.language_model` and 333 under `model.visual`, and NOTHING
#      at `model.layers.*`. The auto-selected plan targets `model.layers.*`, so
#      on 2026-08-06 it matched exactly ONE key (`lm_head`) out of the whole
#      model, no weights were sharded, and every rank materialised a full
#      55.6 GB replica -> OOM at init with 78.59 GiB allocated.
#
#      This is the same wrapper that forced the LoRA targets to be written as
#      `*language_model*q_proj` (docs/rl_context.md 5). Same cause, same fix.
#
#   2. `linear_attn.*` is deliberately NOT sharded. 48 of the 64 layers are
#      gated-delta-net rather than full attention, and sharding conv1d / SSM
#      state is what NVIDIA's NemotronH Mamba recipes avoid. It is 11.12 GB
#      left replicated -- the price of not guessing.
#
# The single most important line is `lm_head`:
#
#     ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)
#
# That is what makes the logits a vocab-sharded DTensor, which is the ONLY way
# to reach ChunkedDistributedLogprob (model_utils.py:250) -- the one logprob
# path that does not materialise a whole-vocab fp32 tensor. The auto plan used
# `output_layouts=Replicate(), use_local_output=True`, which all-gathers back to
# full vocab and returns a plain tensor, so it would have landed in
# `_compute_local_logprobs` at ANY tp size. There is no YAML string for the
# style we need -- `translate_parallel_style` only offers `colwise_rep`, which
# is the Replicate one -- which is why this has to be a Python file.
# NeMo RL's own parallelize.py:454 special-cases exactly this rewrite, but the
# Automodel code path that actually runs does not.
#
# Coverage, verified against the checkpoint's 1199 real tensor names by
# rl/scripts/check_tp_plan.py:
#
#   mlp (64 layers)        34.23 GB   sharded
#   self_attn (16 layers)   3.36 GB   sharded
#   embed_tokens + lm_head  5.08 GB   sharded
#   linear_attn (48 lyr)   11.12 GB   replicated, deliberately
#   visual + mtp            1.77 GB   replicated (not used in training)
#
#   -> 77% shardable. At TP=4: 23.56 GB/GPU of weights.
#
# Divisibility (all clean): intermediate_size 17408, num_attention_heads 24,
# num_key_value_heads 4, vocab_size 248320 -- each divisible by 2 and by 4.
# =============================================================================

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

_LM = "model.language_model"

custom_parallel_plan: dict[str, ParallelStyle] = {
    # Vocab-sharded DTensor logits. THE line this file exists for.
    "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    f"{_LM}.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
    # Full-attention layers only -- 16 of 64 (full_attention_interval: 4).
    # The other 48 have `linear_attn` instead and match none of these.
    f"{_LM}.layers.*.self_attn.q_proj": ColwiseParallel(),
    f"{_LM}.layers.*.self_attn.k_proj": ColwiseParallel(),
    f"{_LM}.layers.*.self_attn.v_proj": ColwiseParallel(),
    f"{_LM}.layers.*.self_attn.o_proj": RowwiseParallel(),
    # MLP is in all 64 layers and is 34.23 GB -- the bulk of the win.
    f"{_LM}.layers.*.mlp.gate_proj": ColwiseParallel(),
    f"{_LM}.layers.*.mlp.up_proj": ColwiseParallel(),
    f"{_LM}.layers.*.mlp.down_proj": RowwiseParallel(),
}
