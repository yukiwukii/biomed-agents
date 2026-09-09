"""Make FSDP2's output_dtype follow policy.precision instead of being hardcoded fp32.

WHY
---
`policy.precision: "bfloat16"` does NOT reach the logits. FSDP2's mixed-precision
policy is constructed with output_dtype hardcoded to fp32
(nemo_rl/models/automodel/setup.py:~443):

    mp_policy=MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,     # <- hardcoded, no config knob
    )

output_dtype is what FSDP casts module OUTPUTS to, so the lm_head logits come out
fp32 no matter what precision says. That is why the grad-buffer patch (which is
dtype-following) halves nothing, and why grad_input costs

    65536 x 62,080 x 4 = 15.16 GiB    instead of    x 2 = 7.58 GiB

WHAT THIS CHANGES, HONESTLY
---------------------------
reduce_dtype stays fp32, so gradient reduction is untouched. The chunked logprob
kernels upcast per chunk internally -- that is their documented purpose -- so the
logprob path is unaffected by a bf16 arrival. The real exposure is any OTHER
consumer of module outputs that assumed fp32; that has not been exhaustively
audited. Treat a loss curve that differs from previous runs as a suspect.

GATING: default ON once patched, opt out with HYPOTEST_BF16_LOGITS=0.

Deliberately NOT default-off. `setup_distributed` runs inside a Ray actor, and if
an opt-IN env var failed to propagate there the patch would silently no-op and
the run would OOM exactly as before -- burning a GPU slot to learn nothing. The
patch is only applied when we want it, so its presence is the intent. It prints
the dtype it selected so the run log proves which branch was taken; grep the log
for HYPOTEST_BF16_LOGITS.
"""

import sys

path = sys.argv[1]
src = open(path).read()

anchor = """        mp_policy=MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
        ),
"""
if anchor not in src:
    raise SystemExit(
        "MixedPrecisionPolicy anchor not found; upstream changed "
        "nemo_rl/models/automodel/setup.py:~440. Re-check the FSDP2Config construction."
    )

patched = """        # HYPOTEST_BF16_LOGITS_PATCH: follow policy.precision instead of pinning
        # fp32. `dtype` is runtime_config.dtype, i.e. exactly what
        # policy.precision asked for. reduce_dtype stays fp32 on purpose -- only
        # the module OUTPUT cast changes, which is what makes the logits (and
        # therefore grad_input) bf16 rather than fp32.
        #
        # Default ON: this runs in a Ray actor, and an opt-in env var that failed
        # to propagate would silently no-op and OOM as before. Opt out with
        # HYPOTEST_BF16_LOGITS=0.
        mp_policy=MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            output_dtype=_hypotest_output_dtype(dtype),
        ),
"""

src = src.replace(anchor, patched, 1)

# Helper, injected once at module scope. Prints which branch it took so the run
# log is proof rather than assumption -- grep for HYPOTEST_BF16_LOGITS.
helper = '''

def _hypotest_output_dtype(dtype):  # HYPOTEST_BF16_LOGITS_PATCH
    """FSDP2 output_dtype: policy.precision unless explicitly disabled."""
    import os as _os

    _off = _os.environ.get("HYPOTEST_BF16_LOGITS") == "0"
    _out = torch.float32 if _off else dtype
    print(
        "[HYPOTEST_BF16_LOGITS] output_dtype=%s (param_dtype=%s, disabled=%s)"
        % (_out, dtype, _off),
        flush=True,
    )
    return _out

'''

marker = "\ndef setup_distributed("
if marker not in src:
    raise SystemExit("setup_distributed def not found; cannot place the helper.")
src = src.replace(marker, helper + marker.lstrip("\n"), 1)

open(path, "w").write(src)
print("patched", path)
