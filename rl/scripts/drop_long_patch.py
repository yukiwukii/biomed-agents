"""Drop over-long episodes from the GRADIENT before the training forward.

WHY: on 80 GB H100 the 27B trainer forward OOMs on a single long episode
(a 48.5k-token episode was ~1 GiB short; 37.4k fit). Capping episode length was
rejected -- the agent must be free to exhaust all 30 turns. This is the backstop:
any episode still over a memory-safe threshold AFTER NB_OUTPUT_LIMIT truncation is
excluded from the gradient instead of crashing the run.

HOW (and why it is safe): grpo.py flattens the batch FROM
repeated_batch["message_log"] right before training, and
repeated_batch["loss_multiplier"] becomes `sample_mask` (the per-episode loss
weight) a few lines later. Immediately BEFORE that flatten we:
  * truncate each over-long episode's message_log to a small leading stub, so the
    forward/logprob pass runs on a cheap sequence (no OOM), AND
  * set its loss_multiplier to 0, so it contributes ZERO gradient.
Zeroing loss_multiplier is the SAME exclusion mechanism upstream's
`overlong_filtering` uses (grpo.py:1676), so it is a supported path. Truncation
drops WHOLE trailing messages only -- never a partial message -- so token_ids /
generation_logprobs / masks stay internally aligned.

The dropped episode's REWARD still sits in the GRPO group baseline (advantages are
computed from rewards, upstream of the forward), so a long successful episode
still shapes the advantage of the rest of its group -- we just don't backprop its
own tokens.

COVERS BOTH training paths: the training-flatten call block is byte-identical in
grpo_train() (~L1701) and async_grpo_train() (~L2762) -- only the preceding
comment differs -- so anchoring on the call block (not the comment) patches both.
The call takes no step arg so the injected text is identical at both sites.

Inert unless HYPOTEST_DROP_LONG_TOKENS>0. Fail-open with a loud message: on any
error no drop is applied (same as unpatched) -- never silently corrupt a batch.

Idempotent: guarded on the HYPOTEST_DROP_LONG_PATCH marker in nemo_rl_setup.sh.
"""

import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()

# The training flatten CALL BLOCK (no comment) -- identical in grpo_train and
# async_grpo_train. The other batched_message_log_to_flat_message calls use
# different target vars (batched_flat / prompt_batched_flat) or omit
# make_sequence_length_divisible_by, so this block is unique to the two training
# flattens.
anchor = (
    "                    flat_messages, input_lengths = batched_message_log_to_flat_message(\n"
    '                        repeated_batch["message_log"],\n'
    '                        pad_value_dict={"token_ids": tokenizer.pad_token_id},\n'
    '                        make_sequence_length_divisible_by=master_config["policy"][\n'
    '                            "make_sequence_length_divisible_by"\n'
    "                        ],\n"
    "                    )\n"
)
n = src.count(anchor)
if n == 0:
    raise SystemExit(
        "training-flatten anchor not found; upstream changed grpo.py. Re-check the "
        "flat_messages = batched_message_log_to_flat_message(...) call in grpo_train / async_grpo_train."
    )

# Insert the drop call BEFORE every training flatten (sync + async). No step arg,
# so the text is identical at both sites; a module-level counter labels the log.
call = ("                    _hypotest_drop_long(repeated_batch)  # HYPOTEST_DROP_LONG_PATCH\n") + anchor
src = src.replace(anchor, call)  # all occurrences

helper = '''

_HYPOTEST_DROP_CALLS = [0]  # HYPOTEST_DROP_LONG_PATCH


def _hypotest_drop_long(repeated_batch):  # HYPOTEST_DROP_LONG_PATCH
    """Exclude episodes over HYPOTEST_DROP_LONG_TOKENS from the gradient.

    Truncates their message_log to a leading stub (cheap forward) and zeroes their
    loss_multiplier (zero gradient). Never raises: on any error it applies NO drop
    and prints loudly, so a failure degrades to the unpatched (may-OOM) behaviour
    rather than a corrupted batch.
    """
    import os

    thresh = int(os.environ.get("HYPOTEST_DROP_LONG_TOKENS", "0"))
    if thresh <= 0:
        return  # inert unless explicitly enabled
    stub = int(os.environ.get("HYPOTEST_DROP_LONG_STUB", "2048"))
    _HYPOTEST_DROP_CALLS[0] += 1
    call_n = _HYPOTEST_DROP_CALLS[0]
    try:
        message_logs = repeated_batch["message_log"]
        lm = repeated_batch["loss_multiplier"]
        lm = lm.clone() if hasattr(lm, "clone") else lm
        dropped = []
        for i, mlog in enumerate(message_logs):
            total = 0
            for m in mlog:
                tid = m.get("token_ids")
                total += int(tid.shape[0]) if tid is not None else 0
            if total > thresh:
                kept, acc = [], 0
                for m in mlog:
                    kept.append(m)
                    tid = m.get("token_ids")
                    acc += int(tid.shape[0]) if tid is not None else 0
                    if acc >= stub:
                        break
                message_logs[i] = kept
                lm[i] = 0
                dropped.append(total)
        repeated_batch["loss_multiplier"] = lm
        print(
            "[HYPOTEST_DROP_LONG] call %d: dropped %d/%d episode(s) over %d tok; lengths=%s"
            % (call_n, len(dropped), len(message_logs), thresh, sorted(dropped, reverse=True)),
            flush=True,
        )
    except Exception as exc:
        print(
            "[HYPOTEST_DROP_LONG] FAILED -- NO drop applied, run may OOM: %r" % (exc,),
            flush=True,
        )

'''

marker = "\ndef grpo_train("
if marker not in src:
    raise SystemExit("grpo_train def not found; cannot place the helper.")
src = src.replace(marker, helper + marker.lstrip("\n"), 1)

open(path, "w", encoding="utf-8").write(src)
print("patched %s (%d training-flatten site(s): sync + async)" % (path, n))
