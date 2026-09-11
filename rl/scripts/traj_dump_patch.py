"""Dump training trajectories BEFORE the training step.

Upstream already logs token_ids, input_lengths, content and masks via
log_batched_dict_as_jsonl("train_data_step{N}.jsonl") -- but at grpo.py:~2068,
*after* the policy.train() at ~1812. Runs 1-9 all OOM'd at ~1812, so that file
was never written and their transcripts are gone.

This injects the same dump where the data first exists (grpo.py:~1700, right
after the message log is flattened), ~100 lines before the training step.

Recorded per episode:
    input_length          the REAL trained sequence length -- the number that
                          sizes grad_input
    messages[].n_tokens   per-message token counts, from token_ids
    messages[].content    message text, taken from flat_messages["content"]
                          (message_log entries carry token_ids but NOT text at
                          this point -- reading them gives empty strings, which
                          is what the first version of this patch did)

Inert unless HYPOTEST_TRAJ_DUMP is set. Wrapped in try/except: a diagnostic must
never be able to kill the run it is observing.
"""

import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()

anchor = (
    "                    # Convert updated LLMMessageLogType to FlatMessagesType for training\n"
    "                    flat_messages, input_lengths = batched_message_log_to_flat_message(\n"
    '                        repeated_batch["message_log"],\n'
    '                        pad_value_dict={"token_ids": tokenizer.pad_token_id},\n'
    '                        make_sequence_length_divisible_by=master_config["policy"][\n'
    '                            "make_sequence_length_divisible_by"\n'
    "                        ],\n"
    "                    )\n"
)
if anchor not in src:
    raise SystemExit(
        "flat-message anchor not found; upstream changed grpo.py:~1700. "
        "Re-check the batched_message_log_to_flat_message call in grpo_train()."
    )

# One call at the anchor; the body lives at module scope so nothing leaks into
# grpo_train's namespace (the driver memory tracker prints dir() and the first
# version filled it with _i/_m/_fp/... noise).
call = anchor + (
    "                    _hypotest_traj_dump(  # HYPOTEST_TRAJ_DUMP_PATCH\n"
    "                        repeated_batch, flat_messages, input_lengths, total_steps + 1\n"
    "                    )\n"
)
src = src.replace(anchor, call, 1)

helper = '''

def _hypotest_traj_dump(  # HYPOTEST_TRAJ_DUMP_PATCH
    repeated_batch, flat_messages, input_lengths, step
):
    """Write one JSONL line per episode. Never raises."""
    import json
    import os

    out_dir = os.environ.get("HYPOTEST_TRAJ_DUMP")
    if not out_dir:
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "traj_step%d.jsonl" % step)

        message_logs = repeated_batch["message_log"]
        # content is a per-episode list of per-message strings; it is NOT in
        # message_log at this point, only in the flattened view.
        contents = flat_messages.get("content") or []
        rewards = (
            repeated_batch["total_reward"]
            if "total_reward" in repeated_batch
            else None
        )

        with open(path, "w") as fh:
            for i, mlog in enumerate(message_logs):
                texts = contents[i] if i < len(contents) else []
                msgs = []
                for j, m in enumerate(mlog):
                    tid = m.get("token_ids")
                    text = texts[j] if j < len(texts) else (m.get("content") or "")
                    msgs.append(
                        {
                            "role": m.get("role"),
                            "n_tokens": int(tid.shape[0]) if tid is not None else 0,
                            "content": text if isinstance(text, str) else str(text),
                        }
                    )
                fh.write(
                    json.dumps(
                        {
                            "idx": i,
                            "input_length": int(input_lengths[i]),
                            "n_messages": len(msgs),
                            "reward": float(rewards[i]) if rewards is not None else None,
                            "messages": msgs,
                        }
                    )
                    + "\\n"
                )

        print(
            "[HYPOTEST_TRAJ_DUMP] wrote %s (%d episodes, input_lengths=%s)"
            % (path, len(message_logs), [int(x) for x in input_lengths]),
            flush=True,
        )
    except Exception as exc:
        print("[HYPOTEST_TRAJ_DUMP] FAILED (run continues): %r" % (exc,), flush=True)

'''

marker = "\ndef grpo_train("
if marker not in src:
    raise SystemExit("grpo_train def not found; cannot place the helper.")
src = src.replace(marker, helper + marker.lstrip("\n"), 1)

open(path, "w", encoding="utf-8").write(src)
print("patched", path)
