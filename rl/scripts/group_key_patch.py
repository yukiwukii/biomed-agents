"""Fix the GRPO advantage-grouping key for this multi-turn agentic env.

THE BUG (root cause of the "flat eval"). NeMo RL groups a prompt's generations
for the leave-one-out baseline by content: it builds a per-episode key from
`_extract_prompt_only_messages()` (every user/system message in the trajectory,
assistant turns excluded), flattens it to token ids, and calls
`torch.unique(prompts, dim=0)` in `calculate_baseline_and_std_per_prompt`
(nemo_rl/algorithms/utils.py). That is correct for standard GRPO, where the only
user/system content is the FIXED prompt and all divergence lives in the excluded
assistant response.

It breaks here because this env records per-turn OBSERVATIONS, env-state summaries
("N commands executed") and time warnings ("... {remaining} seconds remaining",
where remaining is WALL-CLOCK) as USER-role messages. Those diverge across a
prompt's generations, so the grouping key is unique per rollout -> every group
has size 1 -> the `valid_mask.sum() <= 1` branch sets baseline = reward ->
advantage = reward - reward = 0 for the WHOLE batch -> zero gradient. Confirmed on
a real trajectory dump: 12 episodes = 3 prompts x 4, but torch.unique saw 12
groups of size 1 (advantages min=max=std=0.0000 despite real reward variance).

THE FIX. Key grouping on the INITIAL prompt only -- the messages BEFORE the first
assistant turn. That prefix (system + task + initial listing) is identical across
a prompt's generations and distinct across prompts, so torch.unique recovers the
correct groups (verified: the same dump then grouped as 3 x 4). Everything after
the first assistant turn -- responses AND the divergent user-role env output -- is
excluded, which is exactly what GRPO grouping wants. Reward variance within the
group is preserved, so advantages become non-zero and the model actually learns.

Safe / minimal: only changes which messages form the GROUPING KEY. It does NOT
touch what is trained on (that is the flattened message_log with its loss mask,
built separately), the rewards, or the loss. An episode with no assistant turn
(degenerate) keeps its old behaviour (loop simply never breaks). Fail-loud if the
anchor is missing (upstream refactor) rather than silently no-op. Idempotent:
guarded on the HYPOTEST_GROUP_KEY_PATCH marker in nemo_rl_setup.sh.
"""

import sys

path = sys.argv[1]
src = open(path).read()

if "HYPOTEST_GROUP_KEY_PATCH" in src:
    print("[HYPOTEST_GROUP_KEY_PATCH] already applied; skipping")
    sys.exit(0)

# The inner loop of _extract_prompt_only_messages. Anchoring on all three lines
# (not just the `if`) keeps this unique within grpo.py.
anchor = (
    "        for message in message_log:\n"
    '            if message["role"] == "user" or message["role"] == "system":\n'
    "                prompt_only_log.append(message)\n"
)

if anchor not in src:
    # FAIL-OPEN (exit 0) so the pod is NOT blocked. The deployed NeMo RL may not
    # have this exact code (the anchor was written against the bbh-third-party
    # v0.6.0 copy, which is unverified against the actual image). Skipping lets the
    # run proceed so we can OBSERVE the advantages. The log line below tells us
    # which case we're in; combined with the "Advantages stats" line it is a clean
    # empirical test:
    #   patch APPLIED  + advantages non-zero -> grouping was the bug, now fixed
    #   patch APPLIED  + advantages still 0  -> fix targets the wrong thing; re-diagnose
    #   patch SKIPPED  (this branch)         -> deployed grpo.py differs from the
    #                                           reference; re-diagnose against it
    print(
        "[HYPOTEST_GROUP_KEY_PATCH] ANCHOR NOT FOUND -- the deployed grpo.py differs "
        "from the v0.6.0 reference. SKIPPING (grouping left unchanged, run proceeds). "
        "If 'Advantages stats' is still 0/0/0, the deployed code groups differently "
        "and we must re-diagnose against it.",
        flush=True,
    )
    sys.exit(0)

patched = (
    "        for message in message_log:\n"
    "            # HYPOTEST_GROUP_KEY_PATCH: stop at the first assistant turn.\n"
    "            # Everything after it (assistant responses AND this env's per-turn\n"
    "            # observations / env-state / wall-clock time warnings, all recorded\n"
    "            # as user-role) DIVERGES per rollout. Keeping it made torch.unique()\n"
    "            # see every generation as a distinct prompt -> groups of size 1 ->\n"
    "            # leave-one-out baseline == reward -> advantage == 0 (zero gradient).\n"
    "            # The initial prompt (messages before the first assistant turn) is\n"
    "            # identical across a prompt's generations, so it is the correct key.\n"
    '            if message["role"] == "assistant":\n'
    "                break\n"
    '            if message["role"] == "user" or message["role"] == "system":\n'
    "                prompt_only_log.append(message)\n"
)

src = src.replace(anchor, patched, 1)
open(path, "w").write(src)
print("[HYPOTEST_GROUP_KEY_PATCH] applied: GRPO grouping now keys on the initial prompt")
