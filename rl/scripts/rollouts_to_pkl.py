#!/usr/bin/env python
"""Convert NeMo Gym rollouts.jsonl -> trajectories.pkl, in the benchmark's format.

WHY THIS EXISTS
---------------
The two rollout paths in this repo emit different things:

    benchmark_agent.py:240   ldp RolloutManager -> tuple[Trajectory] -> pickle
    ng_collect_rollouts      NeMo Gym -> OpenAI Responses-API items -> JSONL

The GRPO training path uses the second one, so any measurement of *training*
transcript length comes out as JSONL and cannot be fed to analysis code written
against the benchmark's pickles. This script reassembles the JSONL into the same
``tuple[Trajectory]`` object that ``benchmark_agent.py`` pickles, so both can be
compared with one set of tooling.

    .venv/bin/python rl/scripts/rollouts_to_pkl.py \
        rl/results/smoke/rollouts.jsonl \
        -o rl/results/smoke/trajectories.pkl

THE MAPPING, STATED EXPLICITLY
------------------------------
NeMo Gym returns one flat, ordered item stream per rollout::

    reasoning, function_call, function_call_output, reasoning, function_call, ...

ldp stores INCREMENTS, not accumulated history -- verified against a real
benchmark pickle, where ``observation[0]`` is 3 Messages and every later
``observation`` is a single ``ToolResponseMessage``. One step is cut at each
``function_call``:

    observation       what the environment handed the policy this step
                      (step 0: the seed; step i: next_observation[i-1])
    action            OpResult -> ToolRequestMessage, reasoning folded into
                      .content alongside the tool call
    next_observation  ONLY the messages the environment returned this step

That preserves the invariant the benchmark analysis relies on --
``next_observation[i] == observation[i+1]`` -- so de-duplicating by "count each
unique message once" gives the same answer on both sources.

REWARD is terminal-only, matching ``benchmark_agent.py:224``
(``trajectory.steps[-1].reward``): every step gets 0.0 except the last.

THE SEED IS MISSING FROM THE SOURCE DATA
----------------------------------------
``ng_collect_rollouts`` does NOT persist the system prompt, task description or
initial file listing: in the JSONL both ``response.input`` and
``responses_create_params.input`` are empty, and the tool schema (~3 kB) is not
recorded as context either. The benchmark's pickles DO carry them --
``observation[0]`` there is 3 Messages.

So ``observation[0]`` here is empty, and any length computed from this file (or
from analyze_rollouts.py, which reads the same empty fields) UNDER-states the
real context by the seed -- about 5,600-7,000 tokens on measured benchmark
trajectories. Backfill it from the env server if you need like-for-like totals.

FIDELITY CAVEATS -- read before trusting a comparison
-----------------------------------------------------
* ``agent_state``/``next_agent_state``/``value`` have no equivalent in the Gym
  stream and are left None/0.0. The benchmark's own agent may populate them.
* ``action`` is a real ``OpResult`` wrapping the ``ToolRequestMessage``, but its
  ``CallID`` is synthetic -- there was no compute graph, so it cannot be traced
  back to an op run. Fine for reading ``.value``; not for graph analysis.
* Messages are reconstructed as aviary ``Message``/``ToolRequestMessage``/
  ``ToolResponseMessage``. Field-for-field identity with an ldp-native rollout
  is NOT claimed; token counts over ``content`` are the intended use.
* Reasoning is folded into the assistant ``ToolRequestMessage.content`` rather
  than kept as a separate item, matching the benchmark's one-string-per-turn
  shape. If you need them split, re-parse on ``</think>``.

Run with --verify to print a per-trajectory reconciliation against the source.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import uuid
from pathlib import Path
from typing import Any

try:
    from aviary.core import Message, ToolCall, ToolRequestMessage, ToolResponseMessage
    from ldp.data_structures import Trajectory, Transition
    from ldp.graph import OpResult
    from ldp.graph.op_utils import CallID
except ImportError as exc:  # pragma: no cover
    sys.exit(f"need aviary + ldp in this interpreter: {exc}\nTry .venv/bin/python")


def _text(item: dict[str, Any]) -> str:
    """Pull display text out of a Responses-API item, whatever shape it uses."""
    for key in ("text", "content", "summary", "output", "arguments"):
        v = item.get(key)
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            parts = []
            for e in v:
                if isinstance(e, str):
                    parts.append(e)
                elif isinstance(e, dict):
                    for k2 in ("text", "content", "output"):
                        if isinstance(e.get(k2), str):
                            parts.append(e[k2])
                            break
            if parts:
                return "".join(parts)
    return ""


def build_trajectory(rec: dict[str, Any], traj_id: str) -> Trajectory:
    params, resp = rec.get("responses_create_params", {}), rec.get("response", {})

    # Seed: whatever the policy saw before its first token.
    obs: list[Message] = []
    if isinstance(params.get("instructions"), str) and params["instructions"]:
        obs.append(Message(role="system", content=params["instructions"]))
    obs.extend(
        Message(role=m.get("role", "user"), content=_text(m) or "")
        for m in params.get("input") or []
        if isinstance(m, dict)
    )

    run_id = uuid.uuid4()
    steps: list[Transition] = []
    pending: list[str] = []  # this step's reasoning/visible text, not yet closed
    timestep = 0
    # ldp stores INCREMENTS, not accumulated history:
    #   observation[0]   = the seed (system + task + listing)
    #   observation[i]   = next_observation[i-1]
    #   next_observation = only the messages the environment returned this step
    # Verified against a real benchmark pickle: obs[i] == next_obs[i-1] for all i.
    # `incoming` is what this step observed; `produced` is what came back.
    incoming: list[Message] = list(obs)
    produced: list[Message] = []

    for item in resp.get("output") or []:
        itype = item.get("type")

        # Reasoning and visible text are folded into the SAME assistant message
        # as the tool call, not emitted separately. That matches how the
        # benchmark's data is shaped -- one continuous `content` string holding
        # the reasoning, then </think>, then the response -- so counting
        # `content` gives the same total on both sources, with no double count.
        if itype in {"reasoning", "message"}:
            t = _text(item)
            if t:
                pending.append(t)

        elif itype == "function_call":
            call = ToolCall.from_name(item.get("name", "unknown"), arguments=item.get("arguments", ""))
            # content is reasoning + visible text ONLY. Do NOT append _text(item)
            # here: for a function_call item that helper falls through to the
            # "arguments" key and returns the code, which already lives in
            # `tool_calls` and is counted there by any sane token counter. Doing
            # both double-counts every code cell -- it inflated measured action
            # tokens by ~2x on 2026-08-07 and made the agent look far more
            # verbose than the archived benchmark trajectories.
            action_msg = ToolRequestMessage(
                content="".join(pending),
                tool_calls=[call],
            )
            steps.append(
                Transition(
                    timestep=timestep,
                    agent_state=None,
                    next_agent_state=None,
                    observation=list(incoming),
                    action=OpResult(
                        call_id=CallID(run_id=run_id, fwd_id=uuid.uuid4()),
                        op_name="nemo_gym_replay",
                        op_class_name="NemoGymReplay",
                        value=action_msg,
                    ),
                    next_observation=[],  # filled by the function_call_output below
                    reward=0.0,
                    done=False,
                    truncated=False,
                    metadata={"call_id": item.get("call_id")},
                )
            )
            timestep += 1
            pending = []
            produced = []

        elif itype == "function_call_output":
            tool_msg = ToolResponseMessage(
                content=_text(item),
                name=item.get("name") or "tool",
                tool_call_id=item.get("call_id", "") or "",
            )
            produced.append(tool_msg)
            if steps:  # the reply belongs to the step that requested it
                steps[-1].next_observation = list(produced)
                incoming = list(produced)  # becomes the next step's observation

    if pending and steps:  # trailing assistant text with no tool call
        steps[-1].next_observation = [
            *list(steps[-1].next_observation),
            Message(role="assistant", content="".join(pending)),
        ]

    if steps:  # terminal-only reward, as benchmark_agent.py:224 assumes
        steps[-1].reward = float(rec.get("reward", 0.0))
        steps[-1].done = True

    return Trajectory(
        traj_id=traj_id,
        steps=steps,
        metadata={
            "task_idx": rec.get("task_idx"),
            "ng_task_index": rec.get("_ng_task_index"),
            "ng_rollout_index": rec.get("_ng_rollout_index"),
            "source": "ng_collect_rollouts",
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    out = args.out or args.rollouts.with_name("trajectories.pkl")
    trajectories = []
    for line in args.rollouts.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ti, ri = rec.get("_ng_task_index", 0), rec.get("_ng_rollout_index", 0)
        trajectories.append(build_trajectory(rec, f"task_{ti}_rep{ri}"))

    # tuple, not list -- benchmark_agent.py pickles the tuple from zip(*results)
    out.write_bytes(pickle.dumps(tuple(trajectories)))

    print(f"wrote {out}  ({len(trajectories)} trajectories, {out.stat().st_size:,} bytes)")
    print(f"{'traj_id':<20} {'steps':>6} {'reward':>7} {'msgs':>6}")
    for t in trajectories:
        msgs = len(t.steps[0].observation) + sum(len(st.next_observation) for st in t.steps) if t.steps else 0
        print(f"{t.traj_id:<20} {len(t.steps):>6} {t.steps[-1].reward if t.steps else 0:>7.3f} {msgs:>6}")

    if args.verify:
        print("\nreconciliation vs source (unique-message char totals):")
        for line, t in zip([x for x in args.rollouts.read_text().splitlines() if x.strip()], trajectories, strict=True):
            rec = json.loads(line)
            src = sum(len(_text(i)) for i in (rec["response"].get("output") or []))
            # NB: `msgs` above is a COUNT; this is the message list. Different name.
            msg_list = (t.steps[0].observation if t.steps else []) + [m for st in t.steps for m in st.next_observation]
            got = sum(len(str(m.content or "")) for m in msg_list) + sum(
                len(str(st.action.value.content or "")) for st in t.steps if st.action is not None
            )
            flag = "" if abs(got - src) <= 0.15 * max(src, 1) else "   <-- CHECK"
            print(f"  {t.traj_id:<20} source={src:>9,}  rebuilt={got:>9,}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
