#!/usr/bin/env python3
"""Fork a benchmarked trajectory at the rubric model's earliest ``first_wrong_step``.

The critic (rubric model) annotates each graded run's rubric criteria with a
per-criterion ``first_wrong_step`` — the 0-based index of the first *notebook
cell* where the agent went off-track for that criterion (derived in Python from
the judge's ``relevant_steps``; see ``judges.base.derive_first_wrong_step``).
This script takes the *earliest* such cell across all criteria and "forks" the
run: it keeps everything the agent did *before* that cell, then lets the policy
generate a fresh continuation from that point onward, injecting the feedback from
that driving criterion.

Each trajectory is forked exactly **once**. (An earlier revision chained forks —
round N+1 forking round N's notebook until the rollout earned full reward or ran
out of step budget. That machinery, and the judge-side anchoring that made it
converge, have been removed; see ``archive/`` for runs produced under it.)

How a fork works (replay-then-generate):

1. Find the earliest per-criterion ``first_wrong_step`` from the matching
   ``results/<id>/score_info.json`` (matched by submitted answer, reusing
   ``regrade.py``'s helpers — the ``iterN`` dir suffix does NOT line up with the
   ``repN`` trajectory order), or take it from ``--first-wrong-step``.
2. Map ``first_wrong_step`` (a *final-notebook* cell index) to a trajectory step
   ``K`` by simulating cell appends from the recorded ``run_cell`` arguments
   (``execute_and_add_cell`` appends when ``idx is None or idx >= len(cells)``;
   ``reset_kernel`` clears the notebook). ``K`` is the step that *first appends*
   that cell — we fork immediately before it.
3. Recreate a fresh env for the same problem (via ``Dataset.get_new_env_by_idx``,
   exactly like the server) and *replay* the parent's actions for steps ``0..K-1``,
   which faithfully rebuilds both the kernel state and the notebook.
4. From step ``K`` on, drive the *same policy model* (built from the benchmark
   config) live until it submits an answer (or hits ``max_steps``). Submitting
   triggers the rubric model again, producing a new score + new ``first_wrong_step``.
5. Save the fork as a one-element ``trajectories.pkl`` plus the env's own
   ``score_info.json`` (written on close), so it is itself a normal, inspectable,
   re-forkable run.

The step budget covers replay *and* generation: the fork gets a fresh env with
``max_steps = AGENT_MAX_STEPS`` (50, from ``.env``), and the replayed prefix
already advances ``env.step_count``. A fork at step 20 replays 20 and may
generate at most 30 more — so a fork point too near the end is skipped up front
rather than paying for a container that can do nothing (``no_room``).

A trajectory is skipped, not failed, when there is nothing to fork:
``no_wrong_step`` (no criterion flags a cell), ``cell_never_appended``, or
``no_room``. A fork that ran records how it ended in ``stop_reason``:
``full_reward`` (normalized score 1.0), ``truncated`` (hit max_steps without
submitting, so there is no new grade), or ``submitted``.

Output layout is **flat** — every fork is a top-level directory under
``--out-dir``, which is how ``scripts/fork/inspect_fork.py`` and
``scripts/fork/regrade_forks.py`` discover them::

    forks/task_0_rep0-fork_cell4/   trajectories.pkl  fork_info.json  <id>-iter0/score_info.json
    forks/task_1_rep2-fork_cell11/  …
    forks/fork_summary.json         rollup over every fork

Unlike ``regrade.py``, this script imports and drives the real env modules
(which require ``ray`` etc.), so run it from the project venv — NOT the bixbench
conda env: ``source .venv/bin/activate`` first.

By default this forks EVERY trajectory in ``trajectories.pkl`` (reusing the one
expensive policy build + score_info mapping across all of them). Pass
``--traj-id`` to fork just one.

Usage:
    source .venv/bin/activate

    # fork ALL trajectories in the pkl:
    python scripts/fork/fork_trajectory.py \
        --server-config server.yaml \
        --benchmark-config benchmark.yaml \
        --out-dir forks/

    # fork a single trajectory:
    python scripts/fork/fork_trajectory.py --traj-id task_0_rep0 ...

    # override the fork point (single trajectory only):
    python scripts/fork/fork_trajectory.py --traj-id task_0_rep0 --first-wrong-step 7 ...

This runs the real environment (kernel + capsule data) and makes live LLM calls
for both the policy and the rubric model, so run it with the same environment as
the dataset server (e.g. `set -a; source .env`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))  # for `import regrade`
sys.path.insert(0, str(ROOT / "src"))  # hypotest isn't installed in the env; import from source

import regrade  # noqa: E402 — reuse trajectory↔score_info matching helpers
from aviary.core import Message  # noqa: E402
from ldp.data_structures import Trajectory, Transition  # noqa: E402
from tqdm.asyncio import tqdm  # noqa: E402 — concurrent fork progress bar (same as benchmark_agent)

from hypotest.benchmark_agent import SimpleAgentConfig  # noqa: E402 — same policy as benchmark
from hypotest.dataset_server import Dataset, ServerConfig  # noqa: E402
from hypotest.env import config as env_cfg  # noqa: E402 — AGENT_MAX_STEPS, for the budget check


def action_value(step_or_op):
    """Unwrap an ldp ``OpResult`` (or step.action) to the underlying ToolRequestMessage."""
    return getattr(step_or_op, "value", step_or_op)


def tool_calls(action_val):
    """Yield ``(name, args_dict)`` for each tool call in a ToolRequestMessage."""
    for tc in getattr(action_val, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", "") or ""
        args = getattr(fn, "arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        yield name, (args if isinstance(args, dict) else {})


def fork_step_for_cell(action_vals: list, target_cell: int) -> int | None:
    """Find the trajectory step that *first appends* notebook cell ``target_cell``.

    Mirrors ``InterpreterEnvState.execute_and_add_cell`` cell bookkeeping without
    executing anything: a ``run_cell`` appends a new cell (incrementing the count)
    when ``idx`` is missing or ``>= current_count``, otherwise it edits in place;
    ``reset_kernel`` clears the notebook. Returns the step index whose append
    creates ``target_cell`` (i.e. the fork boundary ``K``), or ``None`` if that
    cell is never created.
    """
    count = 0
    for i, av in enumerate(action_vals):
        for name, args in tool_calls(av):
            if name == "run_cell":
                idx = args.get("idx")
                try:
                    idx = int(idx) if idx is not None else None
                except (ValueError, TypeError):
                    idx = None
                if idx is None or idx >= count:  # append
                    if count == target_cell:
                        return i
                    count += 1
                # else: edit in place, count unchanged
            elif name == "reset_kernel":
                count = 0
    return None


def select_fork_criterion(criteria: list) -> tuple[int | None, list[dict]]:
    """Pick the rubric criteria sharing the smallest (earliest) ``first_wrong_step``.

    Each criterion carries its own ``first_wrong_step`` (the cell where the procedure went
    wrong for that criterion specifically; ``null`` when the criterion got full marks). We
    fork at the earliest such cell across all criteria, and surface the feedback from *every*
    criterion that flagged that same earliest cell (e.g. if criteria 1 and 3 both went wrong
    at cell 4, both of their feedback notes are injected).

    Returns ``(first_wrong_step, criteria_at_step)`` or ``(None, [])`` if no criterion flags a
    wrong step at all.
    """
    candidates = [c for c in criteria if isinstance(c, dict) and isinstance(c.get("first_wrong_step"), int)]
    if not candidates:
        return None, []
    earliest = min(c["first_wrong_step"] for c in candidates)
    crits = [c for c in candidates if c["first_wrong_step"] == earliest]
    return earliest, crits


def criterion_feedback(crits: list[dict]) -> str | None:
    """Build a feedback note from every criterion driving the fork point."""
    notes = [fb for c in crits if (fb := (c.get("feedback") or "").strip())]
    if not notes:
        return None
    bullets = "\n".join(f"- {fb}" for fb in notes)
    return f"Guidance for how to proceed (from an evaluator):\n{bullets}\n\nTake this into account as you continue."


def idx_from_traj_id(traj_id: str) -> int:
    """Extract the problem index from a ``task_<idx>[_rep<r>]`` trajectory id."""
    m = re.search(r"task_(\d+)", traj_id)
    if not m:
        raise ValueError(f"Cannot parse problem index from traj_id {traj_id!r} (expected 'task_<idx>...').")
    return int(m.group(1))


class SkipFork(Exception):
    """Raised when a trajectory has nothing forkable — skip it and keep going."""


async def fork_round(
    *,
    parent_traj: Trajectory,
    root_id: str,
    problem_idx: int,
    fork_cell: int,
    fork_k: int,
    feedback: str | None,
    old_score: float | None,
    old_evaluation: dict | None,
    args,
    scfg,
    agent,
) -> dict:
    """Replay ``parent_traj``'s steps ``0..fork_k-1``, then generate live. Returns the summary.

    Shared, expensive resources (parsed configs and the policy ``agent``) are built once by
    ``main`` and threaded in, so a whole batch of forks only pays for them once.
    """
    parent_actions = [action_value(s.action) for s in parent_traj.steps]
    round_id = f"{root_id}-fork_cell{fork_cell}"
    print(
        f"\n=== {root_id}: fork at cell {fork_cell} -> before step {fork_k} "
        f"| replay {fork_k}, generate up to {env_cfg.AGENT_MAX_STEPS - fork_k} ==="
    )
    if feedback:
        print(f"Injecting rubric feedback at fork point:\n{feedback}")
    else:
        print("No criterion-specific rubric feedback available; forking without feedback.")

    # ── recreate the env for the same problem (server-identical wiring) ──────
    # A fresh Dataset per fork keeps each fork's work_dir/save_dir isolated and resets the
    # problem_counter so run_id is a predictable `<id>-iter0`. model_copy() so concurrent
    # forks don't stomp each other's work_dir/save_dir on the shared scfg.dataset object.
    dcfg = scfg.dataset.model_copy()
    out_dir = args.out_dir / round_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dcfg.work_dir = (out_dir / "work").resolve()
    dcfg.save_dir = out_dir.resolve()
    dcfg.work_dir.mkdir(parents=True, exist_ok=True)

    dataset = Dataset(dcfg)
    if problem_idx >= len(dataset):
        raise SkipFork(f"{root_id}: problem idx {problem_idx} out of range (dataset has {len(dataset)} problems)")
    env = dataset.get_new_env_by_idx(problem_idx)

    transitions: list[Transition] = []
    try:
        obs, _tools = await env.reset()
        state = await agent.init_state(env.tools)
        done = False

        # ── phase 1: replay the parent's actions for steps 0..K-1 ────────────
        for i in range(fork_k):
            av = parent_actions[i]
            next_state = state.get_next_state(obs)
            next_state.messages = [*next_state.messages, av]  # force the parent's action
            next_obs, reward, done, trunc = await env.step(av)
            transitions.append(
                Transition(
                    timestep=i,
                    agent_state=state,
                    next_agent_state=next_state,
                    observation=obs,
                    next_observation=next_obs,
                    action=parent_traj.steps[i].action,
                    reward=reward,
                    done=done,
                    truncated=trunc,
                    value=0.0,
                )
            )
            state, obs = next_state, next_obs
            if done:
                print(f"WARNING: replay terminated early at step {i} (done=True before fork point).")
                break

        # ── inject this round's rubric feedback, just before the fork generation ──
        # Only this round's — earlier rounds' guidance is deliberately not replayed.
        if feedback and not done:
            obs = [*obs, Message(content=feedback)]

        # ── phase 2: live generation from the fork point onward ──────────────
        # env.step_count is already at fork_k from the replay, so this loop is what enforces
        # "50 total per round, replay included".
        i = fork_k
        while not done and env.step_count < env.max_steps:
            action_op, next_state, value = await agent.get_asv(state, obs)
            av = action_value(action_op)
            next_obs, reward, done, trunc = await env.step(av)
            transitions.append(
                Transition(
                    timestep=i,
                    agent_state=state,
                    next_agent_state=next_state,
                    observation=obs,
                    next_observation=next_obs,
                    action=action_op,
                    reward=reward,
                    done=done,
                    truncated=trunc,
                    value=value,
                )
            )
            state, obs = next_state, next_obs
            i += 1

        new_score = env.state.score
        new_criteria: list[dict] = env.state.score_metadata.get("criteria") or []
        new_answer = env.state.answer
        submitted = env.state.done
    finally:
        await env.close()  # writes notebook + score_info.json into save_dir

    new_fws, _ = select_fork_criterion(new_criteria)
    fork_traj = Trajectory(traj_id=round_id, steps=transitions)
    (out_dir / "trajectories.pkl").write_bytes(pickle.dumps([fork_traj]))

    # How the fork ended. `truncated` means it never submitted, so there is no new grade at all
    # — distinct from submitting and scoring below full marks.
    if not submitted:
        stop_reason = "truncated"
    elif new_score >= 1.0:
        stop_reason = "full_reward"
    else:
        stop_reason = "submitted"

    summary = {
        "source_traj_id": root_id,
        "traj_id": round_id,
        "problem_idx": problem_idx,
        "out_dir": str(out_dir),
        "fork_cell": fork_cell,
        "fork_step": fork_k,
        "n_steps_total": len(transitions),
        "n_replayed": fork_k,
        "n_generated": len(transitions) - fork_k,
        "steps_available": env_cfg.AGENT_MAX_STEPS - fork_k,
        "submitted": submitted,
        "old_first_wrong_step": fork_cell,
        "new_first_wrong_step": new_fws,
        "old_score": old_score,
        "new_score": new_score,
        "injected_feedback": feedback,
        "new_answer": new_answer,
        "old_evaluation": old_evaluation,
        "stop_reason": stop_reason,
    }
    (out_dir / "fork_info.json").write_text(json.dumps(summary, indent=2))
    print(
        f"── {round_id} done | score {old_score} -> {new_score} ({stop_reason}) "
        f"| new first_wrong_step cell {new_fws} | {fork_k}+{len(transitions) - fork_k} steps ──"
    )
    return summary


async def fork_one(traj, *, args, scfg, agent, mapping) -> dict:
    """Fork ``traj`` once, at the earliest ``first_wrong_step`` its grade reports.

    The fork point comes from the original benchmark run's ``score_info.json`` (matched by
    answer via ``regrade.build_mapping``) or from ``--first-wrong-step``.

    Raises ``SkipFork`` when there is nothing to fork — no matching grade, no criterion
    flagging a wrong step, a cell the trajectory never appended, or a fork point so late that
    no generation budget is left. Returns the fork's summary otherwise.
    """
    root_id = traj.traj_id
    problem_idx = idx_from_traj_id(root_id)

    # ── locate the fork point ────────────────────────────────────────────────
    old_score: float | None = None
    old_evaluation: dict | None = None
    fork_cell: int | None = args.first_wrong_step
    crits: list[dict] = []  # explicit override: no criterion to source feedback from
    if fork_cell is None:
        match = next((m for m in mapping if m[0].traj_id == root_id), None)
        if match is None:
            raise SkipFork(f"{root_id}: no matching score_info.json under {args.results}")
        _, run_name, info, ratio = match
        if info is None:
            raise SkipFork(f"{root_id}: matched run {run_name} has no score_info to fork from")
        if ratio < args.min_match:
            print(f"WARNING: low answer match ({ratio:.3f}) to {run_name} for {root_id}; consider --first-wrong-step.")
        criteria = info.get("criteria") or []
        old_score = info.get("score")
        # Persist the parent's original benchmark evaluation (from ./results) so the fork
        # carries the pre-fork notebook's full judgment, not just old_score.
        old_evaluation = {
            "run_id": run_name,
            "score": info.get("score"),
            "raw_score": info.get("raw_score"),
            "max_score": info.get("max_score"),
            "criteria": criteria,
        }
        fork_cell, crits = select_fork_criterion(criteria)
        if fork_cell is None:
            raise SkipFork(f"{root_id}: no criterion with a first_wrong_step in {run_name}/score_info.json")

    # ── map the cell to the trajectory step that first appends it ────────────
    fork_k = fork_step_for_cell([action_value(s.action) for s in traj.steps], fork_cell)
    if fork_k is None:
        raise SkipFork(f"{root_id}: cell {fork_cell} is never appended by this trajectory")
    # Budget check, before paying for a container: replay counts against max_steps, so a late
    # fork point leaves no room to generate anything.
    if fork_k >= env_cfg.AGENT_MAX_STEPS - 1:
        raise SkipFork(
            f"{root_id}: fork at step {fork_k} leaves no room to generate (AGENT_MAX_STEPS={env_cfg.AGENT_MAX_STEPS})"
        )

    return await fork_round(
        parent_traj=traj,
        root_id=root_id,
        problem_idx=problem_idx,
        fork_cell=fork_cell,
        fork_k=fork_k,
        feedback=criterion_feedback(crits),
        old_score=old_score,
        old_evaluation=old_evaluation,
        args=args,
        scfg=scfg,
        agent=agent,
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--traj-id",
        default=None,
        help="Fork only this trajectory (e.g. task_0_rep0). Omit to fork ALL trajectories in --pkl.",
    )
    ap.add_argument("--server-config", type=Path, default=ROOT / "server.yaml", help="Dataset/server config yaml")
    ap.add_argument(
        "--benchmark-config", type=Path, default=ROOT / "benchmark.yaml", help="Benchmark config (for agent_config)"
    )
    ap.add_argument("--pkl", type=Path, default=ROOT / "benchmark_results/trajectories.pkl")
    ap.add_argument("--results", type=Path, default=ROOT / "results", help="Dir of <id>/score_info.json from the run")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "forks", help="Where to write the forked run(s)")
    ap.add_argument(
        "--first-wrong-step",
        type=int,
        default=None,
        help="Override the fork cell (default: read from matched score_info.json). Only valid with a single --traj-id.",
    )
    ap.add_argument("--min-match", type=float, default=0.95, help="Warn if best answer match ratio is below this")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Resume: skip any trajectory that already has a completed fork "
        "(a forks/<traj_id>-fork_cell*/fork_info.json). Lets you re-run after an interruption "
        "without re-forking or overwriting finished forks.",
    )
    ap.add_argument(
        "--num-parallel",
        type=int,
        default=8,
        help="Max trajectories to fork concurrently (asyncio.Semaphore). Kept below the benchmark's "
        "num_parallel since each fork hits both the policy and rubric models on a single vLLM pod.",
    )
    args = ap.parse_args()

    if args.first_wrong_step is not None and args.traj_id is None:
        raise SystemExit("--first-wrong-step only makes sense with a single --traj-id (one cell index per trajectory).")

    # ── select the trajectories to fork ──────────────────────────────────────
    trajs = pickle.loads(args.pkl.read_bytes())
    if args.traj_id is not None:
        selected = [t for t in trajs if t.traj_id == args.traj_id]
        if not selected:
            raise SystemExit(f"traj_id {args.traj_id!r} not found in {args.pkl} (have: {[t.traj_id for t in trajs]})")
    else:
        selected = list(trajs)
        print(f"Forking ALL {len(selected)} trajectories in {args.pkl}: {[t.traj_id for t in selected]}")

    # Resume: drop trajectories that already have a completed fork on disk. The fork cell is
    # only known after the grade is read, so match the directory by glob rather than by name.
    if args.skip_existing:
        before = len(selected)
        selected = [t for t in selected if not any(args.out_dir.glob(f"{t.traj_id}-fork_cell*/fork_info.json"))]
        print(f"--skip-existing: {before - len(selected)} already forked, {len(selected)} remaining.")

    # ── build shared resources once (configs, policy, answer→score_info map) ──
    scfg = ServerConfig.model_validate(yaml.safe_load(args.server_config.read_text()))
    bench_data = yaml.safe_load(args.benchmark_config.read_text())
    agent = SimpleAgentConfig.model_validate(bench_data["agent_config"]).construct_agent()
    mapping = [] if args.first_wrong_step is not None else regrade.build_mapping(trajs, args.results)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── fork concurrently (semaphore + gather, like benchmark_agent) ─────────
    # The semaphore caps how many forks run at once; each fork's failure is caught here so
    # one bad trajectory never aborts the rest of the batch.
    semaphore = asyncio.Semaphore(args.num_parallel)

    async def run_one(traj) -> tuple[str, str, dict | str]:
        async with semaphore:
            try:
                summary = await fork_one(traj, args=args, scfg=scfg, agent=agent, mapping=mapping)
                return ("forked", traj.traj_id, summary)
            except SkipFork as e:
                print(f"SKIP: {e}")
                return ("skipped", traj.traj_id, str(e))
            except Exception as e:
                print(f"ERROR forking {traj.traj_id}: {e}")
                return ("failed", traj.traj_id, str(e))

    print(
        f"Forking {len(selected)} trajectories with up to {args.num_parallel} concurrent forks. "
        f"Step budget per fork: {env_cfg.AGENT_MAX_STEPS} (replay included)."
    )
    outcomes = await tqdm.gather(*[run_one(t) for t in selected], ncols=0, desc="Forking")

    forked: list[dict] = [r for kind, _tid, r in outcomes if kind == "forked" and isinstance(r, dict)]
    skipped: list[tuple[str, str]] = [(tid, str(r)) for kind, tid, r in outcomes if kind == "skipped"]
    failed: list[tuple[str, str]] = [(tid, str(r)) for kind, tid, r in outcomes if kind == "failed"]

    # ── batch summary (printed + written as forks/fork_summary.json) ─────────
    print("\n" + "=" * 70)
    print(
        f"FORK SUMMARY: {len(forked)} forked, {len(skipped)} skipped, "
        f"{len(failed)} failed (of {len(selected)} selected)"
    )
    for r in forked:
        print(
            f"  {r['source_traj_id']:<24} cell {r['fork_cell']:<4} "
            f"score {str(r['old_score']) + ' -> ' + str(r['new_score']):<14} "
            f"next fws {r['new_first_wrong_step']!s:<6} "
            f"({r['n_replayed']}+{r['n_generated']} steps) {r['stop_reason']}"
        )
    for tid, why in skipped:
        print(f"  SKIP {tid}: {why}")
    for tid, why in failed:
        print(f"  FAIL {tid}: {why}")

    (args.out_dir / "fork_summary.json").write_text(
        json.dumps({"forked": forked, "skipped": skipped, "failed": failed}, indent=2)
    )
    print(f"\nWrote rollup -> {args.out_dir / 'fork_summary.json'}")


if __name__ == "__main__":
    asyncio.run(main())
