# biomed-agents

**Trajectory forking for agent evaluation.** Take an agent run that a rubric
judge already graded, rewind it to the earliest step the judge marked wrong,
hand the agent that judge's feedback, and let it continue from there. The change
in score measures how much of the failure was recoverable at that moment —
turning a rubric from a scoreboard into an intervention.

Built on [EdisonScientific/hypotest](https://github.com/EdisonScientific/hypotest)
(Apache-2.0), which provides the Jupyter-kernel execution environment, dataset
server, and benchmark agent. This is an independent fork, not affiliated with or
endorsed by Edison Scientific. See [`NOTICE`](NOTICE) for what was added and
[`docs/patches.md`](docs/patches.md) for the change-by-change record.

---

## How a fork works

```
parent run   step 0 ── 1 ── 2 ── 3 ── 4 ── 5 ── 6 ── submit ──► graded 0.33
                                 │
                    judge: "cell 3 is the earliest thing that is wrong,
                            here is what to do instead"
                                 │
                                 ▼
fork         step 0 ── 1 ── 2 ──╳
             └── replayed verbatim ──┘   └── generated fresh ──► graded 0.83
                (real kernel, real          (feedback injected
                 notebook state)             as an observation)
```

1. **Locate the fork point.** The judge returns, per rubric criterion, which
   steps it looked at and whether each was correct. `derive_first_wrong_step`
   ([`src/hypotest/env/judges/base.py`](src/hypotest/env/judges/base.py))
   computes the earliest incorrect step **in Python, not in the LLM** — the
   model is never asked to name a number it can be inconsistent about. A
   criterion already at full marks contributes nothing.

2. **Replay.** The environment is rebuilt from scratch and the parent's tool
   calls are re-executed verbatim up to that cell. This runs real code in a real
   kernel, so both kernel state and notebook history are genuinely reconstructed
   rather than simulated.

3. **Inject and generate.** The judge's forward-looking feedback for the
   criteria at that cell is appended as an observation, and the agent takes over
   from there under its normal policy.

Because the fork is graded by the same judge and saved in the same layout as any
other run, **a fork is itself a normal run** — inspectable, re-gradable, and
forkable again.

### The all-feedback ablation

`fork_trajectory_allfb.py` is a second arm that changes exactly one variable: it
injects the feedback from _every_ criterion that lost points, not just the ones
at the fork cell. The fork cell itself is computed identically, so the two arms
are comparable cell-for-cell. It rebinds two functions on the baseline module
rather than copying it, which is what keeps them from drifting apart.

---

## Install

Python 3.11–3.13, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Copy the environment template and fill it in:

```bash
cp .env.example .env
```

> **The judge is an LLM, and a missing API key does not raise.** Every episode
> silently scores 0.0, which looks exactly like a model that cannot do the task.
> If all your scores are zero, check `.env` first. `litellm` auto-loads `.env`
> from the working directory, so run commands from the repo root.

Optionally build the sandboxed kernel image (a long Ubuntu + conda + R +
bioinformatics build; the default local-kernel path works without it):

```bash
make image
```

## Get the task data

Each task is a _capsule_ — a folder of input data plus a hypothesis and rubric:

```bash
hf sync hf://buckets/EdisonScientific/bixbench-hypothesis-capsules capsules/hypotest/
```

Point `capsule_dir` at the directory that **directly contains** the
`CapsuleData-<uuid>/` folders.

---

## Running it

Forking consumes a completed benchmark run, so it is a two-stage flow.

### Stage 1 — run a benchmark

```bash
cp server.example.yaml server.yaml            # edit paths and rubric_model
cp benchmark.example.yaml benchmark.yaml      # edit the policy model

hypotest-server server.yaml                   # terminal 1
hypotest-benchmark benchmark.yaml             # terminal 2
```

The `hypotest-*` commands are console scripts created by `uv sync`. Equivalent
direct invocations, if you prefer not to install the package:

```bash
make server CONFIG=server.yaml
uv run python src/hypotest/benchmark_agent.py benchmark.yaml
```

This writes `rewards.json` and `trajectories.pkl` into `results_dir`, and one
`<uuid>-iter<N>/score_info.json` per rollout into the server's `save_dir`. Those
are the three inputs a fork needs.

Use `num_replications: 3` or more. Forking is most informative when you can ask
whether a task that never passed in k tries becomes reachable with feedback.

### Stage 2 — fork it

```bash
python scripts/fork/fork_trajectory.py \
    --server-config    server.yaml \
    --benchmark-config benchmark.yaml \
    --pkl              benchmark_results/trajectories.pkl \
    --results          results/ \
    --out-dir          forks/ \
    --num-parallel     8
```

Add `--traj-id task_0_rep0` to fork a single trajectory, `--skip-existing` to
resume an interrupted batch, and `--first-wrong-step N` to override the fork
cell for one trajectory.

> **The server config must match the run you are forking** — especially
> `include_protocol` and `capsule_dir`. Replay re-executes the parent's actions,
> so a different prompt or a different capsule set means you are replaying them
> against a different task.

Each fork writes a `fork_info.json` recording `fork_cell`, `fork_step`,
`n_replayed`, `n_generated`, `old_score`, `new_score`, `injected_feedback`,
`old_first_wrong_step`, `new_first_wrong_step` and `stop_reason`, plus a
`fork_summary.json` rollup of forked / skipped / failed.

A trajectory is **skipped**, not failed, when there is nothing to fork: no
criterion has a first wrong step, the target cell is never appended by that
trajectory, or `no_room` — the fork point is so late that no step budget remains.

> **Replayed steps count against `AGENT_MAX_STEPS`.** A fork at step 20 with
> `AGENT_MAX_STEPS=50` gets 30 steps to generate, not 50.

### Stage 3 — read the results

```bash
python scripts/fork/inspect_fork.py forks/ --html forks.html

python scripts/fork/compare_rewards.py \
    --rewards       benchmark_results/rewards.json \
    --fork-summary  forks/fork_summary.json
```

`inspect_fork.py` renders the whole batch as one self-contained HTML page —
score deltas, which prefix was frozen, the feedback that was injected, and the
regenerated answer. `compare_rewards.py` prints the pass@k / avg@k change.

To re-grade with a different judge without re-running any agent, use
`scripts/eval/regrade.py` (whole run) or `scripts/fork/regrade_forks.py` (forks,
optionally with the parent's evaluation supplied as reference context).

---

## Repo map

| Path                              | What it is                                                                                                                                    |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/hypotest/env/`               | Execution environment: Jupyter kernel lifecycle, notebook tools, filesystem tools                                                             |
| `src/hypotest/env/judges/`        | Per-benchmark grading. One module each, owning its prompt, schema and scoring. Adding a benchmark means adding a judge, never editing the env |
| `src/hypotest/dataset_server.py`  | Serves one environment per rollout over HTTP                                                                                                  |
| `src/hypotest/benchmark_agent.py` | Benchmark client; writes `rewards.json` + `trajectories.pkl`                                                                                  |
| `scripts/fork/`                   | **The fork pipeline** — fork, ablation arm, inspect, re-grade, compare                                                                        |
| `scripts/eval/`                   | Judge-agnostic evaluation: re-grading, trajectory and judge-diff viewers                                                                      |
| `scripts/data/`                   | Converters turning each benchmark into capsules + `ProblemInstance` jsonl                                                                     |
| `proposer/`                       | Hypothesis-proposer subsystem: generate candidate hypotheses, then A/B whether showing the source paper helps                                 |
| `rl/`                             | GRPO training stack — **incomplete, see below**                                                                                               |
| `examples/cluster/`               | The run:ai orchestration we used. Site-specific reference, not a supported entry point                                                        |
| `docs/patches.md`                 | Every local change to the upstream environment, with rationale and revert instructions                                                        |

### Supported benchmarks

BixBench, BioMysteryBench, HeurekaBench, BiomniBench-DA and bioagent-bench, via
`scripts/data/convert_*.py` and the matching judge in `src/hypotest/env/judges/`.
Grading is rubric-based by default; some judges score deterministically against
a ground-truth answer key with no LLM call at all.

---

## Status of `rl/` — incomplete

`rl/` holds a GRPO training stack (NeMo RL + NeMo Gym) that trains a policy
against this environment. **It is published as a record of work in progress, not
as something you can run.** Known gaps:

- **It does not learn yet.** The furthest run — Qwen3.6-27B with LoRA r32/α32 on
  8×H100 — was stopped at step 51 with held-out eval flat at ~0.38. The leading
  suspect is a weak effective update (α/r = 1.0 at lr 1e-5) compounded by
  episodes being dropped from the gradient. The next thing to try is raising
  LoRA α.
- **A required dependency is not included.** The NeMo Gym tree is copied from a
  local path (`GYM_SRC`) that is not part of this repo and not publicly
  distributed.
- **It targets one specific cluster.** `rl/runai/` submits Run:ai CRDs against
  EPFL RCP namespaces, PVCs and an internal registry, and
  `rl/scripts/make_rl_runtime.sh` exists solely to work around one NFS export's
  `root_squash` behaviour.
- Expect to need 5–8×H100 and NGC credentials.

`rl/docker/Dockerfile.env` is the most reusable piece: it layers the environment
server onto the interpreter image and works anywhere you have Kubernetes.

Secrets live in `rl/runai/submit.env`, which is gitignored;
`rl/runai/submit.env.example` is the template.

---

## Development

```bash
make test     # pytest -n auto
make lint     # pre-commit hooks + mypy
```

Tests that need a paid API key or network access skip themselves rather than
failing, so a fresh clone with no credentials runs green. Tests needing Docker,
an R kernel, or capsule data skip when those are absent.

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
