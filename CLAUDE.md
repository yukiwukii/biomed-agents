# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a fork of [EdisonScientific/hypotest](https://github.com/EdisonScientific/hypotest)
whose focus is **trajectory forking**: rewinding a graded agent run to the earliest step a
rubric judge marked wrong, injecting that judge's feedback, and re-rolling from there. The
upstream project supplies the Jupyter-kernel execution environment, dataset server and
benchmark agent; the fork pipeline, the judge framework and the benchmark converters are
local additions. Not affiliated with Edison Scientific.

Read `README.md` for the user-facing flow and `docs/patches.md` for the change-by-change
record of how this diverges from upstream. `# PATCH N` markers in the source refer to it.

## Commands

```bash
uv sync                                   # install

make test                                 # pytest -n auto
pytest tests/test_foo.py::test_specific   # single test
make lint                                 # pre-commit hooks + mypy

make image                                # build the sandboxed kernel image
make server CONFIG=server.yaml            # run the dataset server
uv run python src/hypotest/benchmark_agent.py benchmark.yaml
```

After `uv sync` the console scripts `hypotest-server` and `hypotest-benchmark` are also
available.

## Architecture

```
src/hypotest/
├── dataset_server.py   # TaskDatasetServer serving one InterpreterEnv per rollout,
│                       #   plus the idle-environment sweeper
├── benchmark_agent.py  # Benchmark client (ldp RolloutManager); writes rewards.json
│                       #   + trajectories.pkl
└── env/
    ├── config.py           # ExecutionConfig; env knobs (STRIP_IMAGES, NB_OUTPUT_LIMIT,
    │                       #   AGENT_MAX_STEPS, KERNEL_ENV_PATH)
    ├── interpreter.py      # Interpreter: Jupyter kernel lifecycle & code execution
    ├── interpreter_env.py  # InterpreterEnv: the aviary Environment, tools, scoring
    ├── notebook_env.py     # Notebook-oriented environment surface
    ├── kernel_server.py    # Kernel server management; NBLanguage
    ├── problem.py          # ProblemInstance
    ├── prompts.py          # System prompts, software-stack capabilities, judge contract
    ├── code_safety.py      # Guards on executed code
    ├── judges/             # ONE MODULE PER BENCHMARK. Each owns its prompt, response
    │   │                   #   schema and label→points mapping. base.py holds the shared
    │   │                   #   scoring path and derive_first_wrong_step (the fork signal).
    │   ├── base.py         #   Adding a benchmark = adding a judge, never editing the env.
    │   ├── hypotest_judge.py, bixbench.py, biomystery.py, heureka.py,
    │   └── biomni.py, bioagent.py
    ├── tools/filesystem.py # File I/O tools with format support
    └── utils/              # core.py (extraction), img_utils, notebook_utils, workspace_utils

scripts/
├── fork/   fork_trajectory.py, fork_trajectory_allfb.py (ablation arm),
│           inspect_fork.py, regrade_forks.py, compare_rewards.py
├── eval/   regrade.py, inspect_trajectory.py, inspect_judge_diff.py,
│           biomni_judge.py, map_trajs.py
└── data/   convert_*.py for five benchmarks, capsule staging/download

proposer/            Hypothesis-proposer subsystem and its A/B study
rl/                  GRPO training stack — incomplete, see README
examples/cluster/    run:ai orchestration; site-specific reference only
```

**Key patterns:**

- `ExecutionResult` stores notebook outputs in nbformat as the single source of truth
- `ExecutionConfig` uses a factory pattern with deployment profiles
- Tools use `fhaviary` (aviary.core) for Message/Tool abstractions
- Benchmarking uses `ldp` for rollouts and `aviary.core.TaskDatasetServer` for serving envs
- Async throughout — uses jupyter_client's async APIs
- `first_wrong_step` is derived in Python from the judge's per-criterion step labels, never
  emitted directly by the LLM

**Forking invariants** (see `scripts/fork/fork_trajectory.py`):

- The server config used for a fork must match the run being forked, especially
  `include_protocol` and `capsule_dir` — replay re-executes the parent's actions, so a
  different prompt means a different task
- Replayed steps count against `AGENT_MAX_STEPS`; a fork with no budget left is skipped
  with reason `no_room`
- A fork is saved in the same layout as any other run, so it can be inspected, re-graded
  and forked again

## Configuration

Copy `.env.example` → `.env` and `server.example.yaml` / `benchmark.example.yaml` → their
un-suffixed names (all gitignored). `litellm` auto-loads `.env` from the working directory.

**Key environment variables:** `AGENT_MAX_STEPS` (default 30), `KERNEL_ENV_PATH`,
`DEPLOYMENT_PROFILE` (standard/gpu/long_timeout), `USE_DOCKER`, `STRIP_IMAGES` (default
true), `NB_OUTPUT_LIMIT` (default 3000 chars).

**A missing judge API key does not raise** — every episode scores 0.0 instead, which is
indistinguishable from a model that cannot do the task. Check this first when scores are
uniformly zero.

**File limits:** 256KB text, 10MB PDF/PowerPoint, 3000 char notebook output.

## CI

`.github/workflows/tests.yml` runs on PRs and pushes to main: pre-commit (ruff, mypy,
codespell, detect-secrets, prettier), then pytest `-n auto` on Python 3.11 and 3.13.
Tests needing a paid API key, network, Docker, an R kernel or capsule data skip themselves,
so a keyless clone and a fork PR both run green.

## Code Style

- Line length: 120 characters
- Docstrings: Google convention
- Type hints: required; strict mypy with the pydantic plugin
- Pre-commit hooks: ruff, mypy, codespell, detect-secrets, prettier
