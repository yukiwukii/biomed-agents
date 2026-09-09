import argparse
import asyncio
import contextlib
import os
import random
import shutil
import socket
import time
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Literal, Self, cast
from uuid import UUID

import yaml
from aviary.core import TaskDataset, TaskDatasetServer
from datasets import Dataset as HFDataset
from datasets import load_dataset
from lmi import LiteLLMModel
from lmi.utils import update_litellm_max_callbacks
from pydantic import BaseModel, ConfigDict, DirectoryPath, Field, FilePath, field_validator, model_validator

from hypotest.env.config import ExecutionConfig
from hypotest.env.interpreter_env import InterpreterEnv, InterpreterEnvConfig, ProblemInstance
from hypotest.env.kernel_server import NBLanguage

# Sibling of the capsules holding ground-truth files; see Dataset.get_new_env_by_idx.
TRUTH_DIR_NAME = "_truth"

# --- idle-environment sweeper -------------------------------------------------
#
# WHY: an environment that is never `/close`d stays in TaskDatasetServer.envs
# forever, holding BOTH a live Jupyter kernel (memory) and its capsule copy in
# work_dir (disk, up to ~1.1 GB -- get_new_env_by_idx copytree's a full capsule
# per rollout). Nothing expires them.
#
# Measured on the 2026-08-11 GRPO run: 12 `/start` against 9 `/close` over 8
# episodes -- 3 orphans in 2 training steps. The extra `/start`s are retries, and
# a retry abandons the previous environment without closing it. The env pod grew
# to ~275 GiB against a 64 GB request and was evicted by the kubelet:
#
#   Evicted: node was low on resource: memory. Container hypotest-env was using
#   288108748Ki, request is 64G.
#
# InterpreterEnv.close() is correct -- it shuts the kernel down and moves/removes
# work_dir. The bug is only that it is never reached for orphans.
#
# aviary ships a `/close_old_envs` endpoint for exactly this, but nothing has ever
# called it. This sweeps in-process instead, so no caller has to remember.
#
# THRESHOLD: `_get_env` refreshes the timestamp at the START of `/reset` and
# `/step`, so an active episode keeps itself alive -- but a single legitimate cell
# can run for `cell_execution_timeout` (600 s), during which the env looks idle.
# The default is therefore well above that. Sweeping a LIVE environment kills a
# running episode and wastes GPU time, so err long.
ENV_SWEEP_IDLE_SECONDS = float(os.getenv("ENV_SWEEP_IDLE_SECONDS", "1800"))
ENV_SWEEP_PERIOD_SECONDS = float(os.getenv("ENV_SWEEP_PERIOD_SECONDS", "300"))
# Per-env cap so one wedged close cannot stall the whole sweep.
ENV_SWEEP_CLOSE_TIMEOUT = float(os.getenv("ENV_SWEEP_CLOSE_TIMEOUT", "120"))


async def sweep_idle_envs(
    server: TaskDatasetServer,
    idle_seconds: float = ENV_SWEEP_IDLE_SECONDS,
    period_seconds: float = ENV_SWEEP_PERIOD_SECONDS,
) -> None:
    """Periodically close environments no client has touched in `idle_seconds`.

    Runs until cancelled. Never raises: a failure here must not take the server
    down, and must not stop later sweeps.

    Args:
        server: the running TaskDatasetServer whose `envs` map is swept.
        idle_seconds: close environments untouched for at least this long.
            Non-positive disables sweeping entirely.
        period_seconds: how often to check.
    """
    if idle_seconds <= 0:
        print("[env-sweeper] disabled (ENV_SWEEP_IDLE_SECONDS <= 0)", flush=True)
        return

    print(
        f"[env-sweeper] on: closing envs idle > {idle_seconds:.0f}s, checking every {period_seconds:.0f}s",
        flush=True,
    )
    while True:
        await asyncio.sleep(period_seconds)
        try:
            # server.envs stores wall-clock time.time(); compare in the same clock.
            wall_now = time.time()
            async with server.lock:
                stale = [
                    (env_id, env)
                    for env_id, (env, last_used) in list(server.envs.items())
                    if wall_now - last_used > idle_seconds
                ]
                for env_id, env in stale:
                    try:
                        await asyncio.wait_for(env.close(), timeout=ENV_SWEEP_CLOSE_TIMEOUT)
                    except Exception as exc:  # noqa: BLE001
                        # Untrack regardless. As aviary's own /close notes, a
                        # failed close means the env is probably already broken --
                        # keeping it tracked leaks it forever, which is the exact
                        # problem this sweeper exists to fix.
                        print(f"[env-sweeper] close failed for {env_id}, untracking anyway: {exc!r}", flush=True)
                    finally:
                        server.envs.pop(env_id, None)
            if stale:
                print(f"[env-sweeper] closed {len(stale)} idle env(s); {len(server.envs)} still tracked", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[env-sweeper] sweep failed, continuing: {exc!r}", flush=True)


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem_jsonl: FilePath | None = None
    capsule_dir: DirectoryPath
    hf_dataset: str | None = None

    rubric_model: str = "openai/gpt-5"
    rubric_model_config: dict[str, str | list[Any]] = Field(
        default_factory=lambda: cast(dict[str, str | list[Any]], {"reasoning_effort": "medium"})
    )

    work_dir: Path | None = None
    use_ray: bool = False
    use_docker: bool = False
    use_enroot: bool = False
    container_sqsh_path: FilePath | None = None
    force_python: bool = True
    normalize_reward: bool = True
    save_dir: Path | None = None
    max_problems: int | None = None  # [PATCH 1] remove this field to revert

    execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)
    include_protocol: bool = True
    # [PATCH 17/20] Run-wide judge override by registered name (see env/judges/). "auto"
    # (default) lets each problem's own `judge` field decide, falling back to rubric-format
    # sniffing for datasets that predate it. Set to "hypotest"/"biomni"/"heureka"/… to
    # force one judge across the run. Flows into InterpreterEnvConfig via model_dump() splat.
    judge: str = "auto"

    @model_validator(mode="after")
    def validate_enroot(self) -> Self:
        if self.use_enroot and not self.container_sqsh_path:
            raise ValueError("container_sqsh_path cannot be empty when use_enroot is set")
        return self

    @model_validator(mode="after")
    def validate_dataset_source(self) -> Self:
        if not self.problem_jsonl and not self.hf_dataset:
            raise ValueError("Either problem_jsonl or hf_dataset must be provided")
        return self

    @model_validator(mode="after")
    def make_dirs(self) -> Self:
        for d in (self.work_dir, self.save_dir):
            if d:
                d.mkdir(parents=True, exist_ok=True)
        return self

    def load_problems(self) -> list[ProblemInstance]:
        # [PATCH 1] to revert: restore the original one-liner returns (remove if/else and slice)
        if self.hf_dataset:
            problems = self._load_from_hf()
        else:
            assert self.problem_jsonl is not None
            problems = [ProblemInstance.model_validate_json(line) for line in self.problem_jsonl.read_text().splitlines()]
        return problems[: self.max_problems]

    def _load_from_hf(self) -> list[ProblemInstance]:
        ds: HFDataset = load_dataset(self.hf_dataset, split="train")
        return [ProblemInstance.model_validate(row) for row in ds]


class Dataset(TaskDataset[InterpreterEnv]):
    def __init__(self, config: DatasetConfig):
        self.config = config

        self.problems = self.config.load_problems()

        self.rubric_model = LiteLLMModel(name=self.config.rubric_model, config=self.config.rubric_model_config)

        self.problem_counter: Counter[UUID] = Counter()

    def get_new_env_by_idx(self, idx: int) -> InterpreterEnv:
        problem = self.problems[idx]
        problem_count = self.problem_counter[problem.id]
        self.problem_counter[problem.id] += 1
        run_id = f"{problem.id}-iter{problem_count}"

        capsule_path = self.config.capsule_dir / problem.input_data_path
        if not capsule_path.exists():
            capsule_path = self.config.capsule_dir / f"CapsuleData-{problem.id}"
        # [PATCH 24] The answer key for deterministic judges, staged by the converter as a sibling
        # of the capsules (scripts/stage_bioagent_capsules.py writes `_truth/<task_id>/`). Passed
        # to the env but never copied into the workspace, so the agent cannot read it. Benchmarks
        # without truth files simply have no `_truth/` dir and get None.
        truth_path = self.config.capsule_dir / TRUTH_DIR_NAME / problem.input_data_path
        # [PATCH 5] Resolve work_dir/save_dir to absolute paths. A relative work_dir
        # (e.g. `work_dir: tmp/` in server.yaml) gets propagated verbatim into the
        # kernel's PYTHONPATH/PIP_TARGET (via Interpreter._setup_pip_env) as a relative
        # string. Because the kernel's cwd is itself the work_dir, that relative entry
        # re-resolves against cwd into a doubled `work_dir/tmp/<run_id>/pydeps` path,
        # producing a nested tmp dir and a malformed sys.path that breaks numpy import
        # ("do not import numpy from its source directory"). Absolute paths avoid this.
        # To revert: drop the `.resolve()` calls.
        problem_dir = (Path(self.config.work_dir) / run_id).resolve() if self.config.work_dir else Path(mkdtemp())
        if problem_dir.exists():
            shutil.rmtree(problem_dir)
        problem_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(capsule_path, problem_dir, dirs_exist_ok=True)

        save_dir = (Path(self.config.save_dir) / run_id).resolve() if self.config.save_dir else None

        language = (
            NBLanguage.PYTHON if self.config.force_python else NBLanguage.from_string(problem.nb_primary_language)
        )
        language = language if language is not None else NBLanguage.PYTHON  # default auto language to python

        return InterpreterEnv(
            problem=problem,
            rubric_model=self.rubric_model,
            work_dir=problem_dir,
            save_dir=save_dir,
            truth_dir=truth_path if truth_path.exists() else None,
            config=InterpreterEnvConfig(language=language, **self.config.model_dump()),
        )

    def __len__(self) -> int:
        return len(self.problems)


HypotestDataset = Dataset
HypotestDatasetConfig = DatasetConfig


DEFAULT_SERVER_PORT = 8405


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig
    api_key: str = Field(
        description="API key to access the server; passed either by value or as an environment variable."
    )
    port: int = DEFAULT_SERVER_PORT

    @field_validator("api_key")
    @classmethod
    def read_from_env(cls, val: str) -> str:
        return os.getenv(val, val)

    def model_post_init(self, _):
        # Assign a random port when non-positive value.
        if self.port <= 0:
            self.port = random.randint(1024, 65535)


async def launch_server():
    # PATCH 9: Raise litellm's MAX_CALLBACKS limit so rubric-model LiteLLMModel instances
    # don't spam "Cannot add callback" warnings once the default cap of 30 is hit. See
    # litellm#9792 and patches.md "Patch 9".
    # The explicit import populates litellm.litellm_core_utils.logging_callback_manager,
    # which lmi reaches via attribute access but which some litellm versions don't
    # auto-load (AttributeError otherwise). Guarded: this only suppresses noise, so it
    # must never block server startup.
    try:
        import litellm.litellm_core_utils.logging_callback_manager  # noqa: F401

        update_litellm_max_callbacks()
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=FilePath, nargs="?")
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--api-key", type=str, default=os.getenv("HYPOTEST_API_KEY"))
    parser.add_argument("--problem-jsonl", type=FilePath)
    parser.add_argument("--hf-dataset", type=str)
    parser.add_argument("--capsule-dir", type=DirectoryPath)
    parser.add_argument("--rubric-model", type=str)
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--rubric-model-api-base", type=str, default=os.getenv("HYPOTEST_RUBRIC_MODEL_API_BASE"))
    parser.add_argument("--rubric-model-api-key", type=str, default=os.getenv("HYPOTEST_RUBRIC_MODEL_API_KEY"))
    parser.add_argument("--use-docker", action="store_true")
    parser.add_argument("--use-ray", action="store_true")
    parser.add_argument("--use-enroot", action="store_true")
    parser.add_argument("--container-sqsh", type=FilePath)

    args = parser.parse_args()

    if args.config and args.config.exists():
        config = ServerConfig.model_validate(yaml.safe_load(args.config.read_text()))
    else:
        config = ServerConfig(
            dataset=DatasetConfig(
                problem_jsonl=args.problem_jsonl,
                hf_dataset=args.hf_dataset,
                capsule_dir=args.capsule_dir,
                rubric_model=args.rubric_model,
                rubric_model_config={
                    "model_list": [
                        {
                            "model_name": args.rubric_model,
                            "litellm_params": {
                                "model": args.rubric_model,
                                "api_base": args.rubric_model_api_base,
                                "api_key": args.rubric_model_api_key,
                                "reasoning_effort": args.reasoning_effort,
                                "drop_params": True,
                            },
                        },
                    ],
                },
                use_docker=args.use_docker,
                use_ray=args.use_ray,
                use_enroot=args.use_enroot,
                container_sqsh_path=args.container_sqsh,
                execution_config={"cell_execution_timeout": 600},
            ),
            port=args.port,
            api_key=args.api_key,
        )

    dataset = Dataset(config.dataset)
    server = TaskDatasetServer(dataset, port=config.port, api_key=config.api_key)

    ip_address = socket.gethostbyname(socket.gethostname())
    print(f"Starting dataset server: IPAddress={ip_address} Port={config.port}", flush=True)

    # Reap orphaned environments alongside the server. Without this, an env that
    # is never `/close`d (retries abandon one each time) keeps its Jupyter kernel
    # and its ~1 GB capsule copy forever -- see sweep_idle_envs.
    sweeper = asyncio.create_task(sweep_idle_envs(server))
    try:
        await server.astart()
    finally:
        sweeper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweeper


if __name__ == "__main__":
    asyncio.run(launch_server())
