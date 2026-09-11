import os
from pathlib import Path

from pydantic import BaseModel, model_validator

AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "30"))

USE_DOCKER = bool(os.getenv("USE_DOCKER", "false").lower() == "true")
USE_HOST_ENV_VARS = bool(os.getenv("USE_HOST_ENV_VARS", "false").lower() == "true")
REQUIRED_PATH_ENV_VARS = os.getenv("REQUIRED_PATH_ENV_VARS", "PATH,PYTHONPATH,CUDA_HOME,LD_LIBRARY_PATH,PATH']").split(
    ","
)
NB_ENVIRONMENT_DOCKER_IMAGE = os.getenv("NB_ENVIRONMENT_DOCKER_IMAGE", "interpreter-env:latest")
KERNEL_ENV_PATH = os.getenv("KERNEL_ENV_PATH", "/app/kernel_env")
CONTAINER_WORKSPACE_PREFIX = os.getenv("CONTAINER_WORKSPACE_PREFIX", "/tmp/data_workspace")  # noqa: S108

# Kernel server settings (for Docker-based execution)
KERNEL_SERVER_PORT = 8000
KERNEL_SERVER_STARTUP_TIMEOUT = float(
    os.getenv("KERNEL_SERVER_STARTUP_TIMEOUT", "30.0")
)  # seconds to wait for health check

MAX_FILES_TO_UPLOAD = int(os.getenv("MAX_FILES_TO_UPLOAD", "100"))

# Some R error messages can be 100,000 of characters.
# 2026-08-14: env-tunable (was a hardcoded 3000). Tool/cell OUTPUT is ~52% of an
# episode's tokens and is the main lever on episode LENGTH without capping the
# agent's turns -- lowering this shrinks episodes so the 27B trainer forward fits
# on 80 GB H100 (a 48.5k-token episode OOM'd the forward by ~1 GiB). It changes
# what the MODEL SEES per cell (capability/reward-adjacent), so it is a knob, not
# a silent default. Set via NB_OUTPUT_LIMIT in the env pod (workloads.sh).
NB_OUTPUT_LIMIT = int(os.getenv("NB_OUTPUT_LIMIT", "3000"))  # chars

# Strip images before they reach the policy.
#
# WHY: a plot comes back as a base64 data URI. A vision-capable server prices it
# by pixels (~1.8k tokens), but the GRPO path serves the policy with
# `language_model_only: true` -- no vision tower -- so vLLM tokenizes the base64
# as TEXT. One matplotlib figure measured 101,573 tokens (56x its pixel price)
# and another 194,763; both blew the 131,072 context, returned HTTP 400, and
# killed the episode before it could submit an answer, scoring 0. Note that
# NB_OUTPUT_LIMIT above truncates text but has never applied to images, so the
# single largest thing a cell can return was also the only uncapped one.
#
# Nothing downstream wants the pixels: the judge reads `view_notebook`'s markdown
# and its image list is discarded (interpreter_env.py `_score_solution`), and 0
# of 341 rubric criteria across the 51 capsules require a figure.
#
# Images remain in the notebook itself -- this strips them from what the MODEL
# sees, not from the saved record. Set STRIP_IMAGES=false to restore the old
# behaviour when serving a policy that really can consume images.
STRIP_IMAGES = bool(os.getenv("STRIP_IMAGES", "true").lower() == "true")
# Streams from a docker container. Don't set to `sys.stdout.fileno()`
# because we want to differentiate from file I/O
DOCKER_STREAM_TYPE_STDOUT = 1
DOCKER_STREAM_TYPE_STDERR = 2

STAGE = os.getenv("STAGE", "local")
DATA_STORAGE_PATH = Path("storage") if STAGE == "local" else Path("/storage")

VALID_FROM_TASK_KWARGS = [
    "run_notebook_on_edit",
]


# Time management messages
FORCE_MSG = (
    "TIME EXPIRED - IMMEDIATE ACTION REQUIRED:\n"
    "• You are out of time\n"
    "• Do NOT run any more cells\n"
    "• Do NOT perform any more analysis\n"
    "• ONLY action allowed: submit_answer\n"
    "• Submit your answer NOW based on current findings"
)
WARN_MSG = "Warning: {remaining} seconds remaining. Plan to submit your current findings as an answer soon."


class ExecutionConfig(BaseModel):
    """Execution environment configuration - varies by deployment profile."""

    # Timing
    # ! THRESHOLD = SECONDS BEFORE TIMEOUT
    job_timeout: int = 60 * 60
    warn_submit_threshold: int = 20 * 60
    force_submit_threshold: int = 10 * 60
    cell_execution_timeout: int = 15 * 60

    # safety
    safe_execute: bool = os.getenv("SAFE_EXECUTE_SANDBOX", "").lower() == "true"

    # Capabilities
    has_gpu: bool = False

    # Prompt section (set in model_post_init based on capabilities)
    environment_capabilities_prompt: str = ""

    @model_validator(mode="after")
    def check_thresholds(self) -> "ExecutionConfig":
        if self.warn_submit_threshold <= self.force_submit_threshold:
            raise ValueError(
                f"warn_submit_threshold ({self.warn_submit_threshold}) must be greater than "
                f"force_submit_threshold ({self.force_submit_threshold}). "
                "It is seconds before timeout."
            )
        return self

    def model_post_init(self, __context, /) -> None:
        if not self.environment_capabilities_prompt:
            from . import prompts  # noqa: PLC0415

            # PATCH 10: append software-stack capabilities (item 6) after env capabilities (item 5)
            self.environment_capabilities_prompt = (
                (prompts.GPU_ENVIRONMENT_CAPABILITIES if self.has_gpu else prompts.CPU_ENVIRONMENT_CAPABILITIES)
                + "\n\n"
                + prompts.SOFTWARE_STACK_CAPABILITIES
            )

    @classmethod
    def standard(cls, **overrides) -> "ExecutionConfig":
        """Standard CPU environment with default timeouts."""
        return cls(**overrides)

    @classmethod
    def gpu(cls, **overrides) -> "ExecutionConfig":
        """GPU environment with longer timeouts."""
        defaults = {
            "job_timeout": 2 * 60 * 60,
            "warn_submit_threshold": 40 * 60,
            "force_submit_threshold": 25 * 60,
            "cell_execution_timeout": 30 * 60,
            "has_gpu": True,
        }
        return cls(**{**defaults, **overrides})  # noqa: FURB173

    @classmethod
    def long_timeout(cls, **overrides) -> "ExecutionConfig":
        """CPU environment with longer timeouts."""
        defaults = {
            "job_timeout": 2 * 60 * 60,
            "warn_submit_threshold": 40 * 60,
            "force_submit_threshold": 25 * 60,
            "cell_execution_timeout": 30 * 60,
            "has_gpu": False,
        }
        return cls(**{**defaults, **overrides})  # noqa: FURB173

    @classmethod
    def from_profile(cls, profile: str, **overrides) -> "ExecutionConfig":
        """Factory to get config by profile name."""
        factories = {
            "standard": cls.standard,
            "gpu": cls.gpu,
            "long_timeout": cls.long_timeout,
        }
        if profile not in factories:
            raise ValueError(f"Unknown profile: {profile}. Available: {list(factories.keys())}")
        return factories[profile](**overrides)

    @classmethod
    def from_env(cls) -> "ExecutionConfig":
        """Create config from DEPLOYMENT_PROFILE env var."""
        profile = os.getenv("DEPLOYMENT_PROFILE", "standard")
        return cls.from_profile(profile)

    @classmethod
    def from_timeouts(
        cls,
        job_timeout: int | None,
        cell_execution_timeout: int | None,
    ) -> "ExecutionConfig":
        """Create ExecutionConfig from timeout values with proportionally scaled thresholds.

        GPU status and default timeouts are inherited from the DEPLOYMENT_PROFILE
        environment variable. This is useful for subagents that should inherit
        their parent's deployment profile settings.

        Args:
            job_timeout: Overall job timeout in seconds. If None, inherits from
                deployment profile.
            cell_execution_timeout: Per-cell execution timeout in seconds. If None,
                inherits from deployment profile.

        Returns:
            ExecutionConfig with proportionally scaled thresholds.
        """
        # Inherit GPU status from deployment profile
        profile = os.getenv("DEPLOYMENT_PROFILE", "standard")
        has_gpu = profile == "gpu"

        if job_timeout is None:
            job_timeout = cls.from_env().job_timeout

        if cell_execution_timeout is None:
            cell_execution_timeout = cls.from_env().cell_execution_timeout

        # Use standard profile ratios (not GPU/long_timeout which use ~1/5 for force):
        # - warn = 1/3 of job_timeout
        # - force = 1/6 of job_timeout
        warn_submit_threshold = max(job_timeout // 3, 120)
        force_submit_threshold = max(job_timeout // 6, 60)

        # Ensure warn > force (required by validator)
        if warn_submit_threshold <= force_submit_threshold:
            warn_submit_threshold = force_submit_threshold + 60

        return cls(
            job_timeout=job_timeout,
            warn_submit_threshold=warn_submit_threshold,
            force_submit_threshold=force_submit_threshold,
            cell_execution_timeout=cell_execution_timeout,
            has_gpu=has_gpu,
        )
