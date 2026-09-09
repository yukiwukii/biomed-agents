"""Pytest configuration and shared fixtures for hypotest tests."""

import importlib.util
import logging
import os
import shutil
import subprocess
import urllib.request
from uuid import UUID

import pytest
from huggingface_hub import get_token
from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel

from hypotest.env import config as cfg
from hypotest.env.interpreter_env import ProblemInstance

MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None
requires_matplotlib = pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available")

logger = logging.getLogger(__name__)

IN_GITHUB_ACTIONS: bool = os.getenv("GITHUB_ACTIONS") == "true"

# Tests that reach the network or a paid API. A fresh clone has neither, and a
# fork PR cannot see repository secrets, so these skip rather than fail -- an
# outside contributor should get a green run without credentials.
requires_openai = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")


def skip_if_hub_dataset_unavailable() -> None:
    """Skip the calling test when the gated HF dataset cannot be fetched.

    EdisonScientific/bixbench_hypothesis is a GATED dataset: the Hub returns 200
    for its metadata but refuses the download without an accepted-terms token. So
    a missing HF_TOKEN is an access problem, not a transport one, and CI has no
    token. Both that and an unreachable Hub skip; anything else -- a rename, a
    schema change, a genuinely broken loader -- still FAILS, which is the point of
    the test.

    Set HYPOTEST_SKIP_NETWORK_TESTS=1 to skip unconditionally.
    """
    if os.getenv("HYPOTEST_SKIP_NETWORK_TESTS") == "1":
        pytest.skip("HYPOTEST_SKIP_NETWORK_TESTS=1")
    # get_token() is the authoritative check: it covers HF_TOKEN, the legacy
    # env vars, AND a stored `huggingface-cli login` token under ~/.cache. An
    # env-var-only check would skip on a machine that is in fact logged in.
    if not get_token():
        pytest.skip("no HuggingFace token; EdisonScientific/bixbench_hypothesis is gated")
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5).close()
    except OSError as exc:
        pytest.skip(f"huggingface.co unreachable: {exc}")


def docker_image_exists() -> bool:
    """Check if the Docker image exists."""
    try:
        docker_path = shutil.which("docker")
        if not docker_path:
            logger.info("Docker is not installed on this system")
            return False
        result = subprocess.run(  # noqa: S603
            [docker_path, "images", cfg.NB_ENVIRONMENT_DOCKER_IMAGE, "-q"],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            logger.info(f"Docker image {cfg.NB_ENVIRONMENT_DOCKER_IMAGE} found")
            return True
        logger.info(f"Docker image {cfg.NB_ENVIRONMENT_DOCKER_IMAGE} not found")
        return False  # noqa: TRY300
    except subprocess.CalledProcessError:
        logger.info("Docker is not available on this system")
        return False


def should_skip_docker_test(use_docker: bool) -> bool:
    """Determine if Docker tests should be skipped."""
    if use_docker and (IN_GITHUB_ACTIONS or not docker_image_exists()):
        logger.info(f"Skipping docker test in CI environment: {IN_GITHUB_ACTIONS} and use_docker={use_docker}")
        return True
    return False


def should_skip_r_test(language_str: str) -> bool:
    """Determine if R tests should be skipped."""
    if language_str.upper() != "R":
        return False

    try:
        kernel_spec_manager = KernelSpecManager()
        kernel_spec_manager.get_kernel_spec("ir")
    except NoSuchKernel:
        return True
    return False


@pytest.fixture
def skip_if_docker_unavailable(request):
    """Fixture to skip Docker tests when Docker is not available."""
    if hasattr(request, "param") and request.param and should_skip_docker_test(request.param):
        pytest.skip("Docker not available or image not found")


@pytest.fixture
def skip_if_r_unavailable(request):
    """Fixture to skip R tests when R kernel is not available."""
    if hasattr(request, "param") and request.param and should_skip_r_test(request.param):
        pytest.skip("R kernel is not available")


@pytest.fixture
def images_enabled(monkeypatch):
    """Turn image stripping off for tests that assert the multimodal contract.

    ``cfg.STRIP_IMAGES`` defaults to True, so images never reach the policy. The
    plumbing behind it still has to work for a vision-capable deployment, and
    these tests are what guard it -- they assert the ``STRIP_IMAGES=false`` path,
    not the default one. ``_extract_images_from_output`` reads the flag at call
    time, so patching the attribute is enough.
    """
    monkeypatch.setattr(cfg, "STRIP_IMAGES", False)


@pytest.fixture
def default_problem() -> ProblemInstance:
    """Default ProblemInstance for tests."""
    return ProblemInstance(
        id=UUID("00000000-0000-0000-0000-000000000000"),
        hypothesis="Test hypothesis",
        protocol="Test objective",
        answer=True,
        rubric="Test rubric",
        max_points=10,
    )
