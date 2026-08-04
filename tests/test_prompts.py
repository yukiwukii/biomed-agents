"""Tests for prompt templates."""

import re

import pytest

from hypotest.env.config import ExecutionConfig

# InterpreterEnv.reset() calls environment_capabilities_prompt.format(job_timeout=...),
# so job_timeout is the only placeholder the template may contain. Any other literal
# brace -- a dict comprehension or JSON snippet in an example, say -- is read by
# str.format as a replacement field and raises KeyError at reset, failing every rollout.
ALLOWED_PLACEHOLDER = "{job_timeout}"


@pytest.mark.parametrize("profile", ["standard", "gpu", "long_timeout"])
def test_environment_capabilities_prompt_formats(profile: str) -> None:
    config = ExecutionConfig.from_profile(profile)
    prompt = config.environment_capabilities_prompt

    stray = [m for m in re.findall(r"\{[^{}]*\}", prompt) if m != ALLOWED_PLACEHOLDER]
    assert not stray, f"unescaped braces in {profile} prompt: {stray}"

    rendered = prompt.format(job_timeout=config.job_timeout)
    assert str(config.job_timeout) in rendered
