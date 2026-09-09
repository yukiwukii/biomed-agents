import argparse
import asyncio
import contextvars
import json
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Literal, Self, cast

import yaml
from aviary.core import Environment, Message, TaskDatasetClient, ToolRequestMessage
from ldp.agent import Agent, AgentConfig, SimpleAgent
from ldp.agent.simple_agent import SimpleAgentState
from ldp.alg import RolloutManager
from ldp.alg.callbacks import Callback
from ldp.data_structures import Trajectory, Transition
from ldp.graph import OpResult, compute_graph
from ldp.llms import prepend_sys
from lmi.utils import update_litellm_max_callbacks
from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator, model_validator
from tqdm.asyncio import tqdm

from hypotest.dataset_server import DEFAULT_SERVER_PORT

# PATCH 2: HuggingFace Inference API only accepts tool_choice="auto" or "none".
# ldp's SimpleAgent defaults to tool_choice="required", which causes a 400 error.
# HFSimpleAgent overrides get_asv to pass tool_choice="auto" instead.
# To revert: delete HFSimpleAgent and HFSimpleAgentConfig, restore SimpleAgentConfig
# to its original one-liner, and change agent_type in benchmark.yaml back to "SimpleAgent".
class HFSimpleAgent(SimpleAgent):
    """SimpleAgent with tool_choice='auto' for HuggingFace Inference API compatibility."""

    @compute_graph()
    async def get_asv(
        self, agent_state: SimpleAgentState, obs: list[Message]
    ) -> tuple[OpResult[ToolRequestMessage], SimpleAgentState, float]:
        next_state = agent_state.get_next_state(obs)
        messages = (
            prepend_sys(next_state.messages, sys_content=self.sys_prompt)
            if self.sys_prompt is not None
            else next_state.messages
        )
        result = cast(
            "OpResult[ToolRequestMessage]",
            await self._llm_call_op(
                await self._config_op(), msgs=messages, tools=next_state.tools, tool_choice="auto"
            ),
        )
        next_state.messages = [*next_state.messages, result.value]
        return result, next_state, 0.0


class SimpleAgentConfig(AgentConfig):
    agent_type: Literal["SimpleAgent", "HFSimpleAgent"] = "SimpleAgent"  # type: ignore[mutable-override]

    def construct_agent(self):
        if self.agent_type == "HFSimpleAgent":
            return HFSimpleAgent(**self.agent_kwargs)
        return super().construct_agent()


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_url: str = f"http://localhost:{DEFAULT_SERVER_PORT}"
    api_key: str = Field(
        description="API key to access the server; passed either by value or as an environment variable."
    )
    agent_config: SimpleAgentConfig
    # Client-side HTTP timeout (seconds) for each env.reset()/env.step() RPC to the
    # dataset server. A single step blocks until the server finishes the agent's tool
    # call, so this MUST exceed the server's cell_execution_timeout (900s standard,
    # 1800s gpu/long_timeout) plus margin, or long-but-legal cells get killed with
    # ReadTimeout('Timeout on reading data from socket'). Override in benchmark.yaml.
    request_timeout: float = 960
    num_parallel: int = 16
    num_replications: int = 1  # [PATCH 13] k in avg@k / pass@k; 1 = original single-run behaviour
    results_dir: Path

    @model_validator(mode="after")
    def make_dirs(self) -> Self:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self

    @field_validator("api_key")
    @classmethod
    def read_from_env(cls, val: str) -> str:
        return os.getenv(val, val)


# PATCH 8: live per-step thinking log. See patches.md "Patch 8".
# To revert: delete extract_thinking + ThinkingLogger, drop the _task_idx contextvar and
# its set() in rollout(), and restore RolloutManager(agent) without callbacks.
#
# Set per rollout so the ThinkingLogger callback can label each entry with the task
# index. contextvars are copied into the tasks RolloutManager spawns, so the value set
# in rollout(idx) is visible inside after_transition.
_task_idx: contextvars.ContextVar[int] = contextvars.ContextVar("task_idx")


def _reasoning_from_ctx(action) -> str:
    """Return the LLMResult's ``reasoning_content`` for this action, or "".

    Anthropic models return thinking in a separate ``reasoning_content`` field rather
    than inline in the message content, so the ``<think>`` parsing below never sees it.
    ldp's LLMCallOp stashes the whole LLMResult in the op context (``ctx.update(call_id,
    "result", result)``) but returns only ``result.messages[0]`` as the action, so the
    reasoning is reachable only through the compute graph.

    Note this touches ldp internals (``OpResult._get_from_ctx``); an ldp upgrade could
    break it, in which case this returns "" and <think> parsing still works.
    """
    if getattr(action, "call_id", None) is None:
        return ""
    try:
        # Raises ValueError when the compute graph isn't available for this OpResult.
        result = action._get_from_ctx("result", default=None)  # noqa: SLF001
    except (ValueError, KeyError):
        return ""
    reasoning = getattr(result, "reasoning_content", None)
    return reasoning.strip() if isinstance(reasoning, str) else ""


def extract_thinking(action) -> str:
    """Return the model's thinking text for an action, or "".

    Handles both shapes: a separate ``reasoning_content`` field (Anthropic) and an
    inline ``<think>…</think>`` wrapper in the content (Qwen via vLLM).
    """
    if action is None:
        return ""
    reasoning = _reasoning_from_ctx(action)
    if reasoning:
        return reasoning
    content = getattr(getattr(action, "value", action), "content", None)
    if not content or not isinstance(content, str):
        return ""
    text = content.strip()
    if text.startswith("<think>"):
        text = text[len("<think>") :].lstrip()
    if "</think>" in text:
        return text[: text.index("</think>")].strip()
    # No closing tag: treat whole content as thinking only if there are no tool calls
    if not (getattr(getattr(action, "value", action), "tool_calls", None) or []):
        return text
    return ""


class ThinkingLogger(Callback):
    """Writes each step's thinking text to a JSON file the moment the step completes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, list[dict]] = {}
        self.lock = asyncio.Lock()

    async def after_transition(
        self, traj_id: str, agent: Agent, env: Environment, transition: Transition
    ) -> None:
        thinking = extract_thinking(transition.action)
        if not thinking:
            return
        try:
            label = f"task_{_task_idx.get()}"
        except LookupError:
            label = traj_id
        async with self.lock:
            self.data.setdefault(label, []).append(
                {"step": transition.timestep, "thinking": thinking}
            )
            self.path.write_text(json.dumps(self.data, indent=2))


async def main() -> None:
    # PATCH 9: litellm noise suppression (callback limit + pydantic warnings).
    # See patches.md "Patch 9".
    #
    # Raise litellm's MAX_CALLBACKS limit. Each LiteLLMModel created during rollouts
    # appends logging callbacks without deduping, so the default cap of 30 is quickly
    # exceeded, spamming "Cannot add callback" warnings. See litellm#9792.
    # The explicit import populates litellm.litellm_core_utils.logging_callback_manager,
    # which lmi reaches via attribute access but which some litellm versions don't
    # auto-load (AttributeError otherwise). Guarded: this only suppresses noise, so it
    # must never block startup.
    try:
        import litellm.litellm_core_utils.logging_callback_manager  # noqa: F401

        update_litellm_max_callbacks()
    except Exception:  # noqa: BLE001
        pass

    # Silence cosmetic pydantic serializer warnings from litellm's cost tracker. It runs
    # each response through litellm's strictly-typed ModelResponse (choices typed as
    # StreamingChoices, message as litellm.Message), but the actual objects are a
    # non-streaming Choices wrapping an lmi/aviary Message, so pydantic warns on the type
    # mismatch while still serializing correctly. Cost is 0 for the custom vLLM endpoints
    # anyway, so nothing of value is lost.
    warnings.filterwarnings(
        "ignore", message="Pydantic serializer warnings:", category=UserWarning, module="pydantic.main"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=FilePath)
    config_path = parser.parse_args().config
    config = BenchmarkConfig.model_validate(yaml.safe_load(config_path.read_text()))

    client = TaskDatasetClient(config.server_url, api_key=config.api_key, request_timeout=config.request_timeout)
    agent = cast(SimpleAgent, config.agent_config.construct_agent())
    semaphore = asyncio.Semaphore(config.num_parallel)
    # PATCH 8: ThinkingLogger writes per-step thinking to thinking.json live.
    rm = RolloutManager(agent, callbacks=[ThinkingLogger(config.results_dir / "thinking.json")])

    async def rollout(idx: int, rep: int = 0) -> tuple[Trajectory, float]:
        async with semaphore:
            _task_idx.set(idx)  # PATCH 8: label this rollout's thinking entries
            env = client.get_new_env_by_idx(idx)
            trajectory, *_ = await rm.sample_trajectories(environments=[env])
            # [PATCH 13] include rep suffix when running multiple replications
            suffix = f"_rep{rep}" if config.num_replications > 1 else ""
            trajectory.traj_id = f"task_{idx}{suffix}"
            # assume only terminal reward
            return trajectory, trajectory.steps[-1].reward

    # [PATCH 13] run each problem num_replications times for avg@k / pass@k
    k = config.num_replications
    n_problems = len(client)
    t0 = time.monotonic()
    results = await tqdm.gather(
        *[rollout(i, r) for i in range(n_problems) for r in range(k)],
        ncols=0, desc="Rollouts"
    )
    elapsed = time.monotonic() - t0
    trajectories, rewards = zip(*results, strict=True)

    (config.results_dir / "rewards.json").write_text(
        json.dumps({t.traj_id: r for t, r in zip(trajectories, rewards, strict=True)}, indent=2)
    )
    (config.results_dir / "trajectories.pkl").write_bytes(pickle.dumps(trajectories))
    # [PATCH 13] group rewards by problem; each block of k entries belongs to one problem
    problem_rewards = [[rewards[i * k + r] for r in range(k)] for i in range(n_problems)]
    avg_at_k = sum(sum(rs) / k for rs in problem_rewards) / n_problems
    pass_at_k = sum(1 for rs in problem_rewards if max(rs) == 1.0) / n_problems
    print(f"avg@{k}: {avg_at_k:.2f}")
    print(f"pass@{k}: {pass_at_k:.2f}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())


def cli() -> None:
    """Console-script entry point (``hypotest-benchmark``)."""
    asyncio.run(main())
