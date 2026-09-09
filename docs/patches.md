# Patches

Local workaround patches applied on top of the upstream codebase. Each section describes what changed, why, and how to revert it.

> **Status, 2026-08-11 — this file is a record, not a to-do list.**
>
> The setup was cut back to a single model (Qwen3.5-9B). Patches 1-28 all touch
> `src/hypotest` — the benchmark harness, the judge and the fork tooling — and are
> **unchanged and still applied**; the reward path in particular was deliberately
> left alone. Only the *training* side was simplified.
>
> What changed:
>
> | | |
> | --- | --- |
> | **Patch 29** (grad buffer) | still applied, but now conditional — see [`handoff.md`](handoff.md) §3 and §5. If the 32768 cap is chosen it should be deleted, as it halves nothing while the logits arrive fp32 |
> | **Patch 30** (strip images) | unchanged, still applied, and load-bearing |
> | the three other NeMo RL patches | unchanged and **required**: `HYPOTEST_TP_PLAN_PATCH`, `HYPOTEST_FP32_CAST_PATCH`, `HYPOTEST_TRAIN_CHUNK_PATCH`. All three fix a config value that upstream accepts and then ignores. See `handoff.md` §5 |
>
> The open question Patch 29 was written against is now answered: the logits are
> fp32 because `output_dtype=torch.float32` is hardcoded in FSDP2's
> `MixedPrecisionPolicy` (`automodel/setup.py:443`), with no config knob.
>
> Anything below referring to `grpo_hypotest_27b.yaml`, `grpo_hypotest_35b_a3b.yaml`,
> `grpo_hypotest_0_8b.yaml`, `rl/docker/Dockerfile.train` or `qwen3_6_tp_plan.py`
> describes files that no longer exist. The TP plan was renamed to
> `rl/configs/qwen3_tp_plan.py`.

See also: [Runbook — recovering a broken `.venv` or kernel env](#runbook--recovering-a-broken-venv-or-kernel-env) at the end of this file.

---

## Patch 1 — `max_problems` limit in DatasetConfig

**Files:** `src/hypotest/dataset_server.py`

**What:** Added `max_problems: int | None = None` field to `DatasetConfig`. `load_problems()` slices the returned list with `[:self.max_problems]`, so setting this field caps how many problems are served without editing the dataset file.

**Why:** Needed a quick way to run short smoke-test benchmark runs without processing the full dataset.

**To revert:**
- Remove the `max_problems` field from `DatasetConfig` (line marked `[PATCH 1]`).
- In `load_problems()`, restore the original one-liner returns (remove the `if/else` block and the `[:self.max_problems]` slice — the method should just `return self._load_from_hf()` or the list comprehension directly).

---

## Patch 2 — HFSimpleAgent for HuggingFace Inference API compatibility

**Files:** `src/hypotest/benchmark_agent.py`

**What:** Added `HFSimpleAgent` (subclass of `SimpleAgent`) and `HFSimpleAgentConfig`. `HFSimpleAgent.get_asv` passes `tool_choice="auto"` instead of the ldp default `"required"`. `SimpleAgentConfig` in this file is a thin wrapper that selects between the two agent types.

**Why:** The HuggingFace Inference API rejects requests with `tool_choice="required"` with a 400 error. The upstream `SimpleAgent` hardcodes `"required"`.

**To revert:**
- Delete `HFSimpleAgent` and `HFSimpleAgentConfig`.
- Restore `SimpleAgentConfig` to its original one-liner (direct import/alias from ldp).
- Change `agent_type` in `benchmark.yaml` back to `"SimpleAgent"`.

---

## Patch 3 — Python version portability + rubric thinking tokens

**Files:** `src/hypotest/env/interpreter.py`, `src/hypotest/env/interpreter_env.py`

**What:** Three related changes shipped together:

1. **PYTHONPATH version stripping** (`interpreter.py`): Filters out any `python3.X` paths from the *host env* `PYTHONPATH` that don't match the running interpreter version, then merges with `extra_envs`. The filter is applied BEFORE the merge so that paths explicitly set in `extra_envs` (e.g. `kernel_site_packages`) are never stripped — this fixed a bug where the old post-merge filter silently dropped the kernel's site-packages when the outer Python minor version differed from `kernel_env`'s Python version, causing `No module named 'sklearn'` errors.
   - To revert: replace the filtering block (from `current_pyver = ...` through `kwargs["env"] = merged`) with `kwargs["env"] = env | self.extra_envs`.

2. **Dynamic kernel Python version** (`interpreter_env.py`, around `_reset` / `reset`): Resolves the Python version inside `kernel_env` at runtime by globbing `lib/python3.*` instead of hardcoding `python3.12`.
   - To revert: replace the three lines (`lib_path`, `pyver_dirs`, `kernel_pyver`) with `kernel_site_packages = kernel_env_path / "lib" / "python3.12" / "site-packages"`.

3. **Rubric thinking tokens** (`interpreter_env.py`, `_score_solution` / `submit_answer`): Captures `resp.reasoning_content` into `score_metadata["reasoning"]`, and returns the full rubric model output (thinking + evaluation text) as the final observation instead of just `CORRECT_MSG` / `INCORRECT_MSG`.
   - To revert thinking capture: delete the two lines under `# PATCH 3: also capture thinking tokens`.
   - To revert full rubric return: replace the `parts` block in `submit_answer` with `return CORRECT_MSG if correct else INCORRECT_MSG`.
   - **Narrowed by Patch 19:** the `Rubric evaluation:` payload is now the *derived* criteria, not the judge's raw JSON. The thinking block and the raw text in `score_info.json` are unaffected.

---

## Patch 4 — pip installs redirected to work_dir/pydeps (local kernel path)

**Files:** `src/hypotest/env/interpreter.py`

**What:** Added `Interpreter._setup_pip_env()`, called from `Interpreter.start()` for both the `use_host_env_vars=True` and `False` branches. The method:
- Creates `work_dir/pydeps` and `work_dir/pip-cache` directories.
- Sets `PIP_TARGET=work_dir/pydeps` so `pip install` writes packages there instead of `~/.local`.
- Sets `PIP_CACHE_DIR=work_dir/pip-cache` so pip doesn't touch `~/.cache/pip`.
- Appends `work_dir/pydeps` to `PYTHONPATH` so installed packages are importable immediately (originally prepended — changed in Patch 6).

**Why:** On this cluster, `/home/wangsaja` is not writable by the kernel process, so `pip install` in notebooks failed with `Permission denied: '/home/wangsaja'`. The Enroot/Docker paths already handle this via `_prep_workspace_dir`; this patch brings the local kernel path to parity.

**To revert:**
- Delete `_setup_pip_env`.
- In the `if not self.use_host_env_vars` branch, replace `merged = self._setup_pip_env(merged)` with nothing (keep `kwargs["env"] = merged` as-is).
- In the `else` branch, replace `kwargs["env"] = self._setup_pip_env(os.environ | self.extra_envs)` with `kwargs["env"] = os.environ | self.extra_envs`.

---

## Patch 5 — absolute work_dir (fixes nested tmp + numpy import failure)

**Files:** `src/hypotest/dataset_server.py`, `src/hypotest/env/interpreter.py`

**What:**
- `dataset_server.py` (`get_new_env_by_idx`): resolve `problem_dir` and `save_dir` to absolute paths with `.resolve()`.
- `interpreter.py` (`_setup_pip_env`): resolve `pydeps`/`pip-cache` to absolute before putting them on `PYTHONPATH`/`PIP_TARGET`.
- `interpreter.py` (`start`): pass an absolute `cwd` to the kernel (`str(self.work_dir.resolve())`).

**Why:** `server.yaml` sets `work_dir: tmp/` (relative). The relative path flowed verbatim into the kernel's `PYTHONPATH`/`PIP_TARGET` as `tmp/<run_id>/pydeps`. Because the kernel's cwd is itself the work_dir, that relative entry re-resolved against cwd into a doubled `work_dir/tmp/<run_id>/pydeps` — creating a nested `tmp/<run_id>/` directory and a malformed `sys.path` (bare `''` cwd entry + junk relative entry). numpy then failed with the misleading `ImportError: you should not try to import numpy from its source directory`, costing the agent ~14 wasted turns inventing `cd /tmp` / symlink / `sys.path` workarounds (see `traj_36.html`). Absolute paths eliminate the doubling and clean up `sys.path`.

**To revert:** drop the `.resolve()` calls in all three spots.

---

## Patch 6 — pydeps prepended to PYTHONPATH (so the model can install/upgrade libraries)

**Files:** `src/hypotest/env/interpreter.py`, `src/hypotest/env/interpreter_env.py`

**History:** This patch originally *appended* `pydeps` to the end of `PYTHONPATH` so the
curated `kernel_env` always won, working around a crash where a freshly pip-installed numpy
shadowed the ABI-matched kernel_env copy. That crash was a 3.13-vs-3.12 ABI mismatch, since
fixed at the source by **Patch 7** (kernel now launches under kernel_env's own interpreter).
Because appending also made every `pip install --upgrade <pkg>` a silent no-op (kernel_env
won), it has now been **reverted to prepend** so the model can actually install/upgrade libs.

**What (current):** `work_dir/pydeps` is at the *front* of `PYTHONPATH` in all three places it
is set:

- `interpreter.py` (`_setup_pip_env`): `PYTHONPATH = f"{pydeps_str}:{existing}"`.
- `interpreter_env.py` (enroot bash, local-node path): `export PYTHONPATH="$WORKDIR/pydeps${PYTHONPATH:+:$PYTHONPATH}"`.
- `interpreter_env.py` (enroot bash, `/data_workspace` path): `export PYTHONPATH="/data_workspace/pydeps${PYTHONPATH:+:$PYTHONPATH}"`.

The bash forms use `${PYTHONPATH:+:$PYTHONPATH}` to avoid a stray trailing/leading `:` (which
injects the CWD into `sys.path`) when `PYTHONPATH` is empty.

**Why:** pip installs (Patch 4) land in `pydeps`. With `pydeps` appended, anything already in
`kernel_env` shadowed the model's install, so upgrades had no effect. Prepending makes
`pydeps` win, so `pip install`/`pip install --upgrade` take effect. This is safe now because
Patch 7 makes pip run under kernel_env's 3.12 interpreter: installs are ABI-matched, and pip —
seeing kernel_env's packages as already installed — no longer re-drags the full dependency
tree into `pydeps` (the condition that caused the original prepend crash before Patch 7).

**Accepted tradeoff:** an *explicit* upgrade of a core kernel_env lib (e.g.
`pip install --upgrade numpy`) now shadows kernel_env for the whole kernel, which could break
other compiled kernel_env packages built against the old ABI. That is the desired capability
(working upgrades), not a regression — appending only "avoided" it by disabling upgrades.

**To revert:** swap the two `PYTHONPATH` operands back so `pydeps` comes last in all three
spots (and restore the bash `${PYTHONPATH:+$PYTHONPATH:}` prefix form).

---

## Patch 7 — launch local Python kernel under kernel_env's interpreter (fixes 3.13-vs-3.12 numpy)

**Files:** `src/hypotest/env/interpreter.py`

**What:** In `Interpreter.start()`, after creating the `AsyncKernelManager`, override the kernelspec's `argv[0]` to `Path(cfg.KERNEL_ENV_PATH)/"bin"/"python"` for the PYTHON language (guarded by `kernel_python.exists()`). Accessing `kernel_manager.kernel_spec` loads+caches the spec; mutating `argv[0]` in place is honored by the subsequent `start_kernel()`.

**Why:** The local path starts the kernel via `AsyncKernelManager(kernel_name="python")`, which resolves the *ambient* registered kernelspec — whose `argv[0]` is an absolute path to whatever interpreter launched the benchmark. Running `python src/hypotest/benchmark_agent.py` from the project `.venv` (Python 3.13) therefore started a **3.13** kernel, while `extra_envs` (interpreter_env.py:1369) injected kernel_env's **3.12** `site-packages` onto its `PYTHONPATH`. A 3.12-built numpy cannot load its C extensions under 3.13, producing the misleading `ImportError: you should not try to import numpy from its source directory`. (The `%%bash` `which python` shows 3.12 because `PATH` is overridden, masking that the *kernel process* is 3.13.) The enroot path never had this bug because it `exec`s `/app/kernel_env/bin/python` directly (interpreter_env.py:365, :808); this brings the local path to parity. Fix is independent of which venv launches the benchmark.

**Relationship to other patches:** This fixes the interpreter↔site-packages version mismatch *at the source*. **Patch 3 item 1** (host `PYTHONPATH` version-stripping) was a band-aid for the same class of mismatch and is now the one **revert candidate** — but it computes `current_pyver` from the *outer* process, so it's no longer load-bearing for this bug; leave it as harmless defense-in-depth until Patch 7 is verified on the server, then it can likely be removed. Patches 4/5/6 are orthogonal (pip target, absolute paths, pydeps ordering) and stay.

**To revert:** delete the `if self.language == utils.NBLanguage.PYTHON:` block in `start()`.

---

## Patch 8 — live per-step thinking log

**Files:** `src/hypotest/benchmark_agent.py`

**What:** Added `extract_thinking()` and a `ThinkingLogger(Callback)`, registered on the `RolloutManager` via `callbacks=[...]`. Its `after_transition` hook fires the moment each step completes, pulls the model's `<think>…</think>` text out of that step's action content, and rewrites `results_dir/thinking.json` immediately — so the file grows live, step by step, during the run. Entries are keyed `task_{idx}` (matching `rewards.json`) via a `_task_idx` contextvar set in `rollout()`; contextvars are copied into the tasks `RolloutManager` spawns, so the label survives `num_parallel > 1`.

**Why:** Wanted to see the agent's reasoning per step without post-processing `trajectories.pkl`. The thinking is inline in the action message content (the served vLLM endpoints run no reasoning parser, so `<think>` stays in `content`), so it can be parsed the same way `scripts/inspect_trajectory.py` does. True token-by-token streaming isn't possible here: lmi disables streaming whenever tools are passed (`lmi/llms.py` `call()` raises on tools+callbacks), and the agent always passes tools — so the thinking only exists once each step's LLM call returns. Per-step (one write when each step finishes) is the most live granularity available without bypassing lmi.

**Output shape (`results_dir/thinking.json`):**

```json
{
  "task_0": [
    {"step": 0, "thinking": "..."},
    {"step": 1, "thinking": "..."}
  ]
}
```

**To revert:**
- Delete `extract_thinking` and `ThinkingLogger`.
- Restore `rm = RolloutManager(agent)` (drop the `callbacks=[...]` arg).
- Drop the `_task_idx` contextvar and the `_task_idx.set(idx)` line in `rollout()`.
- Remove the now-unused imports (`contextvars`, `Environment`, `Agent`, `Callback`, `Transition`).

---

## Patch 9 — litellm noise suppression

**Files:** `src/hypotest/benchmark_agent.py` (top of `main()`), `src/hypotest/dataset_server.py` (top of `launch_server()`)

**What:** Two cosmetic fixes that silence harmless litellm logging noise (neither affects rewards, trajectories, or the thinking log):

1. **Callback limit** — `update_litellm_max_callbacks()` (from `lmi.utils`), called in both `benchmark_agent.main()` and `dataset_server.launch_server()`. Each `LiteLLMModel` created during rollouts (or for the rubric model) appends logging callbacks without deduping, so litellm's default cap of 30 is quickly exceeded, spamming `"Cannot add callback"` warnings (litellm#9792). This raises the cap. **The call is preceded by an explicit `import litellm.litellm_core_utils.logging_callback_manager` and wrapped in `try/except`:** `lmi` reaches that submodule via attribute access (`litellm.litellm_core_utils.logging_callback_manager.LoggingCallbackManager`), but some litellm versions (e.g. the one in `.venv`, Python 3.13) don't auto-load it, so the bare call raised `AttributeError: module 'litellm.litellm_core_utils' has no attribute 'logging_callback_manager'` and blocked server/benchmark startup. The explicit import populates the attribute; the `try/except` ensures this noise-suppression step can never block startup again.

2. **Pydantic serializer warnings** (`benchmark_agent.py` only) — `warnings.filterwarnings("ignore", message="Pydantic serializer warnings:", category=UserWarning, module="pydantic.main")`. `lmi`'s cost tracker (`lmi/cost_tracker.py`) runs every response through `litellm.cost_calculator.completion_cost`, which serializes litellm's strictly-typed `ModelResponse` (`choices` typed as `StreamingChoices`, `message` as `litellm.Message`). The actual object is a non-streaming `Choices` wrapping an `lmi`/aviary `Message` (5 fields), so pydantic warns on the type mismatch while still serializing correctly. Filter is scoped to `pydantic.main` + that exact message so unrelated warnings still surface. Cost is `0` for the custom vLLM endpoints anyway.

**Why:** Both warnings flood stdout on essentially every LLM call, drowning out the progress bar and real output, despite being benign.

**To revert:**
- In both files, delete the `try/except` block (the `import litellm.litellm_core_utils.logging_callback_manager` + `update_litellm_max_callbacks()`) and the `from lmi.utils import update_litellm_max_callbacks` import.
- In `benchmark_agent.py`, also delete the `warnings.filterwarnings(...)` call and the `import warnings`.

---

## Patch 10 — SOFTWARE_STACK_CAPABILITIES added to system prompt

**Files:** `src/hypotest/env/prompts.py`, `src/hypotest/env/config.py`

**What:** Added `SOFTWARE_STACK_CAPABILITIES` constant to `prompts.py` (item "6." in the environment capabilities section). In `config.py` (`model_post_init`), this string is appended — with a blank-line separator — to whichever of `CPU_ENVIRONMENT_CAPABILITIES` / `GPU_ENVIRONMENT_CAPABILITIES` is selected, so it appears in every system prompt regardless of GPU flag.

**Why:** Agents wasted turns attempting `pip install` for packages already in the container, or failing to attempt bioinformatics CLI tools (samtools, BLAST, etc.) at all. Explicitly listing pre-installed Python packages, R libraries, and command-line tools — plus the correct installation paths (`pydeps` via `!pip install`) — reduces unnecessary installs and lets the agent reach for the right tool on the first try.

**To revert:**
- Delete the `SOFTWARE_STACK_CAPABILITIES` constant from `prompts.py`.
- In `config.py`, remove `+ "\n\n" + prompts.SOFTWARE_STACK_CAPABILITIES` from the `environment_capabilities_prompt` assignment (and the comment line above it).

---

## Patch 11 — Structured JSON rubric scoring (per-criterion, deterministic aggregation)

**Files:** `src/hypotest/env/prompts.py`, `src/hypotest/env/interpreter_env.py`

**What:** Two related changes:

1. **`RUBRIC_SCORE_PROMPT`** (`prompts.py`): Replaced the `<score>…</score>` free-text instruction with a JSON output instruction. The model is now asked to return a `{"criteria": [...]}` object where each element has `"criterion"` (string), `"score"` (int), and `"justification"` (string) — one entry per rubric criterion. No total is requested; it is summed in Python.

2. **Structured parsing** (`interpreter_env.py`):
   - Added `CriterionScore` and `RubricScore` Pydantic models (before `ProblemInstance`).
   - In `_score_solution`, passed `output_type=RubricScore` to `rubric_model.call_single()`. `lmi` responds by switching to `response_format: {"type": "json_object"}` and injecting the JSON schema into the system prompt automatically.
   - Claude (and some other models) prepend chain-of-thought prose before the JSON even in `json_object` mode. The `try` block finds the first `{`, saves everything before it as `score_metadata["chain_of_thought"]`, and parses only the JSON suffix with `RubricScore.model_validate_json()`.
   - `raw_score` is now `sum(c.score for c in rubric_score.criteria)` — aggregated deterministically in Python.
   - Per-criterion breakdown is saved into `score_metadata["criteria"]` (list of dicts) and any prose preamble into `score_metadata["chain_of_thought"]`, so both appear in `score_info.json` and the trajectory pickle.

**Why:** The old approach relied on the model to both evaluate each criterion *and* sum the total, then emit it in a specific `<score>N</score>` format. If the model misspelled the tag or computed the sum incorrectly, parsing failed or the score was wrong. Structured output removes both failure modes: the model only assigns per-criterion integers, and Python sums them. The preamble-stripping fixes a `RetryError` observed in production where Claude emitted valid JSON but preceded it with reasoning prose, causing `model_validate_json` to fail on the full string.

**To revert:**
- In `prompts.py`, restore the last two lines of `RUBRIC_SCORE_PROMPT` to:
  ```
  Be scientifically rigorous. Reason through each criterion in the rubric and provide brief justification for your score. Do not assign partial score for any rubric item (i.e. 0.5 points).
  At the very end, provide an integer score based on the rubric, enclosed in <score>...</score> tags.
  ```
- In `interpreter_env.py`, delete the `CriterionScore` and `RubricScore` model classes.
- In `_score_solution`, remove `output_type=RubricScore` from `call_single`.
- Replace the `try` block body with:
  ```python
  raw_score = int(resp.text.split("<score>")[1].split("</score>")[0])
  self.state.raw_score = raw_score
  ```
  (removing the `json_start` extraction, `chain_of_thought`, `criteria`, and `first_wrong_step` lines).

---

## Patch 12 — first_wrong_step: identify the first notebook cell where the agent erred

**Files:** `src/hypotest/env/prompts.py`, `src/hypotest/env/interpreter_env.py`

**What:** Extended the structured rubric JSON (from Patch 11) with a second top-level field:

- **`RUBRIC_SCORE_PROMPT`** (`prompts.py`): Added a `"first_wrong_step"` key to the requested JSON response. The model is instructed to output the 0-based index of the first notebook cell (as labelled `### Cell N:` in the rendered notebook) where the agent made an error inconsistent with the rubric, or `null` if no such error exists. The instruction asks for the *earliest causal cell*, not the cell where the consequence first became visible.

- **`RubricScore`** (`interpreter_env.py`): Added `first_wrong_step: int | None` field to the model.

- **`_score_solution`** (`interpreter_env.py`): Saves `rubric_score.first_wrong_step` into `score_metadata["first_wrong_step"]`, so it appears in `score_info.json` and the trajectory pickle alongside the per-criterion scores.

**Why:** Wanted to pinpoint *where* in the agent's notebook the analysis went off-track, not just *whether* it did. `first_wrong_step` gives a cell-level signal for error localisation and could be used for process-supervision or reward shaping downstream. Cell indices from `view_notebook` are 0-based and correspond roughly to agent tool-call turns (each cell ≈ one `run_cell` invocation).

**To revert:**
- In `prompts.py`, remove the `"first_wrong_step"` paragraph from `RUBRIC_SCORE_PROMPT`.
- In `interpreter_env.py`, remove `first_wrong_step: int | None` from `RubricScore`.
- In `_score_solution`, delete the `self.state.score_metadata["first_wrong_step"] = ...` line.
- (The `chain_of_thought` extraction and `criteria` saving are part of Patch 11, not this patch.)

---

## Patch 13 — avg@k and pass@k via num_replications

**Files:** `src/hypotest/benchmark_agent.py`, `benchmark.yaml`

**What:** Added `num_replications: int = 1` to `BenchmarkConfig`. When set to `k > 1`, each problem index is rolled out `k` times (the server already handles repeated calls to `get_new_env_by_idx` via its `problem_counter`). Results are grouped by problem index and two metrics are computed:

- **avg@k** — mean of per-problem mean rewards across k runs, then averaged across problems.
- **pass@k** — fraction of problems where at least one of the k runs achieved reward `== 1.0`.

Trajectory IDs gain a `_rep{r}` suffix when `k > 1` (e.g. `task_0_rep0`, `task_0_rep1`) to avoid key collisions in `rewards.json`. When `num_replications=1` (default), IDs remain `task_{idx}` and the output labels change to `avg@1` / `pass@1` (equivalent to the old "Average reward" / "Fraction solved").

`benchmark.yaml` is set to `num_replications: 3` to compute avg@3 and pass@3.

**Why:** A single rollout per problem is noisy given the stochastic temperature-1.0 sampling. Running each problem k times and reporting avg@k / pass@k gives a more stable and informative evaluation signal — avg@k captures mean quality, pass@k captures whether the model can ever solve each problem.

**To revert:**
- Remove `num_replications: int = 1` from `BenchmarkConfig`.
- Restore `rollout(idx: int)` (drop the `rep` parameter and `suffix` logic; restore `trajectory.traj_id = f"task_{idx}"`).
- Restore the gather to `*[rollout(i) for i in range(len(client))]`.
- Replace the `problem_rewards` / `avg_at_k` / `pass_at_k` block with the original:
  ```python
  avg_reward = sum(rewards) / len(rewards)
  frac_solved = sum(1 for r in rewards if r == 1.0) / len(rewards)
  print(f"Average reward: {avg_reward:.2f}")
  print(f"Fraction solved: {frac_solved:.2f}")
  ```
- Remove `num_replications: 3` from `benchmark.yaml`.

---

## Patch 14 — include_protocol toggle (omit step-by-step protocol from task prompt)

**Files:** `src/hypotest/env/prompts.py`, `src/hypotest/env/interpreter_env.py`, `src/hypotest/dataset_server.py`

**What:** Added an `include_protocol: bool = True` flag that controls whether the problem's step-by-step protocol appears in the `<objectives>` block of the task prompt.

- **`prompts.py`**: Added `NO_PROTOCOL_MSG` constant — the fallback text shown when `include_protocol=False`: `"No step-by-step protocol was provided. Define your own analysis plan following Step 1 (Define Analysis Plan) in your system instructions before executing code."`
- **`InterpreterEnvConfig`** (`interpreter_env.py`): Added `include_protocol: bool = True` field.
- **`InterpreterEnv`** (`interpreter_env.py`): Reads `self.include_protocol` from `self.config.include_protocol` and substitutes `NO_PROTOCOL_MSG` for `self.problem.protocol` in `HYPOTHESIS_TASK_DESC.format(...)` when `False`.
- **`DatasetConfig`** (`dataset_server.py`): Added `include_protocol: bool = True` field. Flows into `InterpreterEnvConfig` automatically via `**self.config.model_dump()` in `get_new_env_by_idx`.

**Why:** Wanted to evaluate agents without the curated protocol to measure how much of the benchmark score is attributable to the step-by-step guidance vs. the agent's own analysis planning.

**Usage:** Set in `server.yaml` under `dataset:`:
```yaml
dataset:
  include_protocol: false
```

**To revert:**
- Delete `NO_PROTOCOL_MSG` from `prompts.py`.
- Remove `include_protocol: bool = True` from `InterpreterEnvConfig`.
- In `InterpreterEnv.__init__`, remove `self.include_protocol = self.config.include_protocol`.
- In `_reset` (around `HYPOTHESIS_TASK_DESC.format(...)`), replace `protocol=self.problem.protocol if self.include_protocol else NO_PROTOCOL_MSG` with `protocol=self.problem.protocol`, and remove the `NO_PROTOCOL_MSG` import.
- Remove `include_protocol: bool = True` from `DatasetConfig`.

---

## Patch 15 — per-criterion first_wrong_step (supersedes Patch 12's top-level field)

> **Superseded by Patch 19:** the judge no longer emits `first_wrong_step`; it is derived in Python from `relevant_steps`. The per-criterion *shape* described below still holds — only its source changed.

**Files:** `src/hypotest/env/prompts.py`, `src/hypotest/env/interpreter_env.py`, `scripts/fork_trajectory.py`, `scripts/inspect_trajectory.py`, `scripts/regrade.py`

**What:** Moved `first_wrong_step` from a single top-level rubric field (Patch 12) to a **per-criterion** field, and rewired the fork tooling to use it.

- **`RUBRIC_SCORE_PROMPT`** (`prompts.py`): The JSON response now has a single key `"criteria"`. Each criterion object carries its own `"first_wrong_step"` — the cell where the procedure went wrong *for that criterion specifically* (`null` when the criterion got full marks). The model is told it **must** provide a non-null `first_wrong_step` whenever a criterion does not receive full marks. The old top-level `"first_wrong_step"` key is gone.
- **`CriterionScore` / `RubricScore`** (`interpreter_env.py`): Added `first_wrong_step: int | None = None` to `CriterionScore`; removed `first_wrong_step` from `RubricScore`.
- **`_score_solution`** (`interpreter_env.py`): Dropped the `score_metadata["first_wrong_step"] = ...` line entirely. The per-criterion steps now live solely in `score_metadata["criteria"]` (each dict includes `first_wrong_step`). `score_info.json` no longer has a top-level `first_wrong_step`.
- **`fork_trajectory.py`**: Added `select_fork_criterion(criteria)` (picks the criterion with the smallest non-null `first_wrong_step`, returns `(cell, criterion)`) and `criterion_feedback(crit)` (builds the fork-point feedback note from that one driving criterion's name + justification). `main()` now reads the fork cell via `select_fork_criterion(info["criteria"])`, injects that criterion's feedback, errors out if no criterion flagged a step, and recomputes `new_fws` the same way. Replaced the old `extract_failed_criteria_feedback` (all score==0 criteria). The `--first-wrong-step` override path forks with no injected feedback (no single criterion to source it from).
- **`inspect_trajectory.py`**: `_extract_rubric` derives the overall (earliest) `first_wrong_step` from the per-criterion values when the top-level field is absent. Terminal and HTML rubric views print each criterion's first wrong step inline and relabel the overall one "Earliest wrong step".
- **`regrade.py`**: Updated its local mirror of the models to match (it only sums `.criteria`, so no behavior change).

**Why:** A single top-level `first_wrong_step` conflated independent rubric failures. Per-criterion localisation pinpoints where each specific criterion went wrong, and lets the fork pick the *earliest* failure across criteria and replay the policy from there with that criterion's targeted feedback.

**To revert:**
- In `prompts.py`, restore the two-key JSON instruction with a top-level `"first_wrong_step"` paragraph (see Patch 12).
- In `interpreter_env.py`, remove `first_wrong_step` from `CriterionScore`, add `first_wrong_step: int | None` back to `RubricScore`, and restore `self.state.score_metadata["first_wrong_step"] = rubric_score.first_wrong_step` in `_score_solution`.
- In `fork_trajectory.py`, restore `extract_failed_criteria_feedback(traj)`, read `first_wrong_step` via `info.get("first_wrong_step")`, set `new_fws = env.state.score_metadata.get("first_wrong_step")`, and delete `select_fork_criterion` / `criterion_feedback`.
- In `inspect_trajectory.py`, restore `_extract_rubric` to `return criteria, data.get("first_wrong_step")` and drop the per-criterion lines in the formatters.
- In `regrade.py`, restore the local `RubricScore.first_wrong_step` field.

---

## Patch 16 — task_style: "question" (rubric-graded open research questions, no accept/reject)

**Files:** `src/hypotest/env/prompts.py`, `src/hypotest/env/interpreter_env.py`

**What:** Added an additive `task_style: Literal["hypothesis", "question"] = "hypothesis"` field to `ProblemInstance`, plus a parallel set of prompts so the framework can run rubric-graded tasks that pose an open research question instead of a hypothesis to accept/reject. Default behavior (`task_style="hypothesis"`, the only value that previously existed) is unchanged byte-for-byte.

1. **`ProblemInstance.accepted`** (`interpreter_env.py`): Relaxed from a required `bool` to `bool | None = None`. A new `model_validator` (`check_accepted_required_for_hypothesis`) re-enforces the old requirement for `task_style="hypothesis"` (raises if `accepted` is missing), so existing hypothesis-style data must still supply it — only `task_style="question"` is allowed to omit it.
2. **`RESEARCH_QUESTION_TASK_DESC`** (`prompts.py`): Sibling of `HYPOTHESIS_TASK_DESC`. Frames the agent's task as answering a `<question>` rather than evaluating a `<hypothesis>`, and asks it to submit a "final answer" rather than a "final conclusion".
3. **`RUBRIC_SCORE_PROMPT_QUESTION`** (`prompts.py`): Sibling of `RUBRIC_SCORE_PROMPT`. Drops the `"The hypothesis is known to be {accepted}"` sentence entirely (there is no ground-truth boolean for an open question) and says "answer a research question" / "final answer" instead of "substantiate or reject a hypothesis" / "final conclusion". Both prompts now share their JSON-output-contract tail via a new `_RUBRIC_SCORE_PROMPT_TAIL` string constant, so the scoring/output schema (unchanged from Patch 15) can't drift between the two variants.
4. **`InterpreterEnv.reset()` / `_score_solution()`** (`interpreter_env.py`): Both pick `HYPOTHESIS_TASK_DESC`/`RUBRIC_SCORE_PROMPT` or `RESEARCH_QUESTION_TASK_DESC`/`RUBRIC_SCORE_PROMPT_QUESTION` based on `self.problem.task_style`. The question-style `_score_solution` branch omits the `accepted=` format kwarg (the template has no `{accepted}` placeholder to fill).

**Why:** Needed to load `phylobio/BiomniBench-DA` (open-ended data-analysis questions like "characterize cell subset distribution across tissues and identify tumor-specific types", graded by a 100-point expert rubric with no accept/reject ground truth) through this same `InterpreterEnv`/`Dataset`/rubric-grading pipeline, without touching the existing BixBench-style hypothesis accept/reject tasks or `scripts/generate_hypotheses.py`. User explicitly asked to *add* support rather than change existing behavior.

**To revert:**
- In `interpreter_env.py`: change `ProblemInstance.accepted` back to `bool = Field(alias="answer")` (no default), delete the `check_accepted_required_for_hypothesis` validator and the `task_style` field.
- Remove the `RESEARCH_QUESTION_TASK_DESC` / `RUBRIC_SCORE_PROMPT_QUESTION` imports; restore `HYPOTHESIS_TASK_DESC.format(...)` (unconditional) in `reset()` and the unconditional `RUBRIC_SCORE_PROMPT.format(..., accepted=self.problem.accepted, ...)` in `_score_solution()`.
- In `prompts.py`: delete `RESEARCH_QUESTION_TASK_DESC`, `RUBRIC_SCORE_PROMPT_QUESTION`, and `_RUBRIC_SCORE_PROMPT_TAIL`; restore `RUBRIC_SCORE_PROMPT` as a single inline string (its current content, header + tail, is unchanged — only the factoring changed).

---

## Patch 17 — BiomniBench-DA original A/B/C judge (invoked live for biomni rubrics; usable offline too)

**Files:** `src/hypotest/env/biomni_judge.py` (new), `src/hypotest/env/interpreter_env.py`, `src/hypotest/dataset_server.py`, `scripts/biomni_judge.py` (new)

**What:** Added the *original* BiomniBench-DA grading method (faithful port of `phylobio/BiomniBench-DA` `da-*/tests/llm_judge.py`) as an additive, auto-selected grading path. Its defining property vs. hypotest's default judge: the LLM only **chooses a level (A/B/C)** per rubric criterion; **Python** maps each letter to the criterion's rubric-defined points (`Levels: A=X B=Y C=0`), sums to 0–100, and clamps — eliminating judge arithmetic noise. The default hypotest judge (integer-per-criterion, Patches 11/12/15) is unchanged and still used for all non-biomni rubrics.

1. **`biomni_judge.py`** (new, pure/stdlib-only — single source of truth): `parse_rubric_levels(rubric)` (verbatim from the original), `is_biomni_rubric(rubric)` (true iff levels parse), `build_judge_prompt(rubric, trace, answer)` (the original "pick ONE level A/B/C" prompt), and `score_from_response(response_text, rubric)` → `(total_0_100, criteria, reasoning)` (maps letters→points; never raises — scores 0 on unparseable output, matching the original).
2. **`InterpreterEnvConfig.biomni_grading`** (`interpreter_env.py`): new `Literal["auto","biomni","hypotest"] = "auto"`. `"auto"` uses the biomni judge for biomni-style rubrics (detected by `is_biomni_rubric`) and the hypotest judge otherwise; `"biomni"`/`"hypotest"` force one method.
3. **`_score_solution`** (`interpreter_env.py`): computes `use_biomni` from the config + rubric, then branches. Biomni path builds the A/B/C prompt, calls the rubric model with **`output_type=None`** (plain generation, like the original), and scores via `score_from_response` (stores `score_metadata["criteria"]` as the biomni criteria dict + `["overall_reasoning"]`). Hypotest path is byte-for-byte the old code (RubricScore structured output). Both share the same downstream normalization/`correct`/`score_info.json` write. Also records `score_metadata["grading_method"]` = `"biomni"`/`"hypotest"`. `max_score` for biomni tasks is 100, so `raw/max` normalization and `correct = raw==max` work unchanged.
4. **`DatasetConfig.biomni_grading`** (`dataset_server.py`, + `Literal` import): same field so `server.yaml` can set it; flows into `InterpreterEnvConfig` via the existing `**self.config.model_dump()` splat (line ~129). Default `"auto"` → no config change needed to get biomni grading on biomni datasets.
5. **`scripts/biomni_judge.py`** (new): offline CLI that imports the shared core. Two modes — (a) re-grade saved hypotest runs (`--results <dir>`: extracts `<rubric>`/`<notebook>`/`<proposed-solution>` from each `score_info.json` prompt, grades with the biomni method, prints a hypotest-vs-biomni table, `--write` saves `biomni_score.json` per dir + a summary); (b) native BiomniBench-DA (`--rubric/--trace/--answer` files, like the original). Configurable `--model` (default flow keeps gpt-5/whatever, per the "keep configurable" decision), `--api-base`/`--api-key`, `.env` auto-load.

**Why:** User wanted BiomniBench-DA graded the *original* way when running the benchmark, while keeping normal hypotest tasks on the existing judge. Auto-detection by rubric structure means a biomni run (rubrics carry `Criterion N:` + `Levels: A/B/C`, `max_points=100`) is graded the biomni way with zero config, and BixBench-style hypothesis/question rubrics (no levels) keep the integer-per-criterion judge. The offline script covers re-grading already-completed runs and grading native biomni outputs.

**Caveat:** `parse_rubric_levels` only understands the A/B/C `Levels:` format. In `"auto"`, non-parseable rubrics correctly fall through to the hypotest judge live. But the offline `scripts/biomni_judge.py --results` **forces** the biomni method on every dir, so pointing it at a *mixed* results dir scores non-biomni tasks 0/100 — point it at a biomni-only run.

**To revert:**
- Delete `src/hypotest/env/biomni_judge.py` and `scripts/biomni_judge.py`.
- In `interpreter_env.py`: remove the `from .biomni_judge import ...` line, remove `biomni_grading` from `InterpreterEnvConfig`, and restore `_score_solution` to the single-path version — always build the hypothesis/question prompt, call `call_single(prompt, output_type=RubricScore, ...)`, and parse via `RubricScore.model_validate_json(...)` (drop the `use_biomni` branch, the `grading_method` metadata line, and the `overall_reasoning` branch).
- In `dataset_server.py`: remove `biomni_grading` from `DatasetConfig` and drop `Literal` from the typing import if now unused.

---

## Patch 18 — biomni judge emits hypotest's rich rubric, graded by A/B/C level (supersedes Patch 17's live path)

**Files:** `src/hypotest/env/prompts.py`, `src/hypotest/env/interpreter_env.py`, `src/hypotest/env/biomni_judge.py`, `scripts/inspect_trajectory.py`

**What:** Changed the live biomni grading path so it produces the **same rich per-criterion rubric as the hypotest judge** (`criterion` / `justification` / `relevant_steps` / `first_wrong_step` / `feedback`), with the single difference that the judge picks an **A/B/C `level`** per criterion instead of emitting an integer `score`. Python then maps each level to points via the rubric's `Levels: A=X B=Y C=0` table and aggregates the reward. This replaces Patch 17's live path (the minimal `build_judge_prompt` "pick a level + one-line reason" prompt and its `{criteria: {criterion_N: {level, reason}}, overall_reasoning}` dict output, scored by `score_from_response`). Patch 17's offline `scripts/biomni_judge.py` and its `build_judge_prompt`/`score_from_response`/`parse_rubric_levels`/`is_biomni_rubric` functions are left intact for offline use.

1. **`prompts.py`** — Added `_RUBRIC_LEVEL_PROMPT_TAIL`, `RUBRIC_LEVEL_PROMPT`, and `RUBRIC_LEVEL_PROMPT_QUESTION`, each **derived by `.replace()`** from the existing Patch-16 score prompts (no duplicated prompt text; the two stay in lockstep automatically). The only swaps: the `"score": integer …` bullet → a `"level": exactly one of "A", "B", or "C" …` bullet, and the closing `Do not include a score total …` line → `Do not include any points or totals; points are derived from your chosen levels programmatically.` Everything else (framing, `relevant_steps`/`first_wrong_step`/`feedback` instructions, GOOD/BAD examples) is identical to the hypotest prompt.
2. **`biomni_judge.py`** — Added `score_rich_levels(criteria, rubric) -> int`: maps each rich criterion's `level` → points via the existing `parse_rubric_levels` (matched to `Criterion N:` blocks **by position**), injects the computed `"score"` into each criterion dict in place, sums, clamps to [0, 100]. Unrecognized/empty level → lowest defined value for that criterion.
3. **`interpreter_env.py`** — Added `CriterionLevelScore` (same shape as `CriterionScore`, but `level: str` instead of `score: int`; reuses `StepEvidence`) and `RubricLevelScore`. Changed the import from `biomni_judge` to `is_biomni_rubric, score_rich_levels` (dropped `build_judge_prompt`, `score_from_response`). In `_score_solution`: the `use_biomni` branch now uses `RUBRIC_LEVEL_PROMPT` / `RUBRIC_LEVEL_PROMPT_QUESTION` (same hypothesis/question selection as hypotest), calls the rubric model with **`output_type=RubricLevelScore`** (structured output, no longer plain `output_type=None`), and parses via `RubricLevelScore.model_validate_json(...)` → `score_rich_levels(...)`. The parse `try` block was unified so both judges share the `json_start`/`chain_of_thought` extraction. Removed the `score_metadata["overall_reasoning"]` line (the rich schema has no `overall_reasoning`); `grading_method`, `criteria`, `raw_score`, normalization, and `correct = raw==max` are unchanged.
4. **`scripts/inspect_trajectory.py`** — Since every judge now emits the same rich *list* schema, the viewer has a **single** rubric path (no `--biomni` flag). The default `_extract_rubric` / `_html_rubric` / `_term_fmt_rubric` render every criterion field: `criterion`, the grade badge, `justification`, **`relevant_steps`** (per-cell ✓/✗ + note, via `_html_relevant_steps` / an indented terminal list), `first_wrong_step`, and `feedback`. Fields absent from a given criterion (e.g. `level`/`relevant_steps` on older pkls) render nothing.
   - **Pass/fail is graded by `level` when present, not by score** (`_crit_status` + `_LEVEL_STATUS`). This matters because the serialized judge output in `next_observation` carries only the A/B/C `level` — the numeric `score` is injected later into `score_info.json`, **not** the observation — so the old `score > 0` test marked *every* biomni criterion red (0 > 0 = False), including full-marks level-A ones. Now **A → green pass (✓)**, **B → amber partial (~)**, **C → red fail (✗)** (new `.crit-partial` CSS + a level-colored `.crit-level` chip); hypotest criteria still grade by `score > 0`. The rubric header (`_rubric_header`) shows a level tally (`A×4 B×5 …`) for biomni instead of a `0 pts` total, and the badge shows the level letter (not `0 pt`).
   - (An interim `--biomni` flag + separate dict-schema renderers were added and then removed once the schema converged; the pre-Patch-18 `benchmark_results_biomni/trajectories.pkl` used that old dict schema and no longer renders as a structured rubric — it falls back to raw JSON text.)

**Why:** The original biomni judge (Patch 17) discarded the process-supervision signal — no `relevant_steps`, `first_wrong_step`, or `feedback` — which the fork tooling (Patch 15) and trajectory viewer rely on. This keeps biomni's key property (the LLM only picks A/B/C, Python does the arithmetic — no judge-arithmetic noise) while restoring the full rich rubric, so biomni runs are inspectable and forkable exactly like hypotest runs.

**How to invoke:** unchanged from Patch 17 — set `biomni_grading` in `server.yaml` under `dataset:` (`auto` (default) auto-detects biomni rubrics via `is_biomni_rubric`; `biomni` forces the level judge; `hypotest` forces the integer judge). No config change is needed for BiomniBench-DA datasets since their rubrics carry the `Levels:` table.

**To revert (restore Patch 17's live path):**
- In `prompts.py`: delete `_RUBRIC_LEVEL_PROMPT_TAIL`, `RUBRIC_LEVEL_PROMPT`, `RUBRIC_LEVEL_PROMPT_QUESTION`.
- In `biomni_judge.py`: delete `score_rich_levels`.
- In `interpreter_env.py`: delete `CriterionLevelScore` / `RubricLevelScore`; restore the import to `from .biomni_judge import build_judge_prompt, is_biomni_rubric, score_from_response`; in `_score_solution`, restore the `use_biomni` prompt to `build_judge_prompt(self.problem.rubric, nb_content, solution)`, the call to `output_type=None if use_biomni else RubricScore`, and the biomni parse branch to `raw_score, criteria, overall_reasoning = score_from_response(resp.text, self.problem.rubric)` + `score_metadata["overall_reasoning"] = overall_reasoning` (moving the `json_start`/`chain_of_thought` extraction back inside the `else` branch).
- In `scripts/inspect_trajectory.py`: delete `_LEVEL_STATUS` / `_crit_status` / `_rubric_header` and restore the inline `passed = score > 0` grading + `total pts` header in `_term_fmt_rubric` / `_html_rubric`; remove the `relevant_steps` rendering (`_html_relevant_steps` + the terminal loop) and the level badge; drop the `.crit-partial` / `.crit-level` / `.rstep*` CSS (revert to score-only pass/fail rubric cards).
---

## Patch 19 — first_wrong_step derived in Python (supersedes Patch 15's judge-emitted field)

**Files:** `src/hypotest/env/prompts.py`, `src/hypotest/env/interpreter_env.py`, `scripts/regrade.py`, `scripts/inspect_trajectory.py`

**What:** The judge no longer emits `first_wrong_step`; Python computes it from the `relevant_steps` the judge does emit. Several judging rules changed alongside it.

- **`prompts.py`** (`_RUBRIC_SCORE_PROMPT_TAIL`, so all four variants inherit it):
  - The `"first_wrong_step"` bullet is **commented out**, preserved verbatim above the string for restoration. `"feedback"` now anchors on "the earliest entry in `relevant_steps` with `correct: false`" instead of referencing it.
  - `relevant_steps` entry order swapped to `step` → `note` → `correct`, so the judge states the observation before committing to a verdict (`StepEvidence` field order was swapped to match, since it drives structured-output generation order).
  - **Repair rule:** a step whose error was fixed later (cell re-run, or redone properly further down) is marked `"correct": true`. Only unrepaired problems standing at the end are `false`.
  - **Method-switch rule:** if the agent abandons an approach and satisfies the criterion another way, the superseded steps are `true` — a detour, not an error. Steps leading to a final method that is itself wrong stay `false`.
  - **Consistency rule:** full points requires every entry `correct: true`; any `false` means less than full points.
  - The tail was then condensed (~30% shorter) with all rules intact. **The two exact-string anchors used to derive `_RUBRIC_LEVEL_PROMPT_TAIL` (the `"score"` bullet line and the "Do not include a score total…" line) must stay byte-identical** — if they drift, `.replace()` silently no-ops and the biomni judge is told to emit `"score"` while `RubricLevelScore` expects `"level"`.
- **`interpreter_env.py`**:
  - `first_wrong_step` commented out on `CriterionScore` / `CriterionLevelScore` (old payloads still parse; the extra key is ignored).
  - New `parse_criterion_max_scores(rubric)` reads per-criterion maxima from either rubric family — biomni `Levels: A=X B=Y C=0` (highest level value) or hypotest `N. (X points)`. (**Patch 20** adds bixbench's `* X points:` bullets, without which the full-marks gate below never fired on that dataset.)
  - New `derive_first_wrong_step(criteria, rubric=None)` injects the earliest `relevant_steps` entry with `correct: false`, **nulling any criterion awarded full marks** (the judge does not reliably honour the consistency rule, so Python enforces it). Annotates in place and returns the same list. Without a parseable rubric the full-marks gate is skipped. Called on both grading branches with `self.problem.rubric`.
  - `submit_answer` emits the **derived** criteria as the `Rubric evaluation:` payload instead of the judge's raw JSON, so `inspect_trajectory` matches `score_info.json` exactly. This narrows Patch 3 item 3: the raw judge text is still saved in `score_metadata["response"]` → `score_info.json`, just no longer duplicated into the trajectory. Falls back to the raw text if no criteria parsed.
- **`regrade.py`**: mirrors both functions locally (per the file's existing no-heavy-imports convention) plus a `rubric_text(prompt)` helper that recovers the rubric from the saved prompt's `<rubric>…</rubric>` block.
- **`inspect_trajectory.py`**: `_derive_first_wrong_steps` fills the field in when a criterion **lacks the key** (old runs, or the raw-text fallback). It deliberately keys on key *presence*, not null-ness — since this patch an explicit `null` is a decision (full marks) and must not be overwritten. (**Patch 20** tags these viewer-side derivations `(derived)`, since the viewer has no rubric and so cannot apply the full-marks gate.)

**Why:** Asking the judge for `first_wrong_step` made it do bookkeeping that Python can do deterministically from `relevant_steps`, and the judge routinely contradicted itself — flagging incorrect steps on criteria it had just awarded full marks, which made fully-satisfied criteria drive forks. The repair and method-switch rules stop transient errors and abandoned approaches from becoming fork points, so a fork now lands on the earliest problem that actually survives to the end of the notebook.

**Consequences:**
- `fork_trajectory.py` is unchanged and works as before — it reads `first_wrong_step` off `score_info.json` criteria, which the derivation still populates. Fork *points* differ from pre-patch runs: self-corrected errors and full-marks criteria no longer drive a fork.
- Cell indices are unchanged (`view_notebook` labels `### Cell {idx}` 0-based; `fork_step_for_cell` expects 0-based).
- Criteria are matched to rubric rows **positionally**, as `biomni_judge.score_rich_levels` already does. If the judge emits more criteria than the rubric has rows, the extras skip the full-marks gate.

**To revert:**
- In `prompts.py`: re-insert the commented-out `first_wrong_step` bullet between `relevant_steps` and `feedback`, revert the `feedback` bullet to anchor on it, and drop the repair / method-switch / consistency sentences.
- In `interpreter_env.py`: uncomment `first_wrong_step` on both criterion models; delete `parse_criterion_max_scores`, `derive_first_wrong_step`, the `re` import, and the `parse_rubric_levels` import; unwrap both `derive_first_wrong_step(...)` calls in `_score_solution`; restore `submit_answer`'s payload to `score_metadata["response"]` (see the revert note inline).
- In `regrade.py`: delete both mirrored functions and `rubric_text`; restore `"criteria": [c.model_dump() for c in rubric.criteria]`.
- In `inspect_trajectory.py`: delete `_derive_first_wrong_steps` and its call in `_extract_rubric`.

---

## Patch 20 — bullet-form rubrics parse for the full-marks gate

**Files:** `src/hypotest/env/interpreter_env.py`, `scripts/regrade.py`, `scripts/inspect_trajectory.py`

**What:** Patch 19's full-marks gate silently no-opped on the bixbench rubrics — the dataset this repo actually benchmarks against — so criteria awarded full marks still got a `first_wrong_step`.

- **`interpreter_env.py`** — `_HYPOTEST_CRITERION_POINTS` matched only the numbered layout `N. (X points) …`. The `EdisonScientific/bixbench_hypothesis` rubrics are bullets — `* 1 point: …`, `* 5 points: …` — so `parse_criterion_max_scores` returned `[]`, every `max_pts` was `None`, and `derive_first_wrong_step` fell through to the earliest-incorrect-step rule for *every* criterion. The regex now accepts both layouts as alternatives capturing into groups 1 and 2 (`int(m.group(1) or m.group(2))` at the call site). The biomni `Levels:` branch is unchanged and still takes precedence.
- **`regrade.py`** — same change to the mirrored `_HYPOTEST_CRITERION_POINTS` / `parse_criterion_max_scores`, per the file's keep-in-sync note.
- **`inspect_trajectory.py`** — the viewer cannot apply the gate at all: the rubric text appears nowhere in a trajectory pkl (it lives only in the judge prompt, i.e. `score_info.json`, and mapping a traj to its run dir needs `map_trajs.py`'s fuzzy answer matching). So rather than a fake gate, `_derive_first_wrong_steps` now tags what it computes with `_fws_derived`, and both renderers append `(derived)` via the new `_fws_suffix(cr)` helper. A step shown without that suffix came from the grader and is authoritative; one with it may be a full-marks criterion the viewer could not gate.

**Why:** With the gate skipped, `score_info.json` — not just the viewer — recorded a wrong step on fully-satisfied criteria, and `fork_trajectory.py` picks the smallest non-null step across criteria, so a criterion that scored full marks could drive the fork. That is precisely the failure Patch 19 set out to eliminate; it just never took effect on the bullet-form rubrics.

**Verification:**
- On `archive/all-sonnet/results-trial/33b801bb-…-iter5`: scores `[1,0,1,1,1,0]`, `first_wrong_step` before `[None,11,None,24,None,11]` → after `[None,11,None,None,None,11]`. Only the bogus full-marks entry clears.
- Across 1413 archived `score_info.json` files, parsed criterion counts match judge criterion counts in 1412; 39 criteria archive-wide were full-marks-with-a-step and would now be nulled. The single mismatch (`archive/gpt-judge/forks-wo-protocol/task_23_rep0-fork_cell11/…`) is a 7-row rubric where the judge returned only 6 criteria — a judge omission, and the positional-matching hazard Patch 19 already documents, not a parser bug.

**Consequences:**
- Fork points shift again for bullet-form rubrics: a full-marks criterion can no longer be the earliest flagged one.
- Existing `score_info.json` files are **not** backfilled — re-run `scripts/regrade.py` (its `rubric_text(prompt)` recovers the rubric from the saved prompt) if you need old runs corrected.
- Trajectory pkls written by a server process started before Patch 19's `submit_answer` payload swap still carry the judge's raw JSON, which has no `first_wrong_step` key at all; those render as `(derived)` in the viewer. Restart the dataset server to get derived criteria embedded.

**To revert:**
- In `interpreter_env.py` and `regrade.py`: restore `_HYPOTEST_CRITERION_POINTS = re.compile(r"^\s*\d+\.\s*\(\s*(\d+)\s*points?\s*\)", re.MULTILINE)` and `int(m.group(1))` in `parse_criterion_max_scores`.
- In `inspect_trajectory.py`: delete `_fws_suffix`, the `c["_fws_derived"] = True` line in `_derive_first_wrong_steps`, and the two `{_fws_suffix(cr)}` interpolations in `_term_fmt_rubric` / `_html_rubric`.

---

## Patch 21 — judge ignores environment/tooling cells

**Files:** `src/hypotest/env/prompts.py`

**What:** Extended the `"relevant_steps"` bullet in `_RUBRIC_SCORE_PROMPT_TAIL` (so all four prompt variants inherit it) to exclude setup cells from the rubric evaluation: package installation and dependency resolution (pip, conda, `install.packages`, BiocManager), failed or retried installs, missing-package and import errors, version conflicts, and kernel restarts. The judge is told never to list them as steps and never to treat them as an error against a criterion — and, in the same breath, that if a package never installed it should still judge the criterion on the analysis that did run and on the absence of the analysis that never ran.

In the same pass the `"correct"` bullet was condensed (~470 → ~350 chars) with all four rules unchanged: end-state judging folded into the opening clause, the "Mark true/false…" scaffolding dropped, and the repair rule's parenthetical `(cell re-run, or the step redone properly further down)` replaced by "a later cell fixed it".

Neither of the two exact-string anchors used to derive `_RUBRIC_LEVEL_PROMPT_TAIL` was touched, so the biomni level prompts still inherit both edits (verified: all four variants contain the new text, and the score→level bullet swap still fires).

**Why:** Dependency wrangling is infrastructure noise, not scientific work. The judge was listing install failures in `relevant_steps` and marking them `correct: false`, which cost criteria points for a reason the rubric never asks about — and, since Patch 19 derives `first_wrong_step` from the earliest `correct: false` entry, made a failed `pip install` the fork point for a fresh policy rollout. The closing clause keeps the exemption narrow: a criterion is not excused merely because the tool it needed never installed.

**Consequences:**
- Fork points move off setup cells and onto the first genuine analysis problem.
- Scores may rise slightly on runs that fought their environment; a criterion whose analysis never ran still scores 0.

**To revert:** In `prompts.py`, cut the sentences from "Omit environment and tooling cells entirely…" through "…not on the installation attempts." out of the `"relevant_steps"` bullet, leaving the bullet ending at "Never leave this empty, even at full marks. Each entry has:".

---

## Patch 22 — agent installs contained in a disposable kernel env (fixes agents writing into `.venv`)

**Files:** `src/hypotest/env/prompts.py`. Also out-of-tree: `.env` (new `KERNEL_ENV_PATH`), new env at `/mlbio_scratch/wangsaja/kernel_env`, shim at `kernel_env/bin/uv`.

**What:** Four changes, one causal story.

1. **`.venv` rebuilt.** `uv sync` re-pinned to the original uv-managed CPython 3.13.10. A plain `uv sync` silently rebases onto conda `bixbench`'s 3.12, so the interpreter must be passed explicitly.
2. **Disposable kernel env.** `/mlbio_scratch/wangsaja/kernel_env_conda` — a micromamba conda env reproducing the Dockerfile's `/app/kernel_env` on linux-64: Python 3.12 stack, R 4.3.3 + IRkernel + rpy2 3.5.11 + Seurat/tidyverse/WGCNA/coloc, the Bioconductor set (DESeq2, limma, clusterProfiler, EnhancedVolcano, …), and the bioconda CLI tools (BLAST, samtools, SPAdes, MAFFT, IQ-TREE, FastQC, Trim Galore, HMMER, MMseqs2, metaEuk), plus `rdata`/`pyreadr` via pip. Built by `/mlbio_scratch/wangsaja/build_kernel_env_conda.sh` using `/mlbio_scratch/wangsaja/bin/micromamba` (static binary, no root). Wired by `KERNEL_ENV_PATH` in `.env`. This is the first thing on this host to satisfy Patch 7's `kernel_python.exists()` guard, which was previously inert because `/app/kernel_env` does not exist outside the container — so Patch 7's `argv[0]` pin only starts working now. An earlier uv-venv version of this env remains at `/mlbio_scratch/wangsaja/kernel_env` as rollback; it has the Python stack but no R.

   Built as a **single solve**, not the Dockerfile's staged installs. The `r-base` clobber that PATCH 2's `conda-meta/pinned` file works around only occurs when a later `mamba install` re-solves and bumps R; with one transaction there is no later step, so no pinned file is needed. The Dockerfile's PATCH 1 (ARM Miniconda), PATCH 4 (`r-coloc` has no aarch64 build) and PATCH 6 (`chempy`/`pyodesys` on ARM) are all aarch64 workarounds and do not apply on x86_64 — `r-coloc` installs normally here.
3. **`uv` shim** at `kernel_env/bin/uv`. `kernel_env/bin` is first on the kernel's PATH (`interpreter_env.py`, `extra_envs["PATH"]`), so it intercepts `uv`. It drops `--system` and appends `--python $KERNEL_ENV/bin/python` to every `uv pip` call; non-`pip` subcommands pass through.
4. **`SOFTWARE_STACK_CAPABILITIES`** in `prompts.py`: removed the `!uv pip install --system <pkg>` instruction; corrected the `curl` claim (both `wget` and `curl` are present); gave the CLI tools their real binary names (`spades.py`, `iqtree`, `trim_galore`, `hmmsearch`, `mmseqs`, and `gatk3` — not `gatk`); asserted that the R stack and CLI tools are installed and must not be re-installed; and added two worked recipes — reading `.rds`/`.RData`, and `pydeseq2` differential expression.

   Both recipes were added because trajectory analysis showed agents burning steps rediscovering them. `pydeseq2` 0.5.2 takes `counts=`, not `count_data=`; in the 5-rollout `archive/trial` run, task_0 hit `unexpected keyword argument 'count_data'` **8 times** and scored 0.3. Both recipes are verified against the installed versions and against a real capsule `.rds` file.

   **Any literal `{`/`}` in these prompts breaks every rollout.** `config.py:86-88` folds `SOFTWARE_STACK_CAPABILITIES` into `environment_capabilities_prompt`, and `interpreter_env.py:1476` calls `.format(job_timeout=...)` on it; `str.format` reads a stray brace as a replacement field. A first version of the `.rds` recipe used a dict comprehension (`{k: np.asarray(v) for k, v in ...}`), which raised `KeyError: 'k'` in `reset()` and failed all 5 rollouts at 0%. Both recipes are now written brace-free rather than escaped as `{{`/`}}`, which would render wrong for any caller that does not `.format()`. `tests/test_prompts.py` guards this: it asserts `{job_timeout}` is the only placeholder across all three profiles.

**Why:** On 2026-07-28 a rollout ran `!uv pip install --system pydeseq2` (`archive/all-sonnet/results/4ef3fcd8-…-iter1`) after a plain `pip install` appeared to do nothing — because Patch 4's `PIP_TARGET` had silently redirected it to `pydeps`. uv ignores `PIP_TARGET` entirely, and `--system` deliberately bypasses virtualenvs, so the install landed in the project `.venv`. It left three numpy `dist-info` dirs (1.21.6, 2.4.1, 2.5.1) and dangling NFS silly-rename symlinks in `numpy.libs`, breaking `import numpy` for both the benchmark and every script using `.venv`. The prompt was the proximate teacher: it named that exact command. Patch 4 was never a boundary — it is an env-var default on one tool, and the kernel runs as the invoking user with write access to every env on PATH.

**On the R stack:** `force_python: bool = True` (`dataset_server.py:43`, not overridden in `server.yaml`) forces `NBLanguage.PYTHON` for every task, so the `ir` kernel is never requested. The upstream dataset is 30 Python / 21 R; all 51 run, but R-native tasks execute in Python. In `archive/all-sonnet` they score close on average (mean 0.552 vs 0.572) but fail outright more often (11% zero-reward reps vs 2%), with two R tasks at 0/3 across all reps. One of them (`0923d260`) has DESeq2 `.rds` inputs and its `pydeps` contains an agent-installed `rdata` — hence change 4's reading recipe. R is now installed, but reachable only via `rpy2`/`%%R`/`!Rscript` from Python cells; running the 21 R-native tasks *as* R additionally requires setting `force_python: false` in `server.yaml`.

**Consequences:**
- Agent installs land in `kernel_env` or per-rollout `pydeps`, not `.venv` or the conda env.
- Containment is PATH precedence, not enforcement. Absolute paths, `conda install`, and direct filesystem writes still escape. Only containerizing the kernel makes this a hard boundary.
- `kernel_env` is disposable by design: rebuild it rather than repairing it.
- `pydeseq2` resolves to 0.5.2, not latest. 0.5.4 requires numpy ≥ 2.5, and numba caps numpy at ≤ 2.4, which breaks `scanpy`, `muon`, and `umap`. Keep `numpy==1.26.4` pinned when adding packages here, and re-check those three imports afterwards.
- `gatk=3.8` installs its binary as `gatk3`, not `gatk`.
- `torch` comes from pip (CPU wheels), matching `Dockerfile:252` — it is not a conda package here. Importing `torch` *before* `numba`/`scanpy` raises `OSError: Could not find/load shared object file 'libllvmlite.so'` unless `LD_LIBRARY_PATH` includes `KERNEL_ENV_PATH/lib`. The kernel sets that (`interpreter_env.py:1512`), so agents are unaffected; bare `python -c` probes outside the kernel are not.
- `keras=3.11.2` is installed but unimportable — Keras 3 defaults to the TensorFlow backend and TensorFlow is not installed (also true of the Dockerfile's env). `datasets` was omitted; the Dockerfile pins `datasets=2.2.1`. Neither is advertised in the system prompt.
- The conda env has both `lib/python3.1` (a symlink) and `lib/python3.12`. `interpreter_env.py` takes `sorted(glob("python3.*"))[-1]`, which correctly yields `python3.12`; code that takes the *first* match instead would land on the symlink.

**To revert:** Point `KERNEL_ENV_PATH` back at `/mlbio_scratch/wangsaja/kernel_env` (the uv-venv build, Python-only), or remove it entirely — the kernel then falls back to whatever `python` is first on PATH and Patch 7 goes inert again. `/mlbio_scratch/wangsaja/kernel_env_conda` can be deleted and rebuilt from `build_kernel_env_conda.sh`. Restore the four `prompts.py` edits listed above.

---

## Patch 23 — judging is per-benchmark: a judge registry, plus HeurekaBench's own judge

**Files:** new `src/hypotest/env/judges/{__init__,base,hypotest_judge,biomni,heureka}.py`, new `src/hypotest/env/problem.py`, new `tests/test_judges.py`; `src/hypotest/env/interpreter_env.py`, `src/hypotest/dataset_server.py`, `scripts/convert_heurekabench.py`, `scripts/convert_bioagent_bench.py`, `capsules/heurekabench/problems_heurekabench_oe_full.jsonl`.

**What:** Grading protocol is now a property of the *benchmark*, expressed as one registered function per benchmark, instead of a branch inside `_score_solution`.

1. **The registry (`judges/base.py`).** A judge is `async (JudgeContext, LiteLLMModel) -> JudgeResult`, registered with `@judge("name")`. `JudgeContext` carries the problem, the rendered notebook, and the submitted answer; `JudgeResult` carries `raw_score`, `max_score`, per-criterion dicts, prompt/response metadata, and an optional `correct` override (default: full marks). The judge owns its prompt, its response schema, **how many LLM calls it makes**, and how labels/levels become points. `call_json` is the shared one-call helper (send, brace-slice the JSON, validate, capture reasoning/chain-of-thought). Adding a benchmark = one new module + one import line in `judges/__init__.py`; no existing file changes and there is no dispatch table to extend.
2. **Judge selection (`resolve_judge`).** `ProblemInstance.judge` (new field, set by the converter) → `InterpreterEnvConfig.judge` (run-wide override) → legacy rubric sniffing. The sniff is unchanged (`Levels: A=X B=Y C=0` → biomni, else hypotest), so datasets predating the field grade exactly as before. `biomni_grading: Literal["auto","biomni","hypotest"]` is **replaced** by `judge: str` on both `InterpreterEnvConfig` and `DatasetConfig`.
3. **Existing judges ported verbatim.** Patch 17's two paths became `judges/hypotest_judge.py` (integer points per criterion, summed) and `judges/biomni.py` (A/B/C levels, `score_rich_levels` maps letters→points). Same prompts from `prompts.py`, same schemas, same arithmetic — only their location changed.
4. **HeurekaBench is now a real judge**, not rubric prose. `judges/heureka.py` registers `heureka`. Following biomni's principle — *the LLM labels, Python does the arithmetic* — the model decomposes each sub-question's GT answer into atomic facts and labels each `PRESENT`/`PARTIAL`/`MISSING`/`INCORRECT`, then `_band()` computes the 0-5 G-Eval rating from the label counts. Fact labels land in `score_info.json`, so a rating is auditable fact by fact. **Only the open-ended split is covered** — HeurekaBench's MCQ split has no judge, and the converter leaves `judge` unset on MCQ problems so they keep falling through to hypotest's integer judge reading `MCQ_RUBRIC_PREAMBLE`, exactly as before this patch.
5. **Schemas and `ProblemInstance` moved out of `interpreter_env.py`** — the rubric schemas into the judge module that owns each, `ProblemInstance` into `env/problem.py`, and `derive_first_wrong_step`/`parse_criterion_max_scores` into `judges/base.py`. All are re-exported from `interpreter_env`, so every existing `from ...interpreter_env import X` still resolves. The judges package deliberately imports only pydantic/lmi and the pure-python rubric parsers, never `interpreter_env` — so `scripts/regrade.py` and `scripts/biomni_judge.py` can drop their mirrored copies (not done in this patch).
6. **`_score_solution` shrank from ~90 lines to ~35** and holds only what is shared across benchmarks: resolve the judge, call it, derive `first_wrong_step`, normalize/clamp the reward, write `score_info.json`. The `finally`-writes-`score_info` semantics of Patch 17 are preserved, including on a failed parse before tenacity retries.

**Why:** Three judging protocols already existed in three different *kinds* of place — a Python module (biomni), string constants (hypotest), and a converter's rubric preamble (HeurekaBench). Each new benchmark meant editing the prompt constants, the schema, the aggregation, and the sniffing branch, all inside one method. HeurekaBench in particular could only express its atomic-fact protocol as instructions to an integer-emitting judge, which left the fact decomposition invisible in the output and the 1-5 rating up to the model's arithmetic — the exact problem `biomni_judge.py` was written to remove.

**Verification:**
- `tests/test_judges.py` (16 tests): selection precedence (problem > config > sniff), every registered name resolves, unknown names raise, both legacy rubric families still sniff to their original judge, and the 0-5 band table.
- `TestRubricGrading` (4 live gpt-5-mini rollouts, the default-judge regression) passes unchanged through the new path.
- Live `heureka` run against `problems_heurekabench_oe_full.jsonl[0]` with a synthetic notebook + answer, judged by `anthropic/claude-sonnet-4-6`: sub-q1 decomposed into 7 atomic facts (5 PRESENT, 2 MISSING) → band 4; sub-q2 unanswered → 0; total 4/10, `score_info` criteria carry the per-fact labels.
- Full suite still collects (114 tests + 16 new).

**Consequences:**
- **`server.yaml` breaking change:** `biomni_grading: X` must become `judge: X`. It is a `str` now, not a `Literal`, so an unregistered name fails at grading time with `unknown judge 'X'; registered: [...]` rather than at config-parse time.
- HeurekaBench numbers from this judge are **not comparable** to numbers from grading the same rubrics with the hypotest judge: bands are now mechanical, so identical labels always yield an identical rating. There is nothing to re-grade (no HeurekaBench runs existed when this landed).
- `capsules/heurekabench/problems_heurekabench_oe_full.jsonl` was tagged in place with `"judge": "heureka"` (41/41 rows). Without the tag it would have fallen through the sniff to the hypotest judge.
- `OE_RUBRIC_PREAMBLE` in `convert_heurekabench.py` is now redundant with the judge prompt — its label definitions match, but its "Then award points" table describes work the judge no longer does. Harmless (the response schema has no points field), and left in place so the already-generated jsonl stays valid. Trim it on the next regeneration.
- `scripts/regrade.py` and `scripts/inspect_trajectory.py` still carry their mirrored schema/parsers; they now duplicate `judges/` rather than `interpreter_env`, and can import the real thing whenever someone wants to.

**To revert:** `git checkout` the five new modules away, restore `ProblemInstance`/the four rubric schemas/`derive_first_wrong_step`/`parse_criterion_max_scores` into `interpreter_env.py` (they moved unmodified), restore `biomni_grading` on both configs, and put back the `use_biomni` branch in `_score_solution` (Patch 17's version). Then drop the `"judge"` key from the HeurekaBench jsonl and from `convert_heurekabench.build_problems`.

---

## Patch 24 — deterministic (non-LLM) scoring, and bioagent-bench's own scorer

**Files:** new `src/hypotest/env/judges/bioagent.py`; `src/hypotest/env/judges/{base,__init__,hypotest_judge,biomni,heureka}.py`, `src/hypotest/env/interpreter_env.py`, `src/hypotest/dataset_server.py`, `scripts/convert_bioagent_bench.py`, `scripts/{regrade,regrade_forks,biomni_judge}.py`, `capsules/bioagent-bench/problems_bioagent_bench.jsonl`, `tests/test_judges.py`, `tests/test_interpreter_env.py`.

**What:** A judge may now score with no LLM at all, and bioagent-bench uses that path.

1. **Judges can run without a model (`judges/base.py`).** `JudgeFn`'s model parameter is `LiteLLMModel | None`; `@judge(name, needs_model=True)` records whether a judge needs one; `JUDGES` holds a `Judge` NamedTuple (`name`, `fn`, `needs_model`) and `resolve_judge` returns it rather than a `(name, fn)` tuple. The three LLM judges assert the model is present.
2. **`JudgeContext` gained `work_dir` and `truth_dir`.** LLM judges read the rendered notebook; a deterministic scorer reads the files the agent actually produced and compares them to an answer key. Both default to `None`, so nothing else had to change.
3. **`truth_dir` plumbing.** `InterpreterEnv.__init__` takes it as a constructor arg (like `save_dir`). `Dataset.get_new_env_by_idx` resolves `<capsule_dir>/_truth/<input_data_path>` — the layout `stage_bioagent_capsules.py` already writes — and passes it only when it exists. It is never copied into the workspace, so the agent cannot read it. Benchmark-agnostic: any converter that stages a `_truth/<id>/` gets it.
4. **`submit_answer` no longer skips scoring whenever `rubric_model is None`** — it skips only when the *resolved judge* needs a model. The matching `assert` at the top of `_score_solution` is gone. A bioagent-bench run needs no rubric model configured at all.
5. **`judges/bioagent.py`** (`needs_model=False`) transcribes upstream's `scoring.py`: one check per task_id, run over every table the agent wrote under `results/`, compared against `_truth/<task_id>/`. Returns 0 or 1 out of 1 — `JudgeResult.max_score` overrides the problem's `max_points` of 10, so reward normalizes to 0.0/1.0 and is directly comparable to upstream's published table. Stdlib `csv` only, no pandas, so the judges package stays importable by the offline scripts. Matching policy is **tolerant on column naming, strict on values**: most checks intersect normalized cell values rather than requiring a named column, because agents do not reproduce the truth files' headers and upstream's checks are value comparisons anyway.
6. **Wiring.** `convert_bioagent_bench.py` emits `judge: "bioagent"` and its docstring no longer claims hypotest has no non-LLM reward path; the 9 already-generated problems were tagged in place.
7. **Offline re-graders guarded.** `regrade.py`, `regrade_forks.py` and `biomni_judge.py` all replay `score_info["prompt"]` through a different model. Deterministically-scored runs have no prompt, so those entries are now skipped (with a count printed in `regrade.py`) instead of raising `KeyError`.

**Why:** bioagent-bench's real reward is `float(deterministic_match)` from a per-task Python function, but hypotest could only score through the rubric model. `convert_bioagent_bench.py` worked around that by restating each deterministic check as a prose question for an LLM to grade (criterion 1, 6 of 10 points) — a faithful-ish proxy that still put a language model between a CSV diff and the reward — and skipped `giab` entirely. Patch 23's registry already allowed a judge to make zero LLM calls; the only things missing were file access on the context and a way to say "this judge needs no model".

**Verification:**
- **Self-consistency**, 9/9: each task's own truth file, handed in as the agent's output, satisfies its check. This is the test that catches header/parsing mistakes, and it is parametrized in `tests/test_judges.py::TestBioagentChecks`.
- **Negative sweep**, 9/9: a decoy table carrying every column name any check reads, with values matching no truth file, fails every check; so does an empty `results/`.
- **No-model end-to-end** (`TestDeterministicGrading`): a real `InterpreterEnv` with `rubric_model=None`, agent writes the table from a notebook cell → `reward == 1.0`; no output → `0.0`. `score_info.json` records `grading_method: "bioagent"`, `max_score: 1`, and carries no `prompt` key.
- 53 tests pass including the 4 live `TestRubricGrading` gpt-5-mini rollouts (LLM path untouched); full suite collects 141.

**Consequences:**
- bioagent-bench reward becomes binary 0/1 instead of graded /10. **Not comparable to any existing bioagent-bench run**; directly comparable to upstream's published pass/fail table.
- `single-cell` and `transcript-quant` get strictly harder: the two rubric deviations recorded in `convert_bioagent_bench.py:41-49` (ignore the cluster number; sample transcripts instead of all 278) are *rubric* concessions to LLM grading. The scorer implements upstream's exact condition in both cases.
- The rubric on these problems is now unused by default. It is kept as provenance and as the A/B fallback — `judge: hypotest` in `server.yaml` forces the old graded-/10 path back on the same problems.
- `score_info.json` for these runs has no `prompt`/`response`/`criteria[].relevant_steps`, so `fork_trajectory.py`'s fork-point machinery has nothing to key on. Forking a deterministically-scored run is not supported.
- `JudgeContext` gained fields and `resolve_judge` changed return type — any out-of-tree judge or caller needs updating.

**Not done — `giab`:** still in `SKIP_TASKS`. Its truth VCF is not staged, `hap.py` is not on PATH, and its F1 is continuous where `JudgeResult.raw_score` is an int. When picked up: add a `check_giab` that shells out to `hap.py`, and either threshold the F1 or widen `raw_score` to `int | float`.

**To revert:** delete `judges/bioagent.py` and its import in `judges/__init__.py`; restore `JUDGES: dict[str, JudgeFn]`, the plain `judge(name)` decorator and `resolve_judge`'s `(name, fn)` return; drop `work_dir`/`truth_dir` from `JudgeContext`, `truth_dir` from `InterpreterEnv.__init__` and `Dataset.get_new_env_by_idx`; restore the unconditional `rubric_model is None` skip in `submit_answer` and the `assert` in `_score_solution`; drop the `"judge"` key from `convert_bioagent_bench.build_problem` and from the bioagent jsonl. The three script guards are harmless and can stay.

---

## Patch 25 — BioMysteryBench judge (final-answer-only correctness AND an anti-cheat check)

**Files:** new `src/hypotest/env/judges/biomystery.py`, new `scripts/convert_biomysterybench.py`, new `capsules/biomysterybench/problems_biomysterybench.jsonl`; `src/hypotest/env/judges/__init__.py`, `tests/test_judges.py`.

**What:** Support for `Anthropic/BioMysteryBench-full` (v11, 90 problems: 73 human-solvable, 17 hard), as a registered judge plus a converter.

The dataset ships its grading criterion per problem in the `answer_rubric` column, and all 90 end with the same sentence: *"Score 1.0 if the model did not cheat AND got the answer correct. Score 0 otherwise."* Two properties define it, and the judge is built around both:

1. **Correctness is judged on the FINAL ANSWER, not the path.** Anthropic's write-up is explicit that the benchmark grades "on their final answer, rather than the path the model took to get there" — the exact opposite of hypotest's own judge, which scores the notebook's procedure. So `biomystery` is a **two-call** judge: call 1 sees the question, the rubric and the submitted answer and *nothing else*; the notebook is withheld structurally rather than by asking one judge to ignore what is in front of it. Call 2 decides only whether the agent cheated, which is the one judgment that genuinely needs the transcript.
2. **Scoring is binary and conjunctive.** Python ANDs the two booleans (`passed = answer_correct and not cheated`) rather than asking a model to apply the rule — the same "the LLM labels, Python does the arithmetic" split as `biomni.py` and `bioagent.py`. `JudgeResult.max_score=1`, so reward normalizes to 0.0/1.0.

The cheating policy is transcribed from the dataset's README "Rules" and the v11 CHANGELOG "Grading rule": accession lookups (GEO/SRA/ENA/BioProject) to identify the source dataset, publication or study metadata are disallowed, as is otherwise reverse-identifying the dataset; standard bioinformatics database use (gene ID lookup, sequence annotation, reference genome download, BLAST) is allowed; and **recalling from memory is explicitly not cheating**, even when it includes the source publication. The prompt requires positive evidence and resolves ambiguity to "not cheated".

`scripts/convert_biomysterybench.py` maps `problems.csv` → jsonl: `question` → `hypothesis`, `answer_rubric` → `rubric`, `max_points=1`, `judge: "biomystery"`, `task_style: "question"`, `human_solvable` and `allowed_domains` into metadata. `--extract` unpacks `data/<id>.zip` into per-problem capsules (~145 GB across 90); the default run writes only the jsonl.

**Why:** Anthropic has published no grader prompt (the dataset is gated and its harness is not public), so the prompts here are written to the dataset's own stated criterion rather than copied. The parts that *are* specified — the binary conjunctive rule, the narrow cheating definition, and final-answer-only grading — are implemented literally, which is what "faithful" can mean here.

**Verification:**
- Converter reproduces the CHANGELOG's v11 split exactly: 90 problems, **73 human-solvable / 17 hard**. All 90 rubrics carry the expected scoring sentence (the converter warns if they do not, as a release check).
- Live truth table on real problem `hb002` (answer: *Bacillus licheniformis*), judged by `anthropic/claude-sonnet-4-6`: correct+clean → **1**; wrong+clean → **0**; correct+cheated (notebook does an `esearch` on an SRA accession) → **0**; hedged among three candidates → **0**.
- Unit tests with a stub model (no network): all four conjunction cases; **the correctness prompt provably does not contain the notebook** and the cheat prompt provably does not contain the answer key; metadata carries both calls.
- 56 tests pass.

**Consequences:**
- `protocol` is deliberately empty on these problems — upstream shows the model the question and the extracted data files, nothing else. Run with `include_protocol: false`.
- **Do not force `judge: hypotest` on this dataset.** That judge grades the notebook's procedure against a rubric that is really an answer key, which is neither the benchmark's metric nor comparable to it.
- Two LLM calls per rollout instead of one; the second is short (notebook + policy, no rubric).
- `allowed_domains` is recorded but **not enforced** — hypotest has no per-problem network policy. The anti-cheat check is what catches disallowed lookups, which is also how the benchmark itself defines the rule.
- Capsules are not staged by default. `problems_biomysterybench.jsonl` is written and valid, but a run needs `--extract` first.

**To revert:** delete `judges/biomystery.py`, its import in `judges/__init__.py`, `scripts/convert_biomysterybench.py`, the generated jsonl, and `TestBiomysteryJudge` in `tests/test_judges.py`.

---

## Patch 26 — negative rubric level values parse (BiomniBench-DA penalty criteria)

**Files:** `src/hypotest/env/biomni_judge.py`, `scripts/regrade.py`, `tests/test_judges.py`.

**What:** `parse_rubric_levels` required `\d+` for a level's point value, so a negative value did
not parse. Every one of the **50** BiomniBench-DA rubrics ends with a *Source Reliability* penalty
criterion declaring `Levels: A=0 B=-5 C=-10`. The header regex matched greedily up to the first
negative and stopped, capturing only `"A=0 "`, so the criterion parsed to `{"A": 0}`.

Consequences of that, both now fixed by the same change:

1. **The penalty never applied.** `score_rich_levels` found the judge's chosen level (`B`) absent
   from `{"A": 0}` and fell through to `min(allowed.values()) == 0`. Up to 10 points of inflation
   out of 100 on every BiomniBench-DA rollout.
2. **The Patch 19/20 full-marks gate misfired on that criterion.** `parse_criterion_max_scores`
   read its maximum as `max({"A": 0}.values()) == 0`, so a criterion scoring 0 satisfied
   `score >= max_pts` and `derive_first_wrong_step` forced `first_wrong_step = None` even when the
   judge had marked a relevant step incorrect.

Four patterns gained `-?`: the `Levels:` header and its per-level `finditer` in
`parse_rubric_levels`, the legacy `[A] (N points)` fallback in the same function, and the two
malformed-response fallbacks in `score_from_response` — where a bare `(\d+)` against
`"total_score": -5` matched the digits *after* the minus and read it as **+5**. `scripts/regrade.py`
carries a mirrored copy of the parser and had the same defect in both of its patterns.

Nothing else changed. `score_rich_levels` already summed whatever points it was given, so a penalty
now simply subtracts; its existing `max(0, min(100, total))` clamp keeps the reward non-negative.
Level `A` on a penalty criterion is 0 points, which *is* full marks for it, so the full-marks gate
still correctly returns `None` there.

**Why:** found by the Patch 23/24/25 judge sweep (`judge-test/`). Rollout `c8894f11` was graded
level `B` on *Source Reliability* and scored 0 instead of -5, and the same criterion reported
`first_wrong_step: null` while carrying a step marked `"correct": false` — the two symptoms of one
missing character.

**Verification:**
- `TestRubricLevelParsing` (8 tests): negative levels parse in both rubric formats, the penalty is
  subtracted, level `A` costs nothing, the total still clamps at 0, the full-marks gate keeps
  `first_wrong_step` on a penalised criterion but not on a met one, `"total_score": -5` no longer
  reads as `+5`, and — against the shipped dataset rather than a synthetic rubric — all 50
  `capsules/biomnibench/biomnibench.jsonl` rubrics now retain their negative levels.
- Replayed on the two saved judge-sweep rollouts: `65 → 60` and `24 → 19` out of 100, with
  `first_wrong_step` on the penalty criterion going `None → 5` and `None → 21`. Every other
  criterion is unchanged.
- 61 tests pass (`tests/test_judges.py`, `tests/test_prompts.py`).

**Consequences:**
- **BiomniBench-DA scores drop by 0-10 points per rollout** wherever the judge chose `B` or `C` on
  the penalty criterion. Existing BiomniBench-DA numbers are inflated and not comparable to numbers
  from this commit onward; `scripts/regrade.py` can recompute them from saved `score_info.json`
  without re-running the judge.
- An *unrecognized* level on a penalty criterion now costs `min(...) == -10` rather than 0, per the
  existing "unrecognized level → lowest defined value" rule.

**To revert:** drop the four `-?` in `biomni_judge.py`, the two in `scripts/regrade.py`, and delete
`TestRubricLevelParsing`.

---

## Patch 27 — sequential forking: fork the fork until full reward or out of steps  **[REVERTED 2026-08-04]**

> **Status: reverted.** Sequential forking and its judge-side anchoring are gone from the tree —
> `fork_trajectory.py` forks each trajectory exactly once again, and no judge is told anything
> about a fork. The section below is kept as design history (in particular the anchoring bug in
> piece 3, which is the reason the feedback anchor rule is what it is).
>
> **What was actually removed**, beyond this section's own revert list:
> `PRIOR_SCORES_NOTE`, `judges.base.format_prior_scores` / `_flatten`, `JudgeContext.parent_criteria`
> and `InterpreterEnv.parent_criteria` — the parent-grade calibration block, added after this
> writeup and never recorded here. Also `TestPriorScores` (its tests), `fork_chain`, `RoundResult`,
> the `--max-rounds` flag, `<root>-chain.json`, and the `-rN_cellK` directory naming (back to
> `-fork_cellK`). Stop reasons are now `full_reward` / `truncated` / `submitted`; `no_wrong_step`,
> `cell_never_appended` and `no_room` became `SkipFork` reasons raised before any container starts.
> `fork.bash` lost its `MAX_ROUNDS` block and the `--max-rounds` argument it passed (which argparse
> would now reject), and its default `OUT_DIR` moved off `seq-forks-…`.
>
> **Two deviations from the revert list below:**
> 1. `fork_trajectory.py` was rewritten by hand, not restored from git history — `scripts/` is
>    untracked, so there is no history to restore from.
> 2. `inspect_fork.py` was **deliberately kept** as-is (plus a `submitted` badge class). Its chain
>    grouping degrades correctly to a chain-of-one on the new flat layout — verified by rendering a
>    new-layout directory — and reverting it would lose the chain view for the 433 rounds / 144
>    chains already under `archive/`, which remain readable only through it.
>
> **Consequence for existing artifacts:** every forked run under `archive/` was graded with the
> resume + prior-score prompt notes, so those grades are *not* comparable with anything produced
> from this commit onward. Any fork A/B needs both arms re-run.

**Files:** `scripts/fork_trajectory.py` (rewritten), `scripts/inspect_fork.py`,
`src/hypotest/env/judges/base.py`, `src/hypotest/env/judges/{hypotest_judge,biomni,heureka}.py`,
`src/hypotest/env/prompts.py`, `src/hypotest/env/interpreter_env.py`, `tests/test_judges.py`.

**What:** `fork_trajectory.py` forked each trajectory exactly once — replay to the earliest
per-criterion `first_wrong_step`, inject that criterion's feedback, generate a fresh
continuation, stop. It now repeats that on its own output: **round N+1 forks round N's
notebook**, and the chain runs until the rollout earns full reward or exhausts its step budget.

Three pieces:

1. **A floor on `first_wrong_step`** (`derive_first_wrong_step(criteria, rubric, min_step=0)`).
   Incorrect steps below `min_step` are ignored, and a criterion whose only incorrect steps sit
   below it derives to `None`. This is what makes the chain converge: a forked rollout replays
   its parent's cells `0..K-1` verbatim, so those cells are frozen — without the floor a round
   could flag a frozen prefix cell and the chain would walk backwards forever. The floor is
   **inclusive** (`step >= min_step`): the fork cell itself was regenerated by the new policy, so
   it may legitimately be flagged again — which is why `--max-rounds` exists. `min_step=0` is the
   default and a no-op, so every non-forked run derives exactly as before.
   The floor lives in the shared post-judge path, so it applies to **every** registered judge.

2. **`resume_from_step`**, plumbed from `InterpreterEnv` (an instance attribute, set
   post-construction by the fork script — deliberately *not* an `InterpreterEnvConfig` field,
   since `DatasetConfig` splats into that and it would leak into the dataset-server path) through
   `JudgeContext` into the judges. `_score_solution` passes it as `min_step` and records it in
   `score_metadata`, so `score_info.json` documents the floor that was applied. Also widened
   `InterpreterEnvState.score_metadata` from `dict[str, str | int]` to `dict[str, Any]` — the old
   annotation never matched the list-of-criteria already stored there (a standing mypy error).

3. **`prompts.RESUME_NOTE` + `judges.base.with_resume_note`**, one `with_resume_note(prompt, ctx)`
   wrap in each of the three judges that emit both `relevant_steps` and `feedback`. The note tells
   the judge that cells `0..K-1` are inherited and frozen, that scoring is unchanged, and that
   `feedback` must anchor on a cell at or after `K` — otherwise the guidance injected at the next
   fork point describes work that can no longer be changed. `with_resume_note` is the identity
   when `resume_from_step is None`, so ordinary benchmark runs send byte-identical prompts.

**Chain control flow** (`fork_chain`, wrapping the refactored `fork_round`). Round 1's fork point
comes from the benchmark run's `score_info.json` as before (matched by answer via
`regrade.build_mapping`); every later round reads it straight out of the previous round's env,
in-process — no matching needed, since a round grades exactly one trajectory.

Stop reasons, first one wins:

| reason | condition |
|---|---|
| `full_reward` | normalized score reached 1.0 — every criterion satisfied |
| `no_wrong_step` | no criterion flags a cell at or after the floor |
| `truncated` | the round hit `max_steps` without submitting, so there is no new grade to fork on |
| `no_room` | the fork step reached `AGENT_MAX_STEPS - 1` — nothing left to generate |
| `cell_never_appended` | the flagged cell is never created in the parent's notebook |
| `max_rounds` | the `--max-rounds` cap (new flag, default 6) |

**Step budget is per round, replay included** — unchanged from the single-fork script, just now
load-bearing. Each round is a fresh env with `max_steps = AGENT_MAX_STEPS` (50, from `.env`) and
the replayed prefix already advances `env.step_count`, so the generation loop
(`while env.step_count < env.max_steps`) self-limits:

```
round 1:  [replay 0..19 ][ generate up to 30 ]   K=20, 50 cap
round 2:  [replay 0..27       ][ gen up to 22 ]  K=28, 50 cap
round 3:  [replay 0..40             ][ gen  9 ]  K=41, 50 cap
round 4:  [replay 0..48                    ][ ]  K=49 -> STOP, no room
```

As the fork point creeps later, generation room shrinks to nothing and the chain stops with
`no_room` — checked *before* spinning up a container, so a doomed round costs nothing. Only the
current round's feedback is injected; earlier rounds' guidance is not replayed into the rebuilt
context.

**Output layout stays flat** so `scripts/inspect_fork.py` and `scripts/regrade_forks.py` keep
working unchanged — both scan direct sub-dirs of `forks/` for `trajectories.pkl` + `fork_info.json`,
which nesting rounds would have broken. Each round is its own top-level dir
`forks/<root_traj_id>-r<N>_cell<C>/`; `fork_info.json` gains `round`, `fork_floor`,
`parent_traj_id`, `steps_available`, `submitted` and `stop_reason`, and keeps `old_first_wrong_step`
(= that round's fork cell) so the viewer's "fws cell X → Y" header still renders. Per chain,
`forks/<root_traj_id>-chain.json` holds the ordered rounds, `score_trace` and `stop_reason`;
`--skip-existing` keys on that file. `fork_summary.json` gains a `"chains"` key and keeps the flat
`"forked"` list of every round. `--num-parallel` now caps concurrent *chains* (rounds within a
chain are sequential by construction).

**Viewer (`scripts/inspect_fork.py`).** It rendered each fork dir as an independent panel, which
for a chain loses the thing you actually want to see — the progression. It now groups rounds into
chains and shows it:

- **Grouping.** `find_forks` returns `Round` objects (name, `fork_info.json`, trajectory, root id,
  round number); `group_chains` buckets them by `source_traj_id` and orders by round, attaching
  `<root>-chain.json` when present. Root and round number come from `fork_info.json`, falling back
  to parsing the directory name, so a round with a missing or corrupt `fork_info.json` still lands
  in the right chain. Pre-Patch-27 `<traj>-fork_cell<N>` dirs match a second regex and render as
  chains of one — **old fork directories are unaffected.**
- **Sidebar.** Rounds nest under a chain header carrying the source id, the full `score_trace`
  (`0.50 → 0.62 → 0.62 → 1.00`) and a colour-coded `stop_reason` badge (green `full_reward`, blue
  `no_wrong_step`, amber for the limit reasons).
- **Per-round delta is now against the previous round**, not the original run — that is what the
  round actually changed. The header keeps both: `vs. original` and `vs. previous round`. The
  sidebar dot follows the previous-round delta.
- **Panel header** gains a clickable round strip (`r1 · cell 3 → r2 · cell 9 → r3 · cell 17`, the
  current one highlighted), `round N of M`, `floor`, a `never submitted` flag when
  `submitted: false`, and the chain trace with the current round bolded.
- **A frozen-prefix note** spells out the one genuinely confusing thing about reading a forked
  notebook: "Steps 0–8 were replayed verbatim from `task_1-r1` and are frozen. The policy took over
  at step 9 (notebook cell 9), and this round's grade only flags wrong steps from that cell
  onward." Without it, a reader sees the replayed cells and assumes the policy wrote them.

**Usage:**

```bash
source .venv/bin/activate && set -a && source .env && set +a

# chain every trajectory in the pkl, at most 6 rounds each
python scripts/fork_trajectory.py \
    --server-config server.yaml --benchmark-config benchmark.yaml \
    --results archive/trial/results --out-dir forks/

# one trajectory, at most 3 rounds
python scripts/fork_trajectory.py --traj-id task_0_rep0 --max-rounds 3 ...
```

Then `python scripts/inspect_fork.py forks/ --html forks.html` renders each round as its own
panel, exactly as it did single forks.

**Judge coverage.** Forking has always required a judge that emits both `relevant_steps` and
`feedback`; this patch neither narrows nor widens that.

| judge | `relevant_steps` | `feedback` | forkable | notes |
|---|---|---|---|---|
| `hypotest` | yes | yes | yes | the bixbench-hypothesis path `server.yaml` currently runs |
| `biomni` | yes | yes | yes | A/B/C levels; its prompt is `.replace()`-derived from the hypotest tail, so it inherits the contract verbatim |
| `heureka` | yes | yes | yes | own prompt, same field contract |
| `biomystery` | anti-cheat evidence only | **no** | degenerate | one synthetic criterion whose steps flag *cheating*, not analysis errors, and no `feedback` key — a fork would land on a cheat step with no guidance |
| `bixbench` | hardcoded `[]` | no | **no** | derivation always yields `None` → `SkipFork` |
| `bioagent` | n/a (deterministic file scorer, `needs_model=False`) | no | **no** | same |

The floor (`min_step`, `resume_from_step`, `score_metadata["resume_from_step"]`) and the chain
driver are judge-agnostic and apply to all six — including any judge added later, since the floor
lives in the shared post-judge path in `_score_solution` and the driver only reads
`first_wrong_step` / `feedback` off `score_info.json`. Only the prompt note is per-judge, and it is
one `with_resume_note(prompt, ctx)` line to add.

Pre-existing gap this does not fix: the full-marks gate depends on `parse_criterion_max_scores`,
which understands only hypotest's `N. (X points)` / `* X points:` and biomni's `Levels: A=X B=Y C=0`.
On a heureka rubric it returns `[]`, the gate is skipped, and a full-marks criterion can still drive
a fork — the same class of defect Patch 20 fixed for bixbench rubrics.

**Why:** a single fork answers "would the policy have done better from here?" once. Iterating it
answers the more useful question — can the policy be walked to full reward one localised failure at
a time, and how many corrections does that take? The `score_trace` and `fork_cells` in
`chain.json` are exactly that curve. The floor is not a nicety: without it the chain has no
termination argument at all.

**Verification:**
- `TestSequentialForkFloor` (5 tests): the floor is inclusive and skips the frozen prefix
  (`min_step` 0/3/4/11/12 → 3/3/11/11/None), the default is byte-identical to omitting it, the
  full-marks gate still wins over the floor, and `with_resume_note` is the identity without a fork
  point but names the cell with one.
- All 8 chain stop reasons driven with `fork_round` stubbed (no LLM, no kernel): each terminates
  for the expected reason, and fork cells are non-decreasing in every case.

  | scenario | stop reason | fork cells |
  |---|---|---|
  | round 2 scores 1.0 | `full_reward` | `[4, 9]` |
  | round 2's criteria all clean | `no_wrong_step` | `[4, 9]` |
  | round 1 never submits | `truncated` | `[4]` |
  | keeps finding a later wrong step, `--max-rounds 3` | `max_rounds` | `[4, 5, 6]` |
  | **round 2 flags cell 2 after forking at cell 9** | `no_wrong_step` | `[4, 9]` |
  | fork point reaches cell 49 (`AGENT_MAX_STEPS - 1`) | `no_room` | `[4]` |
  | fork point reaches cell 48 — still 2 steps of room | continues | `[4, 48]` |
  | target cell never appended in the parent | `cell_never_appended` | `[4]` |

  The fifth row is the one that matters: without the floor that round would have forked
  *backwards* to cell 2 and the chain would never terminate.
- 190 tests pass; `mypy --scripts-are-modules` is clean on every touched file (the standing
  `scripts/regrade.py:195` and `scripts/inspect_trajectory.py` errors are untouched and pre-date
  this patch).
- The viewer rendered against a synthetic fixture — a 3-round improving chain (`full_reward`), a
  2-round stalled chain that never submits (`truncated`, same fork cell twice), and one legacy
  `-fork_cell12` dir — all in one directory. All three group correctly, `chain.json`'s
  `stop_reason` takes precedence over the per-round copy, the trace bold and the highlighted round
  pill track the panel, and the output parses with no unclosed or mismatched tags.
- **Not verified: no live end-to-end chain has been run.** That needs the capsules, a kernel, and
  live policy + rubric models. The `score_trace` / fork-cell monotonicity claims above are from the
  stubbed driver, not from a real rollout.

**Test-suite flakiness (not caused by this patch).** Under `pytest -n auto`, kernel-backed tests in
`tests/test_interpreter_env.py` and `tests/test_interpreter.py` intermittently fail with
`zmq.error.ZMQError: Address already in use` / `Kernel died before replying to kernel_info` —
parallel workers race for the same loopback ports. Two such failures appeared in the run above and
both pass when re-run serially. Running two full suites concurrently makes it much worse (observed:
1 failure + 3 errors, and 17 minutes instead of 52 seconds). Re-run the failures with
`-p no:randomly` and no `-n` before treating them as real.

**To revert:** restore the single-fork `fork_trajectory.py` from git history; drop `min_step` from
`derive_first_wrong_step`; delete `RESUME_NOTE`, `with_resume_note`, its export from
`judges/__init__.py`, the three `with_resume_note(prompt, ctx)` wraps, `JudgeContext.resume_from_step`,
`InterpreterEnv.resume_from_step` and the `score_metadata["resume_from_step"]` line; delete
`TestSequentialForkFloor`; restore `inspect_fork.py`'s flat `find_forks`/`write_html` from git
history. The `score_metadata` annotation widening can stay — it is independent.

---

## Patch 28 — ablation: fork feedback from *every* criterion, not just the driving ones

**Files:** `scripts/fork_trajectory_allfb.py`, `fork.allfb.bash` (both **new**; the baseline
`scripts/fork_trajectory.py` and `fork.bash` are untouched).

**What:** an A/B arm for the fork-feedback rule. The baseline forks at the earliest per-criterion
`first_wrong_step` and injects the `feedback` of only the criteria sitting at *that* cell
(`select_fork_criterion` → `criterion_feedback`). This arm keeps the fork **cell** identical and
widens only the note, to the feedback of every criterion that has any, split into two sections:

```
Guidance for how to proceed (from an evaluator).

Start here — the earliest thing to fix:
- <feedback from each criterion driving the fork cell>

Also address as you continue:
- <the rest, ordered by first_wrong_step, unlocated last>

Take this into account as you continue.
```

Criterion *names* are omitted — they are full sentences (median 104 chars, max 288 across the
archived runs) and would swamp the guidance they label.

"Every criterion" resolves to "every criterion that lost points": the judge emits `feedback: null`
at full marks by contract (`prompts.py`, the `"feedback"` bullet), verified over the 1023 criteria
in `archive/sonnet-judge/results-hypotest-wo-protocol` — of the 509 with no feedback, 473 scored
1/1 and 33 scored 5/5. Criteria that lost points with no located wrong cell are included and sort
last, but never actually fire: all 7 sit in one run that has no `first_wrong_step` anywhere and is
skipped outright.

**Implementation:** the variant imports the baseline module and rebinds exactly two module-level
functions rather than duplicating ~660 lines — drift between the arms is precisely what would
invalidate the ablation. `select_fork_criterion` keeps element `[0]` (the fork cell) identical and
widens element `[1]` to shallow copies of every criterion with feedback, each tagged
`_fork_driver`; `criterion_feedback` renders the two sections. The copies are safe because `[1]`
flows only into `criterion_feedback` and is never serialised, so the marker cannot reach
`fork_info.json` / `score_info.json`. `fork.allfb.bash` is `fork.bash` with `-allfb` job names, its
own `OUT_DIR`, the variant's script path, and its own `pgrep` pattern — `pgrep -f
"fork_trajectory.py"` does **not** match `fork_trajectory_allfb.py` (the regex `.` cannot span
`_al`), so a naive copy would wait forever for the process to appear.

**Verification** (offline, no LLM calls — both implementations driven over the 153 archived
`score_info.json` files):
- Fork cell identical to the baseline on all **143** forkable runs.
- Bullets per fork rise from mean **1.90** (max 7) to mean **3.55** (max 9).
- The `Start here` section's bullet set equals the baseline's note exactly, every run.
- Every bullet traces to a criterion with non-null feedback; `_fork_driver` leaks into 0 runs and
  the caller's criteria list is never mutated.
- Fork cell + step reproduce the 141 archived round-1 forks exactly (0 disagreements).

**Consequence:** no archived fork run is a valid control for this arm — they were all graded under
Patch 27's judge anchoring (see its REVERTED banner). Run `./fork.bash` and `./fork.allfb.bash`
against the same `PKL`/`RESULTS`.

**To revert:** delete `scripts/fork_trajectory_allfb.py` and `fork.allfb.bash`. Nothing else
references them.

---

# Known issue — per-rollout capsule copy amplification

**Not a patch — nothing has been changed.** Recorded because BioMysteryBench (Patch 25) makes it
expensive enough to matter. Filed 2026-07-31.

## Symptom

Every rollout byte-copies its entire capsule into a fresh workspace before the agent runs a single
cell, and under enroot it copies that workspace a second time. On multi-GB capsules this dominates
wall-clock time and NFS I/O.

## Where

1. **`src/hypotest/dataset_server.py:128`** — `Dataset.get_new_env_by_idx`:
   ```python
   shutil.copytree(capsule_path, problem_dir, dirs_exist_ok=True)
   ```
   Once per rollout, per replication. There is no sharing between rollouts of the same problem.
2. **`src/hypotest/env/interpreter_env.py:343`** — `_build_kernel_bash_script`, the enroot path:
   ```bash
   cp -a /data_workspace/. $WORKDIR/
   ```
   The workspace is bind-mounted into the container and then copied again onto node-local storage.
   So enroot runs pay **two** full copies per rollout; the Docker path (`Binds`, `interpreter_env.py:976`)
   and the local path pay one.

## Cost

BioMysteryBench, 90 problems, **155.4 GB** unpacked (the `data/*.zip` are stored, not compressed —
1.00x inflation, so unpacked size equals the download):

| capsule | size |
|---|---|
| `reccwgc4buredxvyz` | 27.5 GB |
| `reccniibn7ary80hj` | 23.8 GB |
| `recv1pkneurxhwpo9` | 17.6 GB |
| `recnayu0v8zttjlgf` | 14.2 GB |
| `recea4hqimc4sypon` | 12.8 GB |
| `recnquldskiadnpq8` | 10.5 GB |

One rollout of `reccwgc4buredxvyz` copies 27.5 GB (55 GB under enroot) before any analysis starts.
One pass over all 90 moves 155 GB; at 5 replications, 775 GB — 1.55 TB under enroot.

This is pre-existing, not introduced by Patch 25: HeurekaBench capsules are one-per-paper and hold
multi-GB `.h5ad` files, and several bioagent-bench capsules carry reference genomes. BioMysteryBench
is simply the first dataset where the copy is larger than anything the agent does with it.

Note the converters already hardlink at *staging* time (`stage_bioagent_capsules.py`,
`convert_heurekabench.py` — "hardlinked flat so the agent sees them by basename"). That dedupes the
source tree against the download; it does nothing for the per-rollout copy, which is a real copy.

## Why it is not a one-line fix

The obvious fix — symlink instead of copy — collides with containerization. Only `work_dir` is
mounted into the container:

- enroot: `--mount {work_dir}:/data_workspace` (`interpreter_env.py:409-411`, `:824-826`)
- Docker: `"Binds": [f"{work_dir}:/data_workspace"]` (`interpreter_env.py:976`)

A symlink pointing at `capsules/<id>/foo.bam` resolves fine on the host and **dangles inside the
container**, because that path is not mounted there. `cp -a` at `:343` preserves symlinks rather
than dereferencing them, so the enroot inner copy propagates the dangle rather than fixing it.

Hardlinks avoid the mount problem (they are ordinary directory entries) but are **unsafe**: a
truncating write through a hardlink corrupts the capsule master silently, for every subsequent
rollout of that problem. Copy-on-write (`cp --reflink`) is not available — the scratch space is NFS.

## Options, with trade-offs

1. **Symlink above a size threshold (~100 MB) + `chmod a-w` the capsule masters.** Recommended for
   the non-container path. Genomics inputs are read-only in practice (BAM, FASTQ, h5ad, bigWig,
   `.mtx.gz`), and a read-only master turns an in-place write into a loud permission error instead of
   silent cross-rollout corruption. Small files keep being copied, so anything the agent legitimately
   rewrites still works. **Requires option 2 to work under a container.**
2. **Also bind-mount the capsule dir read-only** (`{capsule_path}:{capsule_path}:ro`) so the symlink
   targets resolve inside the container, and skip the inner `cp -a` for symlinked entries. This is
   the part that makes option 1 usable with `use_docker`/`use_enroot`.
3. **Threshold-only copy** — copy everything under N MB, symlink the rest. Same as 1 but framed as a
   policy knob (`InterpreterEnvConfig.capsule_symlink_threshold_mb`, 0 = always copy = today's
   behaviour), which keeps the change opt-in and revertible.
4. **Do nothing for small datasets.** bixbench and biomni capsules are small; the copy is not worth
   engineering around there. Whatever is built should default to today's behaviour.

## Interactions to check when fixing

- **`close()`** (`interpreter_env.py:1082`, `:1121-1122`): `shutil.rmtree(work_dir)` does *not*
  follow symlinks — it unlinks them — so cleanup is safe. But `shutil.move(work_dir, save_dir)`
  preserves symlinks, so an archived run under `save_dir` would hold links into the capsule tree.
  Those stay valid only while the capsules do; anything that later tars or relocates `save_dir` must
  dereference (`tar -h`) or it will archive broken links.
- **`_prep_workspace_dir`** (`interpreter_env.py:294`) writes `pydeps/`, `pip-cache/` and `pip.conf`
  into the workspace. These must remain real files, never symlinks.
- Agents that `chmod`, `mv`, or index in place (e.g. `samtools index` writing `foo.bam.bai` next to a
  read-only `foo.bam`) will now fail where they previously succeeded against a private copy. A
  read-only master surfaces this as a permission error, which is the intended tradeoff — but expect
  it to change a handful of rollouts.

## How to measure it

- Time from `get_new_env_by_idx` to first executed cell, per problem, against capsule size.
- `du -sh` the work_dir tree during a run; compare with the capsule.
- For enroot specifically, watch node-local disk fill during the `cp -a` at `interpreter_env.py:343`.

---

## Patch 29 — halve the backward `grad_input` buffer in `ChunkedDistributedLogprob`

**Files:** `rl/scripts/nemo_rl_setup.sh` (marker `HYPOTEST_GRAD_BUFFER_PATCH`).
**Patches:** `nemo_rl/distributed/model_utils.py` — **NVIDIA NeMo RL, Apache 2.0**, not our code.
Applied by `sed` at pod start inside `nvcr.io/nvidia/nemo-rl:v0.6.0`; nothing is committed to
NeMo RL or to the `bbh-third-party` checkout. This is the **fourth** local patch to a vendored
NVIDIA file and the **third** to this one file — see the maintenance note at the end.

**Not an upstream bug.** Unlike the three patches in `nemo_rl_setup.sh` that precede it, upstream's
code here is numerically correct and the fp32 buffer is a deliberate precision choice. This patch
is a reordering that reaches the same numbers in half the memory.

### What upstream does

`ChunkedDistributedLogprob.backward` preallocates the whole gradient for the whole sequence, then
performs **two** operations *on that buffer*:

```python
grad_input = torch.zeros_like(vocab_parallel_logits, dtype=torch.float32)   # :~329
...
grad_input_chunk.copy_(is_chosen.float().sub_(softmax_output))              # write
grad_input_chunk.mul_(grad_output[...].unsqueeze(dim=-1))                   # multiply IN-BUFFER
```

Because the multiply happens in the buffer, the buffer's dtype governs the **arithmetic**, not just
the storage. fp32 there means the multiply runs in fp32 and the value is rounded exactly once, by
autograd, on return. The preallocation is itself a memory optimisation — it avoids the forward's
`all_log_probs` list + `torch.cat`, which would briefly hold two full copies.

### Why it had to change

Measured 2026-08-10, run 5 — `train()` died in `backward`:

```
   9.00 GiB ÷ (62,080 columns × 4 bytes) = 38,912 tokens   ← an ordinary sequence
   free: 6.73 GiB
```

38,912 is unremarkable: it sits inside the 21k–48k distribution measured after
[Patch 30](#patch-30--strip-images-before-they-reach-the-policy). This is not a runaway episode.

### The trap: do NOT just flip the dtype

Changing `dtype=torch.float32` to the input dtype **alone** moves the multiply into bf16 and
changes the gradient. Measured on `[1, 4096, 2048]`:

| variant | bit-identical | differing elements | max abs diff |
|---|---|---|---|
| naive dtype swap | **no** | 2,097,311 / 8,388,608 | 1.56e-2 |
| reorder (this patch) | **yes** | 0 / 8,388,608 | 0.0 |

### What this patch does

Folds the multiply into the per-chunk fp32 temporary that already exists, so the narrowing happens
once, on write, and the destination can be the input dtype:

```python
grad_input = torch.zeros_like(vocab_parallel_logits, dtype=vocab_parallel_logits.dtype)
...
grad_input_chunk.copy_(
    is_chosen.float().sub_(softmax_output).mul_(grad_output[...].unsqueeze(dim=-1))
)
```

One rounding either way. The temporary is 1024 tokens wide and costs nothing.

```
   fp32 buffer   38,912 × 62,080 × 4  =  9.00 GiB   ✗ against 6.73 free
   bf16 buffer   38,912 × 62,080 × 2  =  4.50 GiB   ✓ 2.23 GiB spare
```

### Verification

Both real modules loaded side by side (megatron/nemo_rl stubbed), 1-rank gloo group, bf16 input:

```
forward  identical: True
backward identical: True   ndiff=0/1572864  max|diff|=0.000e+00
```

Separately confirmed that autograd narrows a returned fp32 gradient to the input dtype regardless,
so the fp32 buffer never reached the optimiser as fp32 anyway. Scripts:
`scratchpad/verify_patch.py`, `grad_precision.py`, `dtype_probe.py`.

**Caveat:** verified on CPU `torch 2.13.0+cpu`, not the cluster's CUDA build under DTensor. The
narrowing is core autograd semantics, but this is one version-and-backend removed from production.

### Revert

Delete the `HYPOTEST_GRAD_BUFFER_PATCH` stanza from `rl/scripts/nemo_rl_setup.sh` and recreate the
pod. The patch is idempotent and guarded on its marker; both anchors `raise SystemExit` with the
expected line number if upstream moves, so a version bump fails loudly rather than silently
reverting.

### Maintenance note

Three patches now rewrite `model_utils.py`, one rewrites `automodel/setup.py`. Confirm they applied
by grepping the run log for `patching`. Unlike the TP-plan patch — which works around *our* unusual
model layout — this one is generic: any NeMo RL user with a large vocabulary at long sequence length
pays 2× on the largest tensor in the backward pass for no numerical benefit. **Worth reporting
upstream at `NVIDIA/NeMo-RL`**; `scratchpad/verify_patch.py` reproduces it with no hypotest
dependency.

---

## Patch 30 — strip images before they reach the policy

**Files:** `src/hypotest/env/config.py` (`STRIP_IMAGES`), `src/hypotest/env/interpreter.py`,
`src/hypotest/env/tools/filesystem.py`, `src/hypotest/env/prompts.py`,
`rl/runai/workloads.sh`, `tests/conftest.py` (`images_enabled` fixture).

**What:** `STRIP_IMAGES` (default **true**, env-var overridable) removes images from everything the
model sees. The notebook keeps them — this changes the *model's* view, not the saved record.

**Why:** a plot returns as a base64 data URI. A vision-capable server prices it by pixels (~1.8k
tokens), but the GRPO path serves the policy with `language_model_only: true` — no vision tower —
so vLLM tokenizes the base64 as **text**. Measured 2026-08-07 on a 5-task smoke run:

| task | figure | total tokens | outcome |
|---|---|---|---|
| task_0 | 1 | 225,286 (194,763 base64) | HTTP 400 → died → reward 0.000 |
| task_1 | 1 | 124,714 (101,573 base64) | HTTP 400 → died → reward 0.000 |
| task_2/3/4 | 0 | 22k–42k | completed, scored |

Perfect correlation: **every episode that drew a plot died; every one that didn't, finished.** The
episodes are written to `rollouts.jsonl` with reward 0 and no `usage` block, so they are
indistinguishable from genuinely bad work — a corrupted advantage signal, not just lost data.
`NB_OUTPUT_LIMIT` truncates text but never applied to images, so the largest thing a cell could
return was also the only uncapped one.

**Nothing wanted the pixels.** The judge reads `view_notebook`'s markdown and its image list is
discarded (`interpreter_env.py` `_score_solution`, `nb_content, _ = view_notebook(...)`), and 0 of
341 rubric criteria across the 51 capsules require a figure — 5 mention one, 3 of those saying
"optional". The six rubric grading dimensions have no visualisation category.

**Four paths close, not one:** cell execution (`_extract_images_from_output`, the chokepoint for
both `get_images()` and `has_images()`), `read("chart.png")`, `read("nb.ipynb")`. PDF/PPTX are
docstring-only, unimplemented.

**Prompt change (reward-affecting, made deliberately):** `DEFAULT_SYSTEM_PROMPT` §4 previously said
*"this is very important, at the end of the analysis you should always aim to create a final
figure"* — the instruction that killed both episodes, asking for the one artifact the grader cannot
see. Replaced with an explicit prohibition; the ggplot2 mandate, its worked example, and the
plotting libraries in the capability lists were removed too.

**Result (rerun 2026-08-10, same 5 tasks, 50 steps):**

```
HTTP 400s               2 → 0        episodes reaching submit   3/5 → 5/5
base64 tokens     296,336 → 0        zero-reward episodes       2/5 → 0/5
max sequence      225,286 → 43,962   usage recorded             3/5 → 5/5
```

**Revert:** `STRIP_IMAGES=false` restores the old behaviour; the multimodal plumbing is untouched
and still covered by tests via the `images_enabled` fixture. The prompt change is separate and must
be reverted by hand.

---

## Patch 31 — GRPO advantage grouping key (the real cause of the "flat eval")

**File:** `rl/scripts/group_key_patch.py`, applied to
`nemo_rl/algorithms/grpo.py` by `nemo_rl_setup.sh` (marker `HYPOTEST_GROUP_KEY_PATCH`).

**Symptom.** Training advantages were **exactly zero** every step —
`Advantages stats: min=0.0000 max=0.0000 mean=0.0000 std=0.0000` — despite healthy
reward spread (e.g. `min=0.0 max=0.8889 mean=0.2951 std=0.3476`). Zero advantage =
zero policy gradient = the model never learns. This was mis-attributed to
"small-group ties"; the arithmetic rules that out (a tied max group of four at
0.8889 sums to 3.5556, which already exceeds the batch's 3.5412 total reward, so
within-group reward variance **must** exist).

**Root cause.** NeMo RL computes the leave-one-out baseline per *prompt group*, and
it derives the group key from message **content**:
`_extract_prompt_only_messages()` keeps every `user`/`system` message in the
trajectory (assistant excluded), flattens to token ids, and groups with
`torch.unique(prompts, dim=0)` (`calculate_baseline_and_std_per_prompt`,
`algorithms/utils.py`). That is correct for standard GRPO, where all divergence
lives in the excluded assistant response and the only user/system content is the
fixed prompt. It breaks for this **multi-turn agentic** env: per-turn
observations, env-state summaries (`"N commands executed"`) and time warnings
(`"… {remaining} seconds remaining"`, `remaining` = **wall-clock**) are all
recorded as **`user`-role** messages (`Message(content=…)` defaults to
`role="user"`; only cell/tool results are `role="tool"`). Those diverge across a
prompt's generations, so each generation gets a unique key → **every group has
size 1** → the `valid_mask.sum() <= 1` branch sets `baseline = reward` →
`advantage = reward − reward = 0` for the whole batch.

**Confirmed on a real trajectory dump** (`rl/results/traj/…/traj_step37.jsonl`,
27B run): 12 episodes = 3 prompts × 4 generations. Grouped by the **initial
prompt** (messages before the first assistant turn) → the correct **3 groups of
4**. Grouped the way NeMo RL actually does (full user/system content) →
**12 groups of size 1**. Every episode carried 13–29 divergent mid-trajectory
`user`-role messages. Because the bug is content-driven and not async-specific, it
also affected the earlier **27B run** — i.e. it is the true cause of the flat
held-out eval, not `α`, `lr`, KL, or async.

**Fix.** Key grouping on the **initial prompt only** — stop
`_extract_prompt_only_messages()` at the first `assistant` message. That prefix
(system + task + initial listing) is identical across a prompt's generations and
distinct across prompts, so `torch.unique` recovers the correct groups and
within-group reward variance turns into non-zero advantages. It changes only the
grouping key, not what is trained on (the flattened `message_log` + loss mask),
the rewards, or the loss. Fail-loud if the anchor is missing (upstream refactor);
idempotent.

**Revert:** remove the `HYPOTEST_GROUP_KEY_PATCH` block from `nemo_rl_setup.sh`
(the pod re-patches a clean NeMo RL checkout each start). A cleaner long-term fix
would be to group on an explicit prompt/sample index threaded from generation
rather than on message content, but the initial-prompt key needs no NeMo-RL data
plumbing and is verified against the dump.

---

# Runbook — recovering a broken `.venv` or kernel env

Operational recovery notes, not a patch. Background: see Patch 22.

## Symptoms

`.venv` is broken if `import numpy` (or anything importing it, e.g. `datasets`, `scripts/generate_hypotheses.py`) fails with:

```
ImportError: ... Original error was: libscipy_openblas64_-<hash>.so: cannot open shared object file
```

## Diagnose (30 seconds)

```bash
cd /mlbio_scratch/wangsaja/hypotest

# 1. Dangling symlinks pointing at .nfsXXXX files = the NFS silly-rename signature:
#    something deleted/replaced a .so while a process had it mmap'd.
find .venv/lib -xtype l

# 2. More than one numpy dist-info = something installed into .venv that should not have.
ls -d .venv/lib/python3.*/site-packages/numpy*.dist-info

# 3. THE ROOT CAUSE CHECK. If this prints nothing, the agent kernel is running
#    inside .venv and will keep breaking it. Fix this before anything else.
grep '^KERNEL_ENV_PATH=' .env
```

## Repair `.venv`

```bash
# Stop the dataset server and any stray kernels first — deleting files that a live
# process holds open is what causes the dangling-symlink state in the first place.
pgrep -u "$USER" -fa 'dataset_server|benchmark_agent|ipykernel'

cd /mlbio_scratch/wangsaja/hypotest
mv .venv .venv.broken        # rename, do not rm: reversible, and rm is slow on NFS

# The explicit -p is REQUIRED. A bare `uv sync` silently rebases the venv onto
# conda bixbench's Python 3.12, which is the env the launch procedure avoids.
/home/wangsaja/.conda/envs/bixbench/bin/uv sync \
  -p /mlbio_scratch/wangsaja/.uv/cpython-3.13.10-linux-x86_64-gnu/bin/python3.13

# Verify, then delete .venv.broken (~1.1 GB) once satisfied.
env -u PYTHONPATH .venv/bin/python -c \
  "import numpy, datasets, lmi, numpy.linalg as la; print(numpy.__version__, la.norm([3.,4.]))"
find .venv/lib -xtype l | wc -l    # must be 0
```

## Rebuild the kernel env

The kernel env is disposable by design — rebuild rather than repair. The script is
idempotent end-to-end: conda solve, pip extras (`rdata`, `pyreadr`, CPU torch), the
`uv` shim, and a verification block.

```bash
rm -rf /mlbio_scratch/wangsaja/kernel_env_conda
/mlbio_scratch/wangsaja/build_kernel_env_conda.sh   # ~2 GB, prints BUILD_OK
```

Then confirm `.env` still has `KERNEL_ENV_PATH=/mlbio_scratch/wangsaja/kernel_env_conda`.

## Confirm the agent kernel is isolated

Start a kernel the way the server does and check where it actually lands:

```bash
source .venv/bin/activate && unset PYTHONPATH && set -a && source .env && set +a
```

Then in a kernel cell (or any equivalent probe), all four must point at the kernel
env, **not** `.venv` and not conda `bixbench`:

```python
import sys, shutil, os
sys.executable                  # …/kernel_env_conda/bin/python
shutil.which("python")          # …/kernel_env_conda/bin/python
shutil.which("uv")              # …/kernel_env_conda/bin/uv   (the shim)
os.environ["PIP_TARGET"]        # <work_dir>/pydeps
```

If `sys.executable` is `.venv/bin/python`, `KERNEL_ENV_PATH` is unset or points
somewhere that does not exist — Patch 7's `argv[0]` pin is guarded by
`kernel_python.exists()` and fails open, silently falling back to PATH resolution.

## Why this happens

Agent notebook cells run as you, with write access to every env on `PATH`. Patch 4's
`PIP_TARGET` redirects plain `pip` to a per-rollout `pydeps` dir, but it is an
env-var default on one tool, not a boundary: `uv` ignores `PIP_TARGET` entirely and
`uv pip install --system` deliberately bypasses virtualenvs. That is what wrote into
`.venv` on 2026-07-28. Current mitigations — `KERNEL_ENV_PATH`, the `uv` shim, and
the corrected system prompt — are PATH precedence, not enforcement. Absolute paths,
`conda install`, and direct filesystem writes still escape. Only containerizing the
kernel makes this a hard boundary.
