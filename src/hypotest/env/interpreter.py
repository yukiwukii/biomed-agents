"""Unified kernel management module for code interpretation.

This module provides the Interpreter class for managing Jupyter kernels
and executing code, with optional security isolation via SecureKernelManager.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

from jupyter_client.asynchronous.client import AsyncKernelClient
from jupyter_client.manager import AsyncKernelManager
from nbformat import NotebookNode
from pydantic import BaseModel, ConfigDict, Field

from . import config as cfg
from . import utils
from .kernel_server import MessageType

logger = logging.getLogger(__name__)


class ExecutionResult(BaseModel):
    """Structured result from kernel code execution.

    Stores notebook outputs in nbformat format as the single source of truth.
    Text and images are derived lazily from notebook_outputs when needed.
    """

    notebook_outputs: list[NotebookNode] = Field(default_factory=list)
    error_occurred: bool = False
    execution_time: float | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _extract_text_from_output(output: NotebookNode) -> str | None:
        """Extract text from a single notebook output.

        Args:
            output: A NotebookNode output from cell execution

        Returns:
            Formatted text string or None if no text available
        """
        output_type = output.get("output_type", "")

        if output_type == MessageType.STREAM:
            name = output.get("name", "stdout")
            text = output.get("text", "")
            return f"[{name}]\n{text}"

        if output_type in {MessageType.EXECUTE_RESULT, MessageType.DISPLAY_DATA}:
            data = output.get("data", {})
            # Check for images first to add placeholder text
            text_parts = ["[Image generated]"] if utils.JUPYTER_IMAGE_OUTPUT_TYPES.intersection(data.keys()) else []
            # Add text/plain if available
            if "text/plain" in data:
                text_parts.append(data["text/plain"])
            return "\n".join(text_parts) if text_parts else None

        if output_type == MessageType.ERROR:
            traceback = output.get("traceback", [])
            traceback_str = "\n".join(traceback) if isinstance(traceback, list) else traceback
            return (
                f"Error: {output.get('ename', 'Unknown')}\n"
                f"Message: {output.get('evalue', 'No error message')}\n"
                f"Traceback:\n{traceback_str}"
            )

        return None

    @staticmethod
    def _extract_images_from_output(output: NotebookNode) -> list[tuple[str, str]]:
        """Extract images from a single notebook output.

        Args:
            output: A NotebookNode output from cell execution

        Returns:
            List of (mime_type, base64_data) tuples
        """
        images: list[tuple[str, str]] = []
        output_type = output.get("output_type", "")

        if output_type in {MessageType.EXECUTE_RESULT, MessageType.DISPLAY_DATA}:
            data = output.get("data", {})
            for img_type in utils.JUPYTER_IMAGE_OUTPUT_TYPES:
                if img_type in data:
                    try:
                        encoded = utils.encode_image_to_base64(data[img_type])
                        images.append((img_type, encoded))
                    except RuntimeError:
                        logger.exception("Error encoding image.")

        return images

    def get_text_outputs(self) -> list[str]:
        """Extract formatted text from all notebook outputs.

        Returns:
            List of text strings extracted from outputs
        """
        return [text for output in self.notebook_outputs if (text := self._extract_text_from_output(output))]

    def get_images(self) -> list[tuple[str, str]]:
        """Extract images from all notebook outputs.

        Returns:
            List of (mime_type, base64_data) tuples
        """
        return [img for output in self.notebook_outputs for img in self._extract_images_from_output(output)]

    def get_combined_text(self) -> str:
        """Get all text outputs combined as a single string."""
        text_outputs = self.get_text_outputs()
        if not text_outputs:
            return "Code executed successfully (no output)"
        return "\n".join(text_outputs)

    def has_images(self) -> bool:
        """Check if execution result contains images."""
        return any(self._extract_images_from_output(output) for output in self.notebook_outputs)

    def get_truncated_text(self) -> str:
        """Get the combined text, truncated to the output limit."""
        return utils.limit_notebook_output(self.get_combined_text())

    def get_error_message(self) -> str | None:
        """Extract the error message from outputs if an error occurred.

        Returns:
            Formatted error message or None if no error
        """
        if not self.error_occurred:
            return None

        for output in self.notebook_outputs:
            if output.get("output_type") == MessageType.ERROR:
                return self._extract_text_from_output(output)
        return None

    def to_message(self) -> dict[str, Any]:
        """Convert ExecutionResult to MCP tool result format.

        Returns a dict with 'content' array containing text and/or images.
        The SDK will automatically wrap this in a ToolResultBlock.
        """
        content: list[dict[str, Any]] = []

        # Always include text output (even if there are images)
        text = self.get_combined_text()
        if text:
            content.append({"type": "text", "text": text})

        # Add any images - now using extracted tuples directly
        for mime_type, base64_data in self.get_images():
            content.append({
                "type": "image",
                "mimeType": mime_type,
                "data": base64_data,
            })

        return {"content": content}


class Interpreter:
    """Manages Python/R interpreter kernels for code execution.

    This class handles kernel lifecycle, code execution, and maintains
    execution history and error tracking. It supports both AsyncKernelManager
    (development) and SecureKernelManager (production with isolation).
    """

    def __init__(
        self,
        work_dir: Path,
        language: utils.NBLanguage = utils.NBLanguage.PYTHON,
        *,
        execution_timeout: float = 600,
        use_host_env_vars: bool = False,
        extra_envs: dict[str, str] | None = None,
    ):
        """Initialize the interpreter.

        Args:
            work_dir: Working directory for the kernel
            language: Programming language (Python or R)
            execution_timeout: Timeout for code execution in seconds
            use_host_env_vars: Whether to use host environment variables
            extra_envs: Additional environment variables to pass to the kernel
        """
        self.work_dir = work_dir
        self.language = language
        self.execution_timeout = execution_timeout
        self.use_host_env_vars = use_host_env_vars
        self.extra_envs = extra_envs or {}

        # Execution state
        self.execution_history: list[ExecutionResult] = []

        # Kernel state
        self.kernel_manager: AsyncKernelManager
        self.client: AsyncKernelClient | None = None
        self._is_ready = False

    def _setup_pip_env(self, env: dict[str, str]) -> dict[str, str]:
        # PATCH 4: Redirect pip installs to work_dir/pydeps so the kernel can install
        # packages without needing write access to ~/.local or the system site-packages.
        # work_dir/pydeps is PREPENDED to PYTHONPATH so packages the model installs OR
        # upgrades into pydeps take precedence over the curated kernel_env copy — appending
        # (the original Patch 6) made `pip install --upgrade <pkg>` a silent no-op because
        # kernel_env always won. Prepending is safe because Patch 7 launches the kernel under
        # kernel_env's own 3.12 interpreter, so whatever pip drops into pydeps is ABI-matched
        # (the earlier prepend crash was a 3.13-vs-3.12 numpy mismatch, fixed at the source by
        # Patch 7) and pip — seeing kernel_env's packages as already installed — no longer
        # re-drags the full dependency tree into pydeps. The accepted tradeoff: an explicit
        # upgrade of a core lib now shadows kernel_env for the whole kernel (the desired
        # behavior). Mirrors what _prep_workspace_dir does for the Enroot/Docker paths.
        # To revert: delete this method, replace `merged = self._setup_pip_env(merged)` with
        #   `kwargs["env"] = merged`, and replace `kwargs["env"] = self._setup_pip_env(...)`
        #   with `kwargs["env"] = os.environ | self.extra_envs`.
        # Resolve to absolute so the PYTHONPATH/PIP_TARGET entries below are never
        # relative. A relative entry would re-resolve against the kernel's cwd (which
        # is the work_dir itself), producing a doubled `work_dir/<work_dir>/pydeps`
        # path — a nested tmp dir and a malformed sys.path that breaks numpy import.
        pydeps = (self.work_dir / "pydeps").resolve()
        pip_cache = (self.work_dir / "pip-cache").resolve()
        pydeps.mkdir(parents=True, exist_ok=True)
        pip_cache.mkdir(parents=True, exist_ok=True)

        pydeps_str = str(pydeps)
        existing = env.get("PYTHONPATH", "")
        if pydeps_str not in existing.split(os.pathsep):
            env["PYTHONPATH"] = f"{pydeps_str}{os.pathsep}{existing}" if existing else pydeps_str

        env.setdefault("PIP_TARGET", pydeps_str)
        env.setdefault("PIP_CACHE_DIR", str(pip_cache))
        return env

    async def start(self) -> None:
        """Start the kernel and prepare for execution."""
        if self._is_ready:
            return

        kernel_name = self.language.make_kernelspec()["name"]
        self.kernel_manager = AsyncKernelManager(kernel_name=kernel_name)

        # PATCH 6: Launch the Python kernel under kernel_env's own interpreter instead of
        # whatever interpreter resolved the ambient "python" kernelspec. The registered
        # kernelspec bakes an absolute path to the launching interpreter into argv[0], so
        # running the benchmark from e.g. a Python 3.13 .venv starts a 3.13 kernel — while
        # extra_envs (interpreter_env.py) injects kernel_env's 3.12 site-packages onto its
        # PYTHONPATH. A 3.12-built numpy can't load under 3.13, producing the misleading
        # "import numpy from its source directory" ImportError. Pointing argv[0] at
        # kernel_env's python makes the interpreter and site-packages agree regardless of
        # which venv launched the benchmark — matching what the enroot path already does by
        # exec'ing /app/kernel_env/bin/python directly (interpreter_env.py:365, :808).
        # To revert: delete this block.
        if self.language == utils.NBLanguage.PYTHON:
            kernel_python = Path(cfg.KERNEL_ENV_PATH) / "bin" / "python"
            if kernel_python.exists():
                # Accessing .kernel_spec loads and caches the spec on the manager; mutating
                # argv[0] in place is picked up by the subsequent start_kernel() call.
                self.kernel_manager.kernel_spec.argv[0] = str(kernel_python)

        # Prepare kernel startup kwargs with environment variables
        kwargs: dict[str, Any] = {"cwd": str(self.work_dir.resolve())}
        if not self.use_host_env_vars:
            env = {
                required_env_var: os.environ[required_env_var]
                for required_env_var in cfg.REQUIRED_PATH_ENV_VARS
                if os.environ.get(required_env_var)
            }
            # PATCH 3: Strip Python-version-specific paths from PYTHONPATH that
            # don't match the running interpreter (e.g. python3.12 paths when the
            # kernel runs Python 3.13). Filter is applied ONLY to the host-inherited
            # env paths, not to extra_envs (which are intentionally set for the
            # kernel, e.g. kernel_site_packages) — otherwise the filter would
            # silently drop the kernel's own site-packages when the outer Python
            # minor version differs from kernel_env's version.
            # To revert: replace the block below (through kwargs["env"] = merged) with
            #   kwargs["env"] = env | self.extra_envs
            current_pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
            if "PYTHONPATH" in env:
                env["PYTHONPATH"] = os.pathsep.join(
                    p for p in env["PYTHONPATH"].split(os.pathsep)
                    if not any(
                        f"python3.{minor}" in p
                        for minor in range(20)
                        if f"python3.{minor}" != current_pyver
                    )
                )
            merged = env | self.extra_envs
            merged = self._setup_pip_env(merged)
            kwargs["env"] = merged
        else:
            kwargs["env"] = self._setup_pip_env(os.environ | self.extra_envs)

        await self.kernel_manager.start_kernel(**kwargs)
        self.client = self.kernel_manager.client()
        self.client.start_channels()

        try:
            await self.client.wait_for_ready()
            self._is_ready = True
            logger.debug(f"Kernel {kernel_name} started successfully in {self.work_dir}")
        except Exception as e:
            # Capture kernel process info for debugging
            debug_info: list[str] = []
            if hasattr(self.kernel_manager, "provisioner") and self.kernel_manager.provisioner:
                prov = self.kernel_manager.provisioner
                debug_info.extend((
                    f"pid={getattr(prov, 'pid', None)}",
                    f"connection_file={self.kernel_manager.connection_file}",
                ))
                if hasattr(prov, "process") and prov.process:
                    proc = prov.process
                    debug_info.extend((f"returncode={proc.returncode}", f"poll={proc.poll()}"))
            debug_str = "; ".join(debug_info) if debug_info else "no debug info"
            raise RuntimeError(f"Kernel failed to start: {e} ({debug_str})") from e

    async def _execute_code(self, code: str) -> ExecutionResult:
        """Internal method to execute code and collect outputs.

        Uses MessageType.to_notebook_output to convert kernel messages
        to nbformat outputs, storing them as the single source of truth.
        """
        if not self.client:
            raise ValueError("Kernel client not initialized")

        start_time = time.perf_counter()
        msg_id = self.client.execute(code)
        logger.debug(f"Executing code with message ID: {msg_id}")

        notebook_outputs: list[NotebookNode] = []
        error_occurred = False

        while True:
            msg = await self.client.get_iopub_msg()
            logger.debug(f"Received message type: {msg['msg_type']}")

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = MessageType.from_string(msg["msg_type"])
            if msg_type is None:
                continue  # Unknown message type, skip

            content = msg["content"]

            if msg_type == MessageType.STATUS and content.get("execution_state") == "idle":
                break

            output = msg_type.to_notebook_output(content)
            if output:
                notebook_outputs.append(output)
                if msg_type == MessageType.ERROR:
                    error_occurred = True

        execution_time = time.perf_counter() - start_time

        return ExecutionResult(
            notebook_outputs=notebook_outputs,
            error_occurred=error_occurred,
            execution_time=execution_time,
        )

    async def execute_code(
        self,
        code: str,
        execution_timeout: float | None = None,
        extract_code: bool = False,
    ) -> ExecutionResult:
        r"""Execute code in the kernel session.

        The code will be executed in the current session context, maintaining
        all variables, imports, and state from previous executions.

        Code wrapped in markdown backticks (```code```, ```\ncode\n```, or
        ```language\ncode\n```) will be extracted if extract_code is True.

        Args:
            code: Code to execute
            execution_timeout: Optional timeout in seconds. If None, uses the
                instance's execution_timeout. Useful for dynamic timeout based
                on remaining job time.
            extract_code: Whether to extract code from md backticks and language identifiers.

        Returns:
            ExecutionResult containing text outputs and images
        """
        # Preprocess code to extract from markdown backticks and language identifiers
        if extract_code:
            code = utils.extract_code_from_markdown(code)

        if not self._is_ready:
            await self.start()

        # Use provided timeout or fall back to instance default
        timeout = execution_timeout if execution_timeout is not None else self.execution_timeout

        try:
            async with asyncio.timeout(timeout):
                result = await self._execute_code(code)
        except TimeoutError:
            timeout_output = MessageType.ERROR.to_notebook_output({
                "ename": "TimeoutError",
                "evalue": f"Code execution timed out after {timeout} seconds",
                "traceback": [f"TimeoutError: Code execution timed out after {timeout} seconds"],
            })
            result = ExecutionResult(
                notebook_outputs=[cast(NotebookNode, timeout_output)],
                error_occurred=True,
            )
        except Exception as e:
            error_output = MessageType.ERROR.to_notebook_output({
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": [f"{type(e).__name__}: {e}"],
            })
            result = ExecutionResult(
                notebook_outputs=[cast(NotebookNode, error_output)],
                error_occurred=True,
            )

        self.execution_history.append(result)
        return result

    async def execute_cells(self, cells: list[NotebookNode], cell_idx: int | None = None) -> list[str]:
        """Execute notebook cells using the kernel.

        This method supports the notebook workflow by executing cells
        and handling notebook-specific output formats.

        Args:
            cells: List of notebook cells
            cell_idx: Specific cell index to execute, or None to execute all

        Returns:
            List of error messages (empty if no errors)
        """
        if not self._is_ready:
            await self.start()

        if not self.client:
            raise ValueError("Kernel client not initialized")

        try:
            async with asyncio.timeout(self.execution_timeout):
                error_messages = await utils.nbformat_run_notebook(cells=cells, client=self.client, cell_idx=cell_idx)
        except TimeoutError as err:
            raise TimeoutError(f"Cell execution timed out after {self.execution_timeout} seconds") from err

        return error_messages

    async def reset(self) -> None:
        """Reset the kernel to a clean state."""
        if self._is_ready:
            await self.close()
        await self.start()

        # Clear execution history
        self.execution_history.clear()

    async def close(self) -> None:
        """Shutdown the kernel and cleanup resources."""
        if self._is_ready:
            # Properly stop client channels first
            if self.client:
                self.client.stop_channels()
                self.client = None

            # Then shutdown the kernel
            await self.kernel_manager.shutdown_kernel(now=True)

            # Clean up the kernel manager
            self._is_ready = False
            logger.debug("Kernel shutdown complete")

    def get_execution_summary(self) -> dict[str, Any]:
        """Get a summary of execution history and current state.

        Returns:
            Dictionary with execution statistics and recent activity
        """
        error_count = sum(1 for r in self.execution_history if r.error_occurred)
        recent_errors = [
            r.get_error_message() for r in self.execution_history[-3:] if r.error_occurred and r.get_error_message()
        ]

        return {
            "total_executions": len(self.execution_history),
            "error_count": error_count,
            "recent_errors": recent_errors,
            "last_execution": (self.execution_history[-1] if self.execution_history else None),
            "is_ready": self._is_ready,
            "language": self.language.value,
            "work_dir": str(self.work_dir),
        }

    @property
    def is_ready(self) -> bool:
        """Check if the kernel is ready for execution."""
        return self._is_ready
