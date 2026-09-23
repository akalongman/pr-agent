"""Share the subprocess plumbing of AI handlers that run a local command-line agent once per call.

An adapter subclass decides how a prompt becomes a command (argv and stdin) and how the
command's stdout becomes text and token counts. This base owns the subprocess, its
timeout, failure mapping, logging and run-details accounting.
"""
import asyncio
import os
import shlex
import signal
from abc import abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from typing import Optional

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.algo.run_details import record_ai_call
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger

# Linux rejects a single argv element above 128 KiB (MAX_ARG_STRLEN); keep a margin below it.
MAX_ARGV_ELEMENT_BYTES = 120_000


class CliHandlerError(RuntimeError):
    """Signal a failed CLI-backed call; raise it (never swallow it) so retry_with_fallback_models moves on."""


@dataclass
class CliCommand:
    argv: list[str]
    stdin: Optional[str] = None


@dataclass
class CliResponse:
    text: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: Optional[float] = None


class CliAIHandler(BaseAiHandler):
    settings_section: str = ""  # lower-case TOML section holding binary, extra_args and timeout
    default_binary: str = ""
    supports_images: bool = False

    def __init__(self):
        super().__init__()
        section = self.settings_section.upper()
        settings = get_settings()
        self.binary = settings.get(f"{section}.BINARY", None) or self.default_binary
        raw_extra_args = settings.get(f"{section}.EXTRA_ARGS", None) or []
        if isinstance(raw_extra_args, str):
            try:
                self.extra_args = shlex.split(raw_extra_args)
            except ValueError as e:
                raise CliHandlerError(
                    f"{type(self).__name__}: cannot parse {self.settings_section}.extra_args: {e}") from e
        else:
            self.extra_args = [str(arg) for arg in raw_extra_args]
        timeout = settings.get(f"{section}.TIMEOUT", None)
        self.timeout = float(timeout) if timeout else float(settings.config.ai_timeout)

    @property
    def deployment_id(self):
        return None

    @abstractmethod
    def build_command(self, model: str, system: str, user: str) -> CliCommand:
        """Return the full argv (binary first) and the text to feed on stdin."""

    @abstractmethod
    def parse_response(self, stdout: str) -> CliResponse:
        """Turn the command's stdout into text and counts; raise CliHandlerError on a reported failure."""

    def map_model(self, model: str) -> str:
        """Return the last ``/``-separated segment of ``model``; adapters may narrow this.

        Drop every provider prefix, so ``openrouter/anthropic/claude-x`` becomes ``claude-x``.
        """
        return model.rsplit("/", 1)[-1]

    @staticmethod
    def reasoning_effort() -> str:
        """Return ``config.reasoning_effort`` lower-cased, or an empty string when it is unset."""
        return str(get_settings().get("CONFIG.REASONING_EFFORT", "") or "").lower()

    async def chat_completion(self, model: str, system: str, user: str, temperature: float = 0.2, img_path: str = None):
        if img_path and not self.supports_images:
            get_logger().warning(
                f"Image path is not supported for {type(self).__name__}. Ignoring image path: {img_path}")
        mapped_model = self.map_model(model)
        command = self.build_command(mapped_model, system, user)
        for arg in command.argv:
            if len(arg.encode("utf-8")) > MAX_ARGV_ELEMENT_BYTES:
                raise CliHandlerError(
                    f"{type(self).__name__}: a command argument exceeds {MAX_ARGV_ELEMENT_BYTES} bytes; "
                    "the prompt does not fit in one argv element")
        get_logger().debug("Prompts", artifact={"system": system, "user": user})
        stdout = await self._run(command)
        try:
            response = self.parse_response(stdout)
        except CliHandlerError:
            raise
        except Exception as e:
            raise CliHandlerError(
                f"{type(self).__name__}: cannot parse the output of {command.argv[0]}: {type(e).__name__}: {e}") from e
        get_logger().info("AI response", response=response.text, finish_reason=response.finish_reason,
                          model=mapped_model, prompt_tokens=response.prompt_tokens,
                          completion_tokens=response.completion_tokens)
        usage = {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.prompt_tokens + response.completion_tokens,
        }
        record_ai_call(usage, model=mapped_model, cost_usd=response.cost_usd)
        return response.text, response.finish_reason

    async def _run(self, command: CliCommand) -> str:
        stdin_pipe = asyncio.subprocess.PIPE if command.stdin is not None else asyncio.subprocess.DEVNULL
        try:
            process = await asyncio.create_subprocess_exec(
                *command.argv, stdin=stdin_pipe, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=(os.name == "posix"))
        except FileNotFoundError as e:
            raise CliHandlerError(f"{type(self).__name__}: command not found: {command.argv[0]}") from e
        except (OSError, ValueError) as e:
            # Report every other launch failure as a CliHandlerError too, so every CLI failure reaches
            # callers as one exception type: an OSError (PermissionError, E2BIG "Argument list too long")
            # or the ValueError subprocess raises for an argument that contains a NUL byte.
            raise CliHandlerError(f"{type(self).__name__}: cannot run {command.argv[0]}: {e}") from e
        payload = command.stdin.encode("utf-8") if command.stdin is not None else None
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(payload), timeout=self.timeout)
        except asyncio.TimeoutError as e:
            await self._kill_process_tree(process)
            raise CliHandlerError(f"{type(self).__name__}: {command.argv[0]} timed out after {self.timeout:g}s") from e
        except asyncio.CancelledError:
            await self._kill_process_tree(process)
            raise
        if process.returncode != 0:
            detail_parts = []
            for label, stream in (("stderr", stderr), ("stdout", stdout)):
                stream_tail = stream.decode("utf-8", errors="replace")[-2000:]
                if stream_tail:
                    detail_parts.append(f"{label}: {stream_tail}")
            detail = "; ".join(detail_parts)
            raise CliHandlerError(
                f"{type(self).__name__}: {command.argv[0]} exited with {process.returncode}: {detail}")
        return stdout.decode("utf-8", errors="replace")

    @staticmethod
    async def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
        """Kill every process the CLI spawned, not just the direct child, then reap it.

        A grandchild that inherited the pipes keeps them open after the direct child dies,
        so ``process.wait()`` would otherwise block until that grandchild exits on its own
        (Python resolves it only once the pipes close). Killing the whole process group
        avoids that: the child was started as its own session leader, so its pid doubles as
        the process group id.
        """
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            with suppress(ProcessLookupError):
                process.kill()
        await process.wait()
