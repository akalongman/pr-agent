"""Shared base for AI handlers that run a local command-line agent once per call.

An adapter subclass decides how a prompt becomes a command (argv and stdin) and how the
command's stdout becomes text and token counts. This base owns the subprocess, its
timeout, failure mapping, logging and run-details accounting.
"""
import asyncio
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.algo.run_details import record_ai_call
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger

# Linux rejects a single argv element above 128 KiB (MAX_ARG_STRLEN); keep a margin below it.
MAX_ARGV_ELEMENT_BYTES = 120_000


class CliHandlerError(RuntimeError):
    """A CLI-backed call failed. Raised (never swallowed) so retry_with_fallback_models moves on."""


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
        self.extra_args = [str(arg) for arg in (settings.get(f"{section}.EXTRA_ARGS", None) or [])]
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
        """Strip a LiteLLM provider prefix such as ``anthropic/``; adapters may narrow this."""
        return model.rsplit("/", 1)[-1]

    async def chat_completion(self, model: str, system: str, user: str, temperature: float = 0.2, img_path: str = None):
        if img_path and not self.supports_images:
            get_logger().warning(f"Image path is not supported for {type(self).__name__}. Ignoring image path: {img_path}")
        mapped_model = self.map_model(model)
        command = self.build_command(mapped_model, system, user)
        for arg in command.argv:
            if len(arg.encode("utf-8")) > MAX_ARGV_ELEMENT_BYTES:
                raise CliHandlerError(
                    f"{type(self).__name__}: a command argument exceeds {MAX_ARGV_ELEMENT_BYTES} bytes; "
                    "the prompt does not fit in one argv element")
        get_logger().info("System: ", system)
        get_logger().info("User: ", user)
        stdout = await self._run(command)
        response = self.parse_response(stdout)
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
                *command.argv, stdin=stdin_pipe, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        except FileNotFoundError as e:
            raise CliHandlerError(f"{type(self).__name__}: command not found: {command.argv[0]}") from e
        payload = command.stdin.encode("utf-8") if command.stdin is not None else None
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(payload), timeout=self.timeout)
        except asyncio.TimeoutError as e:
            process.kill()
            await process.wait()
            raise CliHandlerError(f"{type(self).__name__}: {command.argv[0]} timed out after {self.timeout:g}s") from e
        if process.returncode != 0:
            tail = stderr.decode("utf-8", errors="replace")[-2000:]
            raise CliHandlerError(f"{type(self).__name__}: {command.argv[0]} exited with {process.returncode}: {tail}")
        return stdout.decode("utf-8", errors="replace")
