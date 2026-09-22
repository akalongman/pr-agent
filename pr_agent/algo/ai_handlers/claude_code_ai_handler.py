"""Adapter that runs Claude Code headless (``claude -p``) once per call.

The flag set turns the harness into a plain completion: its own system prompt is replaced,
tools, settings files and MCP servers are off, and nothing is persisted. Authentication is
the harness's own login; this adapter reads no credentials.
"""
import json

from pr_agent.algo.ai_handlers.cli_ai_handler import CliAIHandler, CliCommand, CliHandlerError, CliResponse
from pr_agent.config_loader import get_settings

CLAUDE_CODE_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
CLAUDE_ALIASES = frozenset({"opus", "sonnet", "haiku", "fable"})


class ClaudeCodeAIHandler(CliAIHandler):
    settings_section = "claude_code"
    default_binary = "claude"

    def map_model(self, model: str) -> str:
        name = model.rsplit("/", 1)[-1]
        if name.startswith("anthropic."):
            name = name[len("anthropic."):]
        if "claude" not in name and name not in CLAUDE_ALIASES:
            raise CliHandlerError(f"ClaudeCodeAIHandler supports Claude models only, got '{model}'")
        return name

    def build_command(self, model: str, system: str, user: str) -> CliCommand:
        argv = [
            self.binary, "-p",
            "--restricted", "--strict-mcp-config", "--tools", "",
            "--no-session-persistence", "--output-format", "json",
            "--system-prompt", system, "--model", model,
        ]
        effort = str(get_settings().get("CONFIG.REASONING_EFFORT", "") or "").lower()
        if effort in CLAUDE_CODE_EFFORTS:
            argv += ["--effort", effort]
        argv += self.extra_args
        return CliCommand(argv=argv, stdin=user)

    def parse_response(self, stdout: str) -> CliResponse:
        try:
            data = json.loads(stdout)
        except ValueError as e:
            raise CliHandlerError("Claude Code did not return a JSON result") from e
        if not isinstance(data, dict) or data.get("is_error") or data.get("subtype") != "success":
            detail = data.get("result") if isinstance(data, dict) else stdout
            raise CliHandlerError(f"Claude Code reported an error: {str(detail)[:500]}")
        usage = data.get("usage") or {}
        prompt_tokens = sum(int(usage.get(key) or 0)
                            for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        completion_tokens = int(usage.get("output_tokens") or 0)
        finish_reason = "length" if data.get("stop_reason") == "max_tokens" else "stop"
        cost = data.get("total_cost_usd")
        return CliResponse(text=data.get("result") or "", finish_reason=finish_reason,
                           prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                           cost_usd=float(cost) if cost else None)
