"""Adapter that runs the OpenAI Codex CLI non-interactively (``codex exec``) once per call.

Global flags go before the subcommand: approvals off, a read-only sandbox, the model, and
pr-agent's system prompt as ``developer_instructions``. The run is ephemeral, ignores the
user's config and rules, and starts in an empty directory so no AGENTS.md is discovered.
Authentication is the CLI's own login; this adapter reads no credentials.
"""
import json
import tempfile

from pr_agent.algo.ai_handlers.cli_ai_handler import CliAIHandler, CliCommand, CliHandlerError, CliResponse
from pr_agent.config_loader import get_settings

CODEX_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh"})


def toml_basic_string(text: str) -> str:
    """Quote text as a TOML basic string so a ``--config key=value`` override carries it verbatim."""
    escaped = []
    for char in text:
        if char == "\\":
            escaped.append("\\\\")
        elif char == '"':
            escaped.append('\\"')
        elif char == "\n":
            escaped.append("\\n")
        elif char == "\r":
            escaped.append("\\r")
        elif char == "\t":
            escaped.append("\\t")
        elif ord(char) < 0x20 or char == "\x7f":
            escaped.append(f"\\u{ord(char):04X}")
        else:
            escaped.append(char)
    return '"' + "".join(escaped) + '"'


class CodexAIHandler(CliAIHandler):
    settings_section = "codex"
    default_binary = "codex"

    def __init__(self):
        super().__init__()
        # An empty working directory: no AGENTS.md is discovered and sandboxed commands run in scratch space.
        self.workdir = tempfile.mkdtemp(prefix="pr-agent-codex-")

    def build_command(self, model: str, system: str, user: str) -> CliCommand:
        argv = [
            self.binary,
            "--ask-for-approval", "never", "--sandbox", "read-only",
            "--model", model,
            "--config", "developer_instructions=" + toml_basic_string(system),
        ]
        effort = str(get_settings().get("CONFIG.REASONING_EFFORT", "") or "").lower()
        if effort == "max":
            effort = "xhigh"
        if effort in CODEX_EFFORTS:
            argv += ["--config", f'model_reasoning_effort="{effort}"']
        argv += [
            "exec", "--json", "--ephemeral", "--ignore-user-config", "--ignore-rules",
            "--skip-git-repo-check", "--color", "never", "--cd", self.workdir,
        ]
        argv += self.extra_args
        argv.append("-")
        return CliCommand(argv=argv, stdin=user)

    def parse_response(self, stdout: str) -> CliResponse:
        text = None
        prompt_tokens = completion_tokens = 0
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("type")
            if kind in ("error", "turn.failed"):
                detail = event.get("message") or event.get("error") or event
                raise CliHandlerError(f"Codex reported a failure: {str(detail)[:500]}")
            if kind == "item.completed" and (event.get("item") or {}).get("type") == "agent_message":
                text = event["item"].get("text") or ""
            elif kind == "turn.completed":
                usage = event.get("usage") or {}
                prompt_tokens = int(usage.get("input_tokens") or 0)
                completion_tokens = (int(usage.get("output_tokens") or 0)
                                     + int(usage.get("reasoning_output_tokens") or 0))
        if text is None:
            raise CliHandlerError("Codex produced no agent message")
        return CliResponse(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
