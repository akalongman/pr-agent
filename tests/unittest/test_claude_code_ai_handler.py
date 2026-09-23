import json
from types import SimpleNamespace

import pytest

import pr_agent.algo.ai_handlers.claude_code_ai_handler as claude_module
import pr_agent.algo.ai_handlers.cli_ai_handler as cli_module
from pr_agent.algo.ai_handlers.claude_code_ai_handler import ClaudeCodeAIHandler
from pr_agent.algo.ai_handlers.cli_ai_handler import CliHandlerError


class _FakeSettings:
    def __init__(self, values=None):
        self.values = values or {}
        self.config = SimpleNamespace(ai_timeout=30)

    def get(self, key, default=None):
        return self.values.get(key, default)


def _install_settings(monkeypatch, values=None):
    factory = lambda: _FakeSettings(values)  # noqa: E731
    monkeypatch.setattr(cli_module, "get_settings", factory)
    monkeypatch.setattr(claude_module, "get_settings", factory)


def _success_payload(**overrides):
    payload = {
        "type": "result", "subtype": "success", "is_error": False, "result": "review text",
        "stop_reason": "end_turn", "total_cost_usd": 0,
        "usage": {"input_tokens": 400, "cache_creation_input_tokens": 50, "cache_read_input_tokens": 12,
                  "output_tokens": 30},
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_build_command_uses_the_restricted_headless_flag_set(monkeypatch):
    _install_settings(monkeypatch, {"CONFIG.REASONING_EFFORT": "xhigh", "CLAUDE_CODE.EXTRA_ARGS": ["--verbose"]})
    command = ClaudeCodeAIHandler().build_command("claude-opus-5", "SYS", "USER")

    assert command.argv == [
        "claude", "-p", "--restricted", "--strict-mcp-config", "--tools", "", "--no-session-persistence",
        "--output-format", "json", "--system-prompt", "SYS", "--model", "claude-opus-5", "--effort", "xhigh",
        "--verbose",
    ]
    assert command.stdin == "USER"


def test_build_command_omits_effort_the_harness_does_not_accept(monkeypatch):
    _install_settings(monkeypatch, {"CONFIG.REASONING_EFFORT": "minimal"})
    command = ClaudeCodeAIHandler().build_command("claude-opus-5", "SYS", "USER")
    assert "--effort" not in command.argv


@pytest.mark.parametrize("given, expected", [
    ("anthropic/claude-opus-5", "claude-opus-5"),
    ("bedrock/anthropic.claude-sonnet-5", "claude-sonnet-5"),
    ("claude-opus-5", "claude-opus-5"),
    ("opus", "opus"),
])
def test_map_model_strips_provider_prefixes(monkeypatch, given, expected):
    _install_settings(monkeypatch)
    assert ClaudeCodeAIHandler().map_model(given) == expected


def test_map_model_rejects_non_claude_models(monkeypatch):
    _install_settings(monkeypatch)
    with pytest.raises(CliHandlerError, match="Claude models only"):
        ClaudeCodeAIHandler().map_model("gpt-5.6")


def test_parse_response_reads_text_usage_and_unknown_cost(monkeypatch):
    _install_settings(monkeypatch)
    response = ClaudeCodeAIHandler().parse_response(_success_payload())

    assert response.text == "review text"
    assert response.finish_reason == "stop"
    assert response.prompt_tokens == 462
    assert response.completion_tokens == 30
    assert response.cost_usd is None


def test_parse_response_maps_max_tokens_to_length_and_keeps_a_reported_cost(monkeypatch):
    _install_settings(monkeypatch)
    response = ClaudeCodeAIHandler().parse_response(_success_payload(stop_reason="max_tokens", total_cost_usd=0.25))
    assert response.finish_reason == "length"
    assert response.cost_usd == 0.25


def test_parse_response_raises_on_reported_error(monkeypatch):
    _install_settings(monkeypatch)
    with pytest.raises(CliHandlerError, match="Not logged in"):
        ClaudeCodeAIHandler().parse_response(_success_payload(is_error=True, result="Not logged in"))


def test_parse_response_raises_on_non_json(monkeypatch):
    _install_settings(monkeypatch)
    with pytest.raises(CliHandlerError, match="JSON"):
        ClaudeCodeAIHandler().parse_response("plain text")


async def test_chat_completion_raises_cli_handler_error_on_non_numeric_usage(monkeypatch):
    _install_settings(monkeypatch)
    handler = ClaudeCodeAIHandler()

    async def fake_run(command):
        return _success_payload(usage={"input_tokens": 400, "output_tokens": "many"})

    monkeypatch.setattr(handler, "_run", fake_run)

    with pytest.raises(CliHandlerError, match="ClaudeCodeAIHandler") as exc_info:
        await handler.chat_completion(model="anthropic/claude-opus-5", system="s", user="u")
    assert isinstance(exc_info.value.__cause__, ValueError)
