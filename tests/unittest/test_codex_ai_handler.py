import gc
import json
import os
from types import SimpleNamespace

import pytest

import pr_agent.algo.ai_handlers.cli_ai_handler as cli_module
import pr_agent.algo.ai_handlers.codex_ai_handler as codex_module
from pr_agent.algo.ai_handlers.cli_ai_handler import CliHandlerError
from pr_agent.algo.ai_handlers.codex_ai_handler import CodexAIHandler, toml_basic_string


class _FakeSettings:
    def __init__(self, values=None):
        self.values = values or {}
        self.config = SimpleNamespace(ai_timeout=30)

    def get(self, key, default=None):
        return self.values.get(key, default)


def _install_settings(monkeypatch, values=None):
    factory = lambda: _FakeSettings(values)  # noqa: E731
    monkeypatch.setattr(cli_module, "get_settings", factory)
    monkeypatch.setattr(codex_module, "get_settings", factory)


def _events(*events):
    return "\n".join(json.dumps(event) for event in events) + "\n"


def test_toml_basic_string_escapes_quotes_backslashes_and_control_characters():
    assert toml_basic_string('a "b" \\ c\nd\te\x01') == '"a \\"b\\" \\\\ c\\nd\\te\\u0001"'


def test_build_command_places_global_flags_before_exec(monkeypatch):
    _install_settings(monkeypatch, {"CONFIG.REASONING_EFFORT": "max", "CODEX.EXTRA_ARGS": ["--profile", "ci"]})
    handler = CodexAIHandler()
    command = handler.build_command("gpt-5.6", "SYS line1\nline2", "USER")

    assert command.argv == [
        "codex", "--ask-for-approval", "never", "--sandbox", "read-only", "--model", "gpt-5.6",
        "--config", 'developer_instructions="SYS line1\\nline2"',
        "--config", 'model_reasoning_effort="xhigh"',
        "exec", "--json", "--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
        "--color", "never", "--cd", handler.workdir, "--profile", "ci", "-",
    ]
    assert command.stdin == "USER"


def test_build_command_skips_unknown_effort(monkeypatch):
    _install_settings(monkeypatch, {"CONFIG.REASONING_EFFORT": "none"})
    command = CodexAIHandler().build_command("gpt-5.6", "SYS", "USER")
    assert not any(arg.startswith("model_reasoning_effort") for arg in command.argv)


def test_parse_response_takes_the_agent_message_and_turn_usage(monkeypatch):
    _install_settings(monkeypatch)
    stdout = _events(
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"id": "i1", "type": "reasoning", "text": "thinking"}},
        {"type": "item.completed", "item": {"id": "i2", "type": "agent_message", "text": "review text"}},
        {"type": "turn.completed", "usage": {"input_tokens": 14541, "cached_input_tokens": 11520,
                                             "output_tokens": 6, "reasoning_output_tokens": 4}},
    )
    response = CodexAIHandler().parse_response(stdout)

    assert response.text == "review text"
    assert response.finish_reason == "stop"
    assert response.prompt_tokens == 14541
    assert response.completion_tokens == 10
    assert response.cost_usd is None


def test_parse_response_raises_on_turn_failed(monkeypatch):
    _install_settings(monkeypatch)
    with pytest.raises(CliHandlerError, match="quota"):
        CodexAIHandler().parse_response(_events({"type": "turn.failed", "error": {"message": "quota exceeded"}}))


def test_parse_response_raises_on_error_event(monkeypatch):
    _install_settings(monkeypatch)
    with pytest.raises(CliHandlerError, match="bad request"):
        CodexAIHandler().parse_response(_events({"type": "error", "message": "bad request"}))


def test_parse_response_raises_without_an_agent_message(monkeypatch):
    _install_settings(monkeypatch)
    with pytest.raises(CliHandlerError, match="no agent message"):
        CodexAIHandler().parse_response(_events({"type": "turn.completed", "usage": {}}))


def test_parse_response_skips_non_object_json_and_non_dict_item_or_usage(monkeypatch):
    _install_settings(monkeypatch)
    stdout = _events(
        None,
        42,
        ["x"],
        {"type": "item.completed", "item": "not-a-dict"},
        {"type": "turn.completed", "usage": ["not", "a", "dict"]},
        {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "review text"}},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2, "reasoning_output_tokens": 1}},
    )
    response = CodexAIHandler().parse_response(stdout)

    assert response.text == "review text"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 3


def test_workdir_is_removed_when_the_handler_is_garbage_collected(monkeypatch):
    _install_settings(monkeypatch)
    handler = CodexAIHandler()
    workdir = handler.workdir
    assert os.path.isdir(workdir)

    del handler
    gc.collect()

    assert not os.path.exists(workdir)
