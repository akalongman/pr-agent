import asyncio
import os
import sys
import time
from types import SimpleNamespace

import pytest

import pr_agent.algo.ai_handlers.cli_ai_handler as cli_module
from pr_agent.algo.ai_handlers.cli_ai_handler import (
    MAX_ARGV_ELEMENT_BYTES,
    CliAIHandler,
    CliCommand,
    CliHandlerError,
    CliResponse,
)
from pr_agent.algo.run_details import get_run_details, init_run_details


class _FakeSettings:
    def __init__(self, values=None, ai_timeout=30):
        self.values = values or {}
        self.config = SimpleNamespace(ai_timeout=ai_timeout)

    def get(self, key, default=None):
        return self.values.get(key, default)


class _EchoAdapter(CliAIHandler):
    settings_section = "echo"
    default_binary = "echo-bin"

    def __init__(self):
        super().__init__()
        self.seen_command = None
        self.canned_stdout = "canned"

    def build_command(self, model, system, user):
        return CliCommand(argv=[self.binary, "--model", model, "--system", system, *self.extra_args], stdin=user)

    def parse_response(self, stdout):
        return CliResponse(text=stdout.upper(), finish_reason="stop", prompt_tokens=7, completion_tokens=3,
                           cost_usd=0.5)

    async def _run(self, command):
        self.seen_command = command
        return self.canned_stdout


def _install_settings(monkeypatch, values=None, ai_timeout=30):
    monkeypatch.setattr(cli_module, "get_settings", lambda: _FakeSettings(values, ai_timeout))


def test_settings_section_supplies_binary_extra_args_and_timeout(monkeypatch):
    _install_settings(monkeypatch, {"ECHO.BINARY": "/opt/echo", "ECHO.EXTRA_ARGS": ["--x", 1], "ECHO.TIMEOUT": 5})
    handler = _EchoAdapter()
    assert handler.binary == "/opt/echo"
    assert handler.extra_args == ["--x", "1"]
    assert handler.timeout == 5.0
    assert handler.deployment_id is None


def test_extra_args_as_string_is_shell_split_into_the_built_command(monkeypatch):
    # Shell-split an env-var-supplied value (e.g. CLAUDE_CODE__EXTRA_ARGS=--verbose), which Dynaconf
    # keeps a plain string rather than a list, instead of iterating it character by character.
    _install_settings(monkeypatch, {"ECHO.EXTRA_ARGS": "--verbose --foo 'a b'"})
    handler = _EchoAdapter()

    assert handler.extra_args == ["--verbose", "--foo", "a b"]
    command = handler.build_command("m", "sys", "usr")
    assert command.argv[-3:] == ["--verbose", "--foo", "a b"]


def test_unparseable_extra_args_string_raises_naming_the_setting(monkeypatch):
    _install_settings(monkeypatch, {"ECHO.EXTRA_ARGS": "--foo 'unbalanced"})

    with pytest.raises(CliHandlerError, match=r"echo\.extra_args"):
        _EchoAdapter()


def test_defaults_fall_back_to_class_binary_and_config_timeout(monkeypatch):
    _install_settings(monkeypatch, {}, ai_timeout=42)
    handler = _EchoAdapter()
    assert handler.binary == "echo-bin"
    assert handler.extra_args == []
    assert handler.timeout == 42.0


async def test_chat_completion_runs_command_and_records_usage(monkeypatch):
    _install_settings(monkeypatch)
    init_run_details()
    handler = _EchoAdapter()

    response = await handler.chat_completion(model="anthropic/claude-opus-5", system="sys", user="usr")

    assert response == ("CANNED", "stop")
    assert handler.seen_command.argv == ["echo-bin", "--model", "claude-opus-5", "--system", "sys"]
    assert handler.seen_command.stdin == "usr"
    details = get_run_details()
    assert details.num_ai_calls == 1
    assert details.prompt_tokens == 7
    assert details.completion_tokens == 3
    assert details.total_tokens == 10


async def test_image_path_is_ignored_with_a_warning(monkeypatch):
    _install_settings(monkeypatch)
    warnings = []
    monkeypatch.setattr(cli_module, "get_logger", lambda: SimpleNamespace(
        warning=lambda msg, **kw: warnings.append(msg), info=lambda *a, **kw: None, debug=lambda *a, **kw: None))
    handler = _EchoAdapter()

    await handler.chat_completion(model="m", system="s", user="u", img_path="/tmp/x.png")

    assert any("Ignoring image path" in message for message in warnings)


async def test_chat_completion_logs_the_prompts_as_a_debug_artifact(monkeypatch):
    _install_settings(monkeypatch)
    debug_calls = []
    monkeypatch.setattr(cli_module, "get_logger", lambda: SimpleNamespace(
        debug=lambda msg, **kw: debug_calls.append((msg, kw)), info=lambda *a, **kw: None))

    await _EchoAdapter().chat_completion(model="m", system="sys", user="usr")

    assert debug_calls == [("Prompts", {"artifact": {"system": "sys", "user": "usr"}})]


async def test_oversized_argv_element_raises_before_running(monkeypatch):
    _install_settings(monkeypatch)
    handler = _EchoAdapter()

    with pytest.raises(CliHandlerError):
        await handler.chat_completion(model="m", system="x" * (MAX_ARGV_ELEMENT_BYTES + 1), user="u")
    assert handler.seen_command is None


class _BrokenParserAdapter(_EchoAdapter):
    def parse_response(self, stdout):
        raise ValueError("unexpected output shape")


async def test_parse_failure_surfaces_as_cli_handler_error(monkeypatch):
    _install_settings(monkeypatch)

    with pytest.raises(CliHandlerError, match="_BrokenParserAdapter") as exc_info:
        await _BrokenParserAdapter().chat_completion(model="m", system="s", user="u")
    assert isinstance(exc_info.value.__cause__, ValueError)


async def test_cli_handler_error_from_parse_response_propagates_unwrapped(monkeypatch):
    _install_settings(monkeypatch)
    reported = CliHandlerError("reported failure")

    class _ReportingAdapter(_EchoAdapter):
        def parse_response(self, stdout):
            raise reported

    with pytest.raises(CliHandlerError) as exc_info:
        await _ReportingAdapter().chat_completion(model="m", system="s", user="u")
    assert exc_info.value is reported


class _RealProcessAdapter(_EchoAdapter):
    async def _run(self, command):
        return await CliAIHandler._run(self, command)


async def test_run_returns_stdout_of_a_successful_process(monkeypatch):
    _install_settings(monkeypatch)
    handler = _RealProcessAdapter()
    command = CliCommand(argv=[sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"], stdin="hi")

    assert (await handler._run(command)).strip() == "HI"


async def test_run_raises_on_non_zero_exit_with_stderr_tail(monkeypatch):
    _install_settings(monkeypatch)
    handler = _RealProcessAdapter()
    command = CliCommand(argv=[sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"])

    with pytest.raises(CliHandlerError, match="exited with 3: stderr: boom"):
        await handler._run(command)


async def test_run_raises_with_stdout_detail_on_non_zero_exit_when_stderr_is_empty(monkeypatch):
    # Carry the stdout detail in the raised message: Claude Code reports most failures (not logged
    # in, model not found, rate limit) as an `is_error` JSON result on stdout while exiting 1 and
    # writing nothing to stderr, so the stderr tail alone would come up empty.
    _install_settings(monkeypatch)
    handler = _RealProcessAdapter()
    script = (
        "import sys; "
        "sys.stdout.write('{\"type\": \"result\", \"is_error\": true, \"result\": \"boom\"}'); "
        "sys.exit(1)"
    )
    command = CliCommand(argv=[sys.executable, "-c", script])

    with pytest.raises(CliHandlerError) as exc_info:
        await handler._run(command)
    message = str(exc_info.value)
    assert "exited with 1" in message
    assert '"is_error": true' in message
    assert '"boom"' in message
    assert "stdout:" in message
    assert "stderr:" not in message


async def test_run_raises_on_timeout(monkeypatch):
    _install_settings(monkeypatch, {"ECHO.TIMEOUT": 0.2})
    handler = _RealProcessAdapter()
    command = CliCommand(argv=[sys.executable, "-c", "import time; time.sleep(5)"])

    with pytest.raises(CliHandlerError, match="timed out"):
        await handler._run(command)


async def test_run_raises_when_binary_is_missing(monkeypatch):
    _install_settings(monkeypatch)
    handler = _RealProcessAdapter()

    with pytest.raises(CliHandlerError, match="command not found"):
        await handler._run(CliCommand(argv=["/nonexistent/pr-agent-cli-binary"]))


async def test_run_raises_when_binary_is_not_executable(monkeypatch, tmp_path):
    _install_settings(monkeypatch)
    handler = _RealProcessAdapter()
    binary = tmp_path / "not-executable"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o644)

    with pytest.raises(CliHandlerError, match="_RealProcessAdapter") as exc_info:
        await handler._run(CliCommand(argv=[str(binary)]))
    message = str(exc_info.value)
    assert str(binary) in message
    assert "Permission denied" in message
    assert isinstance(exc_info.value.__cause__, PermissionError)


async def test_run_kills_process_tree_on_timeout(monkeypatch):
    # Spawn a grandchild from the direct child that inherits stdout/stderr and outlives it: a plain
    # process.kill() on the direct child would leave the grandchild holding the pipes open, so
    # communicate() would not see EOF until the grandchild's multi-second sleep ends.
    _install_settings(monkeypatch, {"ECHO.TIMEOUT": 0.3})
    handler = _RealProcessAdapter()
    script = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)']); "
        "time.sleep(5)"
    )
    command = CliCommand(argv=[sys.executable, "-c", script])

    start = time.monotonic()
    with pytest.raises(CliHandlerError, match="timed out"):
        await handler._run(command)
    elapsed = time.monotonic() - start

    assert elapsed < 2


async def test_run_kills_process_group_on_cancellation(monkeypatch, tmp_path):
    _install_settings(monkeypatch)
    handler = _RealProcessAdapter()
    pid_file = tmp_path / "pid"
    script = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(5)"
    command = CliCommand(argv=[sys.executable, "-c", script])

    task = asyncio.ensure_future(handler._run(command))
    while not pid_file.exists():
        await asyncio.sleep(0.01)
    pid = int(pid_file.read_text())

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
