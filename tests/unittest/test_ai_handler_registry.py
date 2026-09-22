import pytest

import pr_agent.algo.ai_handlers.registry as registry
from pr_agent.agent.pr_agent import PRAgent
from pr_agent.algo.ai_handlers.claude_code_ai_handler import ClaudeCodeAIHandler
from pr_agent.algo.ai_handlers.codex_ai_handler import CodexAIHandler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.algo.ai_handlers.openai_ai_handler import OpenAIHandler
from pr_agent.config_loader import get_settings


@pytest.fixture
def ai_handler_setting():
    settings = get_settings()
    before = settings.get("CONFIG.AI_HANDLER", None)
    yield settings
    settings.set("config.ai_handler", before if before is not None else "litellm")


def test_default_is_litellm(ai_handler_setting):
    ai_handler_setting.set("config.ai_handler", "litellm")
    assert registry.resolve_ai_handler() is LiteLLMAIHandler


@pytest.mark.parametrize("name, expected", [
    ("openai", OpenAIHandler),
    ("claude_code", ClaudeCodeAIHandler),
    ("codex", CodexAIHandler),
    (" Codex ", CodexAIHandler),
])
def test_named_handlers_resolve(name, expected):
    assert registry.resolve_ai_handler(name) is expected


def test_config_value_selects_the_handler(ai_handler_setting):
    ai_handler_setting.set("config.ai_handler", "claude_code")
    assert registry.resolve_ai_handler() is ClaudeCodeAIHandler


def test_unknown_handler_raises_with_the_known_names():
    with pytest.raises(ValueError, match="claude_code, codex, langchain, litellm, openai"):
        registry.resolve_ai_handler("gemini")


def test_pr_agent_resolves_from_config_when_nothing_is_injected(ai_handler_setting):
    ai_handler_setting.set("config.ai_handler", "codex")
    assert PRAgent().ai_handler is CodexAIHandler


def test_pr_agent_keeps_an_injected_handler(ai_handler_setting):
    ai_handler_setting.set("config.ai_handler", "codex")
    assert PRAgent(ai_handler=OpenAIHandler).ai_handler is OpenAIHandler
