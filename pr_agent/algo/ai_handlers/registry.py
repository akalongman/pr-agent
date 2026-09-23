"""Map AI handler names to their classes and resolve the one ``config.ai_handler`` selects (default ``litellm``)."""
from importlib import import_module
from typing import Optional

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.config_loader import get_settings

DEFAULT_AI_HANDLER = "litellm"

# Keep (module, class) pairs, like _BUILTIN_GIT_PROVIDERS, so the registry imports the selected
# handler's module on demand.
AI_HANDLERS: dict[str, tuple[str, str]] = {
    "litellm": ("pr_agent.algo.ai_handlers.litellm_ai_handler", "LiteLLMAIHandler"),
    "openai": ("pr_agent.algo.ai_handlers.openai_ai_handler", "OpenAIHandler"),
    "langchain": ("pr_agent.algo.ai_handlers.langchain_ai_handler", "LangChainOpenAIHandler"),
    "claude_code": ("pr_agent.algo.ai_handlers.claude_code_ai_handler", "ClaudeCodeAIHandler"),
    "codex": ("pr_agent.algo.ai_handlers.codex_ai_handler", "CodexAIHandler"),
}


def resolve_ai_handler(name: Optional[str] = None) -> type[BaseAiHandler]:
    """Return the handler class named by ``name`` or, when omitted, by ``config.ai_handler``."""
    key = str(name or get_settings().get("CONFIG.AI_HANDLER", None) or DEFAULT_AI_HANDLER).strip().lower()
    if key not in AI_HANDLERS:
        raise ValueError(f"Unknown config.ai_handler '{key}'; known handlers: {', '.join(sorted(AI_HANDLERS))}")
    module_name, class_name = AI_HANDLERS[key]
    handler = getattr(import_module(module_name), class_name)
    if not (isinstance(handler, type) and issubclass(handler, BaseAiHandler)):
        raise TypeError(f"{module_name}.{class_name} is not a BaseAiHandler")
    return handler
