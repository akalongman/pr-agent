# Extending PR-Agent

Contributors extending a model, git provider, or tool start here. To only change
the model, use [Changing a Model](./changing_a_model.md).

## Adding a model

Tool calls go through LiteLLM (`pr_agent/algo/ai_handlers/litellm_ai_handler.py`), the default handler. Most models need only configuration:

```toml
[config]
model="<model-name>"
fallback_models=["<fallback-model-name>"]
```

Set these under `[config]` in `pr_agent/settings/configuration.toml`.
Keep model names in configuration, not in tool code.

Models that behave differently are registered in `pr_agent/algo/__init__.py`:
`NO_SUPPORT_TEMPERATURE_MODELS` for models that reject a temperature
parameter; `CLAUDE_EXTENDED_THINKING_MODELS` for Claude models that
take extended thinking. For Claude models with provider-prefixed aliases
(bare, `anthropic/`, `vertex_ai/`, `bedrock/`), declare the canonical family
in `_CLAUDE_MODEL_FAMILIES` to expand them across registries automatically.
Other models can be added directly to the matching list.

Context windows are registered in `MAX_TOKENS` in `pr_agent/algo/__init__.py`:
Claude model families declared in `_CLAUDE_MODEL_FAMILIES` populate their
aliases automatically. For other models, add the model name and its
context-window token count to `MAX_TOKENS`, or set `config.custom_model_max_tokens`
in `configuration.toml`. Without either, `get_max_tokens()` raises.

Verify with `PYTHONPATH=. uv run pytest tests/unittest`.

## Adding an AI handler

Handlers implement `BaseAiHandler` (`pr_agent/algo/ai_handlers/base_ai_handler.py`) and are
selected by name through `config.ai_handler`, resolved by `resolve_ai_handler` in
`pr_agent/algo/ai_handlers/registry.py`. Register a new one as a lazy `(module, class)` pair
in `AI_HANDLERS`, so its dependencies load only when it is selected.

A handler that runs a local command-line agent subclasses `CliAIHandler`
(`pr_agent/algo/ai_handlers/cli_ai_handler.py`), which owns the subprocess, the timeout
(`config.ai_timeout` unless the section sets `timeout`), failure mapping, logging and usage
accounting. The subclass declares `settings_section` (a TOML section with `binary`,
`extra_args` and `timeout`) and `default_binary`, and implements:

- `build_command(model, system, user) -> CliCommand`: the full argv (binary first) and the
  text for stdin. Keep the large prompt on stdin; a single argv element above 120,000 bytes
  raises rather than truncating.
- `parse_response(stdout) -> CliResponse`: the text, a finish reason (`stop` or `length`),
  prompt and completion token counts, and a cost when the CLI reports one. Raise
  `CliHandlerError` on a reported failure so `fallback_models` still work.
- `map_model(model)` when stripping the provider prefix is not enough.

Every CLI handler section must be added to `REPO_OVERRIDABLE_KEYS_BY_HOST_SECTION` in
`pr_agent/config_security.py` (an empty `frozenset()`, the same idiom as `push_outputs` and
`prompt_fragments`): the handler runs a configurable binary, so a repository's
`.pr_agent.toml` or a comment argument must never be able to select or configure it.

Tests: cover `build_command` and `parse_response` with canned output in a
`tests/unittest/test_<name>_ai_handler.py`, add the name to
`tests/unittest/test_ai_handler_registry.py`, and add the section to the host-only cases in
`tests/unittest/test_apply_repo_settings_security.py` and `tests/unittest/test_cli_args_security.py`.
`claude_code_ai_handler.py` and `codex_ai_handler.py` are the two shipped examples.

## Adding a git provider

Implement a `GitProvider` subclass and register it:

1. Create `pr_agent/git_providers/<name>_provider.py`, extending the interface in `pr_agent/git_providers/git_provider.py` (`gitlab_provider.py` is the reference).
2. Add the built-in provider to `_BUILTIN_GIT_PROVIDERS` in `pr_agent/git_providers/__init__.py` as a `(module_path, class_name)` pair. Built-ins are imported lazily when selected. Keys already used: `github`, `gitlab`, `bitbucket`, `bitbucket_server`, `azure`, `codecommit`, `local`, `gerrit`, `gitea`, `plain-diff`.
3. Select it via `[config]` → `git_provider="<name>"` in `pr_agent/settings/configuration.toml`.
4. Add `docs/docs/installation/<name>.md` (see [`gitlab.md`](../installation/gitlab.md)) and register it under `Installation` in both `docs/mkdocs.yml` and `docs/docs/summary.md`.
5. Select provider-dependent behavior with capability checks like `provider.is_supported("feature")` rather than provider-type checks.
6. Add unit tests under `tests/unittest/test_<name>_provider.py` (see `test_bitbucket_provider.py`) and list the required env vars in `pr_agent/settings/.secrets_template.toml`.

### Registering a provider from another package

A provider does not have to live in this repository. Call `register_git_provider` from your own package before PR-Agent resolves the provider, for example from the module that starts your server or wraps the CLI:

```python
from pr_agent.git_providers import register_git_provider

from my_package.forge_provider import ForgeProvider

register_git_provider("forge", ForgeProvider)
```

Then select it with `git_provider="forge"` under `[config]`. The class must extend `GitProvider`. Registering the same class twice is a no-op, and registering a different class under an id that is already taken raises, so a package cannot silently replace a built-in provider.

## Adding a tool

1. Implement the tool class in `pr_agent/tools/pr_<name>.py` with an `async def run(self)` entry point (see `pr_reviewer.py`).
2. Add a `[pr_<tool>]` section in `pr_agent/settings/configuration.toml` for the option keys the tool reads (`[pr_reviewer]` is the pattern to follow).
3. Add a prompt TOML under `pr_agent/settings/` and register it in the `settings_files=[...]` list in `pr_agent/config_loader.py` — it is not loaded otherwise.
4. Match the TOML section name to the settings key the tool reads: `[pr_review_prompt]` in `pr_reviewer_prompts.toml` ↔ `get_settings().pr_review_prompt` in `pr_reviewer.py`.
5. Register the tool in `command2class` in `pr_agent/agent/pr_agent.py` under a command name, e.g. `"my_tool": PRMyTool`. Then add it to the hardcoded help surfaces, or it will not show up in `/help`: `pr_agent/tools/pr_help_message.py`, `pr_agent/servers/help.py`, and the command list in `pr_agent/cli.py`.
6. Add a row to the tool list in `docs/docs/tools/index.md`, a page `docs/docs/tools/<name>.md` (see [`review.md`](../tools/review.md)), and register the page under `Tools` in both `docs/mkdocs.yml` and `docs/docs/summary.md`.
7. Add tests under `tests/unittest/` and verify with `PYTHONPATH=. uv run pytest tests/unittest`.
