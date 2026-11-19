"""LLM providers for the planner. ``claude-cli`` is the default."""

from __future__ import annotations

from .base import BoundaryError, Provider, ProviderError, assert_text_only, structured

PROVIDERS = ("claude-cli", "ollama", "anthropic", "openai")
DEFAULT_PROVIDER = "claude-cli"


def get_provider(name: str = DEFAULT_PROVIDER, model: str | None = None, **kwargs) -> Provider:
    if name == "claude-cli":
        from .claude_cli import ClaudeCLIProvider

        return ClaudeCLIProvider(model=model, **kwargs)
    if name == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(model=model, **kwargs)
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=model, **kwargs)
    if name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model=model, **kwargs)
    raise ValueError(f"Unknown provider {name!r}; choose from {', '.join(PROVIDERS)}")


__all__ = [
    "DEFAULT_PROVIDER",
    "PROVIDERS",
    "BoundaryError",
    "Provider",
    "ProviderError",
    "assert_text_only",
    "get_provider",
    "structured",
]
