"""Anthropic API provider (``pip install llm-video-editor[anthropic]``)."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from .base import ProviderError, assert_text_only

M = TypeVar("M", bound=BaseModel)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None, effort: str = "medium"):
        self.model = model or "claude-opus-5-5"
        self.effort = effort
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ProviderError("Install the `anthropic` extra to use --provider anthropic") from e
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, prompt: str, schema: type[M]) -> M:
        assert_text_only(system, prompt)
        import anthropic

        try:
            response = self._get_client().messages.parse(
                model=self.model,
                max_tokens=16000,
                system=system,
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
            )
        except anthropic.APIError as e:
            raise ProviderError(f"Anthropic API error: {e}") from e
        if response.stop_reason == "refusal":
            raise ProviderError("Anthropic model declined the planning request")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise ProviderError(f"Anthropic response incomplete (stop_reason={response.stop_reason})")
        return response.parsed_output
