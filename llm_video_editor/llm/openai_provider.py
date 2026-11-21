"""OpenAI provider (``pip install llm-video-editor[openai]``)."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from .base import ProviderError, assert_text_only

M = TypeVar("M", bound=BaseModel)


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str | None = None):
        self.model = model or "gpt-4.1"
        self._client = None

    def complete(self, system: str, prompt: str, schema: type[M]) -> M:
        assert_text_only(system, prompt)
        try:
            from openai import OpenAI, OpenAIError
        except ImportError as e:
            raise ProviderError("Install the `openai` extra to use --provider openai") from e
        if self._client is None:
            self._client = OpenAI()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema.__name__,
                                    "schema": schema.model_json_schema()},
                },
            )
        except OpenAIError as e:
            raise ProviderError(f"OpenAI error: {e}") from e
        return schema.model_validate_json(resp.choices[0].message.content or "")
