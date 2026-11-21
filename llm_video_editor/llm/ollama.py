"""Local Ollama provider using /api/chat with a JSON-schema ``format``."""

from __future__ import annotations

from typing import TypeVar

import requests
from pydantic import BaseModel

from .base import ProviderError, assert_text_only

M = TypeVar("M", bound=BaseModel)


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str | None = None, base_url: str = "http://localhost:11434",
                 timeout: float = 600):
        self.model = model or "qwen2.5:14b-instruct"
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, system: str, prompt: str, schema: type[M]) -> M:
        assert_text_only(system, prompt)
        body = {
            "model": self.model,
            "stream": False,
            "format": schema.model_json_schema(),
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise ProviderError(f"Ollama request failed: {e}") from e
        content = r.json().get("message", {}).get("content", "")
        return schema.model_validate_json(content)
