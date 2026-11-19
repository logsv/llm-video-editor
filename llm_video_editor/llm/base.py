"""Provider-neutral structured LLM calls.

Every provider returns a validated Pydantic model. Prompts are checked against
the zero-upload boundary first: only text (transcripts, scene metadata, B-roll
index summaries) leaves the machine, never media paths or binary data.
"""

from __future__ import annotations

import re
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)

_MEDIA_REF = re.compile(
    r"[^\s\"'<>]*\.(?:mp4|mov|mkv|avi|m4v|webm|mxf|wav|mp3|m4a|aac|flac)\b", re.IGNORECASE
)


class ProviderError(RuntimeError):
    """The provider failed or returned something unusable."""


class BoundaryError(ValueError):
    """A prompt tried to cross the zero-upload boundary."""


def assert_text_only(*texts: str) -> None:
    """Refuse prompts that mention media file paths or carry binary data."""
    for text in texts:
        if "\x00" in text:
            raise BoundaryError("binary data in prompt")
        if m := _MEDIA_REF.search(text):
            raise BoundaryError(
                f"prompt references a media file ({m.group(0)!r}); send clip ids, not paths"
            )


class Provider(Protocol):
    name: str

    def complete(self, system: str, prompt: str, schema: type[M]) -> M:
        """Return ``schema`` validated from the model's structured output."""
        ...


def structured(provider: Provider, system: str, prompt: str, schema: type[M],
               retries: int = 1) -> M:
    """Call ``provider`` with the boundary check and one validation-repair retry."""
    assert_text_only(system, prompt)
    attempt_prompt = prompt
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            return provider.complete(system, attempt_prompt, schema)
        except (ValidationError, ProviderError) as e:
            last = e
            attempt_prompt = (
                f"{prompt}\n\nYour previous answer was rejected:\n{str(e)[:1500]}\n"
                "Return only output that matches the schema exactly."
            )
    raise ProviderError(f"{provider.name} failed after {retries + 1} attempts: {last}")
