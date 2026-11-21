"""Royalty-free media: search results and the credits a published video must carry."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Checkpoint, Strict

MediaKind = Literal["music", "sfx", "video"]


class MediaResult(Strict):
    result_id: str  # "<provider>:<id>", stable across searches
    provider: str  # openverse | pixabay | pexels
    kind: MediaKind
    title: str
    creator: str = ""
    creator_url: str = ""
    license: str  # e.g. "cc0 1.0", "by 4.0", "pixabay", "pexels"
    license_url: str = ""
    attribution: str = ""  # the credit line to paste (provider text when it gives one)
    attribution_required: bool = False
    landing_url: str = ""  # human page for the item
    download_url: str
    preview_url: str = ""
    duration: float | None = None  # seconds
    width: int | None = None
    height: int | None = None
    tags: list[str] = Field(default_factory=list)


class MediaSearch(Checkpoint):
    query: str
    kind: MediaKind
    results: list[MediaResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)  # per-provider failures (missing key, HTTP)


class Credit(Strict):
    file: str  # absolute path of the local copy
    kind: MediaKind
    title: str
    creator: str = ""
    provider: str  # openverse | pixabay | pexels | youtube-audio-library | manual
    license: str
    license_url: str = ""
    landing_url: str = ""
    attribution: str = ""
    attribution_required: bool = False
    added_at: str = ""


class Credits(Checkpoint):
    items: list[Credit] = Field(default_factory=list)

    def for_file(self, path: str) -> Credit | None:
        return next((c for c in self.items if c.file == path), None)
