"""<video>.upload.json: everything needed to upload a render to YouTube."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Checkpoint, Strict


class SourceMeta(Strict):
    """What the camera recorded about the footage (from container tags)."""

    recorded_at: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    device: str | None = None
    place: str | None = None  # only with --geocode (sends the coordinates to OpenStreetMap)


class Thumbnail(Strict):
    path: str
    text: str
    style: Literal["bold", "question", "reveal"]
    t_out: float  # moment of the edit the frame comes from
    why: str = ""
    renderer: Literal["hyperframes", "ffmpeg"] = "ffmpeg"


class UploadPackage(Checkpoint):
    video: str
    titles: list[str]  # best first
    description: str  # ready to paste (chapters, credits and hashtags included)
    tags: list[str] = Field(default_factory=list)
    hashtags: list[str] = Field(default_factory=list)
    category: str = ""
    pinned_comment: str = ""
    chapters: list[tuple[float, str]] = Field(default_factory=list)
    credits: list[str] = Field(default_factory=list)  # credit lines that are in the description
    thumbnails: list[Thumbnail] = Field(default_factory=list)
    source: SourceMeta = Field(default_factory=SourceMeta)
    duration: float = 0.0
    width: int = 0
    height: int = 0
    warnings: list[str] = Field(default_factory=list)
