"""clips_manifest.json: one entry per source clip, with A-roll/B-roll classification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Checkpoint, Strict

ClipKind = Literal["talking_head", "broll"]


class SceneCut(Strict):
    start: float
    end: float


class ClipInfo(Strict):
    clip_id: str
    source_file: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    video_codec: str | None = None
    audio_codec: str | None = None
    kind: ClipKind = "talking_head"
    speech_seconds: float = 0.0
    speech_ratio: float = 0.0
    scene_change_rate: float = 0.0  # scene cuts per minute
    motion_score: float = 0.0  # mean ffmpeg scene score, 0..1
    scenes: list[SceneCut] = Field(default_factory=list)
    overview_sheets: list[str] = Field(default_factory=list)

    @property
    def orientation(self) -> Literal["landscape", "portrait", "square"]:
        if self.width > self.height:
            return "landscape"
        if self.height > self.width:
            return "portrait"
        return "square"


class ClipsManifest(Checkpoint):
    batch_hash: str
    clips: list[ClipInfo] = Field(default_factory=list)

    def get(self, clip_id: str) -> ClipInfo | None:
        return next((c for c in self.clips if c.clip_id == clip_id), None)

    def by_source(self, source_file: str) -> ClipInfo | None:
        return next((c for c in self.clips if c.source_file == source_file), None)

    @property
    def total_duration(self) -> float:
        return sum(c.duration for c in self.clips)
