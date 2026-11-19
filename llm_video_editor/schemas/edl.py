"""edl_raw.json / edl_audited.json: the timeline the renderer executes."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Checkpoint, Strict


class EDLClip(Strict):
    clip_id: str  # unique within the EDL, e.g. "c001"
    source_clip_id: str
    source_file: str
    start: float
    end: float
    role: Literal["a_roll", "broll"] = "a_roll"
    priority: int = 1
    crop_center: tuple[float, float] | None = None  # normalised (x, y); None = centre
    caption_position: Literal["bottom", "top", "center"] | None = None  # set by take-geometry
    notes: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Overlay(Strict):
    overlay_id: str
    kind: Literal["broll"] = "broll"
    source_clip_id: str
    source_file: str
    src_start: float = 0.0
    anchor_clip_id: str  # EDLClip.clip_id the overlay sits on
    offset: float = 0.0  # seconds after the anchor clip starts in the output
    duration: float
    zone: Literal["full", "pip"] = "full"
    corner: Literal["tl", "tr", "bl", "br"] = "br"  # picture-in-picture corner


class CaptionSettings(Strict):
    enabled: bool = True
    style: Literal["lines", "word_highlight"] = "lines"
    max_words_per_line: int = 6
    position: Literal["auto", "bottom", "top", "center"] = "auto"


class EDL(Checkpoint):
    target_platform: str
    width: int
    height: int
    fps: int = 30
    target_duration: float | None = None
    fit_mode: Literal["cover", "contain"] = "cover"
    clips: list[EDLClip] = Field(default_factory=list)
    overlays: list[Overlay] = Field(default_factory=list)
    captions: CaptionSettings = Field(default_factory=CaptionSettings)

    @property
    def total_duration(self) -> float:
        return sum(c.duration for c in self.clips)

    def output_starts(self) -> list[float]:
        """Output-timeline start time of every clip."""
        starts, t = [], 0.0
        for c in self.clips:
            starts.append(t)
            t += c.duration
        return starts

    def overlay_start(self, overlay: Overlay) -> float | None:
        """Output-timeline start of an overlay, or None if its anchor is gone."""
        for clip, start in zip(self.clips, self.output_starts(), strict=True):
            if clip.clip_id == overlay.anchor_clip_id:
                return start + overlay.offset
        return None
