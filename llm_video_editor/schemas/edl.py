"""edl_raw.json / edl_audited.json: the timeline the renderer executes."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

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
    zoom: float = Field(1.0, ge=1.0, le=2.0)  # punch-in on top of the fit (1.0 = none)
    chapter: str | None = None  # a YouTube chapter starts at this clip
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


class Graphic(Strict):
    """A HyperFrames motion graphic (rendered to transparent WebM) over the edit."""

    graphic_id: str
    template: str | None = None  # built-in: lower-third, title-card, callout
    composition: str | None = None  # or a HyperFrames project folder (has index.html)
    variables: dict[str, str | float | bool] = Field(default_factory=dict)
    anchor_clip_id: str  # EDLClip.clip_id the graphic starts on
    offset: float = 0.0  # seconds after the anchor clip starts in the output
    duration: float = Field(gt=0)  # may run past the anchor clip
    box: tuple[float, float, float, float] | None = None  # x, y, w, h (0-1); None = full frame

    @model_validator(mode="after")
    def _one_source(self):
        if (self.template is None) == (self.composition is None):
            raise ValueError("set exactly one of template or composition")
        return self


class CaptionSettings(Strict):
    enabled: bool = True
    style: Literal["lines", "word_highlight"] = "lines"
    max_words_per_line: int = 6
    position: Literal["auto", "bottom", "top", "center"] = "auto"
    # ASR fixes applied to burned-in and sidecar captions, e.g. {"cloud": "Claude"}
    replacements: dict[str, str] = Field(default_factory=dict)


class Pacing(Strict):
    """Jump-cut tightening and punch-ins, applied by cut-audit after the editorial fixes."""

    tighten: bool = False  # remove pauses longer than max_gap inside A-roll clips
    max_gap: float = Field(0.6, gt=0.0)
    lead: float = 0.08  # air kept before the first word of each speech island
    tail: float = 0.18  # air kept after the last word
    min_island: float = 0.25  # islands shorter than this are dropped
    drop_fillers: bool = False  # treat standalone um/uh as pauses
    punch_in: Literal["off", "alternate"] = "off"  # alternate zoom on consecutive jump cuts
    punch_zoom: float = Field(1.12, ge=1.0, le=2.0)


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
    pacing: Pacing = Field(default_factory=Pacing)
    graphics: list[Graphic] = Field(default_factory=list)

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

    def anchor_start(self, clip_id: str) -> float | None:
        """Output-timeline start of a clip, or None if it is not in the cut."""
        for clip, start in zip(self.clips, self.output_starts(), strict=True):
            if clip.clip_id == clip_id:
                return start
        return None

    def overlay_start(self, overlay: Overlay) -> float | None:
        """Output-timeline start of an overlay, or None if its anchor is gone."""
        for clip, start in zip(self.clips, self.output_starts(), strict=True):
            if clip.clip_id == overlay.anchor_clip_id:
                return start + overlay.offset
        return None
