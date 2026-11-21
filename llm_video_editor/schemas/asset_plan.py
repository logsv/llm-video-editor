"""asset_plan.json: which source windows and B-roll the edit uses, resolved to times."""

from __future__ import annotations

from pydantic import Field

from .base import Checkpoint, Strict


class PlannedSegment(Strict):
    clip_id: str
    source_file: str
    start: float
    end: float
    first_segment_id: int | None = None
    last_segment_id: int | None = None
    purpose: str = ""
    priority: int = 1  # 1 = most important; higher numbers are dropped first

    @property
    def duration(self) -> float:
        return self.end - self.start


class BrollPlacement(Strict):
    over_segment: int  # index into AssetPlan.segments
    offset: float = 0.0  # seconds after that segment starts in the output
    duration: float
    query: str = ""
    clip_id: str | None = None
    source_file: str | None = None
    src_start: float = 0.0


class AssetPlan(Checkpoint):
    prompt: str
    target_platform: str
    target_duration: float | None = None
    segments: list[PlannedSegment] = Field(default_factory=list)
    broll: list[BrollPlacement] = Field(default_factory=list)
    notes: str = ""
