"""B-roll library index (stored next to the footage) and per-edit selection report."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Checkpoint, Strict
from .manifest import SceneCut


class BrollItem(Strict):
    item_id: str  # same id scheme as clip ids: path-free, stable per (path, mtime, size)
    rel_path: str  # relative to the library root
    mtime_ns: int
    size: int
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    orientation: Literal["landscape", "portrait", "square"]
    motion_score: float = 0.0
    scenes: list[SceneCut] = Field(default_factory=list)
    thumbnail: str = ""  # relative to the library root
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    described_by: Literal["filename", "sidecar", "manual"] = "filename"


class BrollIndex(Checkpoint):
    root: str
    updated_at: str = ""
    items: list[BrollItem] = Field(default_factory=list)

    def get(self, item_id: str) -> BrollItem | None:
        return next((i for i in self.items if i.item_id == item_id), None)


class Candidate(Strict):
    item_id: str
    score: float


class SlotChoice(Strict):
    slot: int
    over_segment: int
    query: str
    duration: float
    candidates: list[Candidate] = Field(default_factory=list)
    chosen: str | None = None
    reason: str = ""
    generated: bool = False


class BrollSelection(Checkpoint):
    library: str | None = None
    reranked_by: str | None = None
    slots: list[SlotChoice] = Field(default_factory=list)
