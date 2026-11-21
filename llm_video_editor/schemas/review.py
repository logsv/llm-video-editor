"""review/review.json: the pre-render preview and its approval state."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Checkpoint, Strict
from .geometry import BBox


class ReviewTile(Strict):
    index: int
    clip_id: str
    overlay_id: str | None = None
    graphic_id: str | None = None
    t_out: float  # output-timeline seconds shown in the tile
    source_clip_id: str
    src_t: float
    caption: str = ""
    caption_position: str = "bottom"
    subject_box_out: BBox | None = None


class Review(Checkpoint):
    status: Literal["pending", "approved", "rejected"] = "pending"
    edl_hash: str
    geometry_hash: str
    sheets: list[str] = Field(default_factory=list)  # relative to the workspace
    tiles: list[ReviewTile] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: str = ""
    decided_at: str | None = None
