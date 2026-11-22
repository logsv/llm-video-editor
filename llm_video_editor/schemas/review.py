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


class GraphicCheck(Strict):
    """How a motion graphic sits on the frame under it (measured, not decided)."""

    graphic_id: str
    t_out: float
    bbox: BBox | None = None  # painted area, 0-1 of the frame
    contrast: float | None = None  # WCAG ratio between the text and what is behind it
    backdrop: Literal["panel", "video", "none"] = "none"  # its own card, or the shot
    busy: float | None = None  # luminance spread of the shot behind it (0-0.5)
    issues: list[str] = Field(default_factory=list)
    palette: list[str] = Field(default_factory=list)  # the shot's main colours (hex)
    suggested: dict[str, str] = Field(default_factory=dict)  # match / contrast / text colours


class Review(Checkpoint):
    status: Literal["pending", "approved", "rejected"] = "pending"
    edl_hash: str
    geometry_hash: str
    sheets: list[str] = Field(default_factory=list)  # relative to the workspace
    tiles: list[ReviewTile] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    graphic_checks: list[GraphicCheck] = Field(default_factory=list)
    notes: str = ""
    decided_at: str | None = None
