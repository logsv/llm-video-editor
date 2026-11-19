"""geometry_map.json: subject boxes per scene and safe placement per EDL clip.

All boxes are normalised ``[x_min, y_min, x_max, y_max]`` in 0..1. Source boxes
are relative to the source frame; ``subject_box_out`` is relative to the
rendered output frame after the crop/scale.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Checkpoint, Strict

BBox = tuple[float, float, float, float]
Corner = Literal["tl", "tr", "bl", "br"]


class Detection(Strict):
    t: float  # source seconds
    label: Literal["face", "person"]
    bbox: BBox
    score: float


class SceneGeometry(Strict):
    start: float
    end: float
    detections: list[Detection] = Field(default_factory=list)
    focus: tuple[float, float] | None = None  # normalised point to keep centred
    subject_box: BBox | None = None  # union of the primary subject across samples


class ClipGeometry(Strict):
    clip_id: str
    source_file: str
    width: int
    height: int
    scenes: list[SceneGeometry] = Field(default_factory=list)


class ClipPlacement(Strict):
    clip_id: str  # EDL clip id
    crop_center: tuple[float, float]
    subject_box_out: BBox | None = None
    caption_position: Literal["bottom", "top", "center"] = "bottom"
    caption_overlap: float = 0.0  # fraction of the caption band covered by the subject
    pip_corner: Corner = "br"


class GeometryMap(Checkpoint):
    detector: str
    edl_hash: str
    clips: list[ClipGeometry] = Field(default_factory=list)
    placements: list[ClipPlacement] = Field(default_factory=list)

    def placement(self, clip_id: str) -> ClipPlacement | None:
        return next((p for p in self.placements if p.clip_id == clip_id), None)
