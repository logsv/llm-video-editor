"""Pydantic v2 schemas for every JSON checkpoint the pipeline writes."""

from __future__ import annotations

from .asset_plan import AssetPlan, BrollPlacement, PlannedSegment
from .base import SCHEMA_VERSION, Checkpoint, Strict
from .broll import BrollIndex, BrollItem, BrollSelection, Candidate, SlotChoice
from .cut_audit import CutAudit, Violation
from .edl import EDL, CaptionSettings, EDLClip, Overlay
from .geometry import ClipGeometry, ClipPlacement, Detection, GeometryMap, SceneGeometry
from .manifest import ClipInfo, ClipsManifest, SceneCut
from .render import RenderQC
from .review import Review, ReviewTile
from .sheets import Sheet, SheetIndex, Tile
from .transcript import ClipTranscript, Segment, Transcript, Word

CHECKPOINTS: dict[str, type[Checkpoint]] = {
    "transcript": Transcript,
    "clips_manifest": ClipsManifest,
    "sheets_index": SheetIndex,
    "asset_plan": AssetPlan,
    "edl": EDL,
    "cut_audit": CutAudit,
    "qc_report": RenderQC,
    "geometry_map": GeometryMap,
    "review": Review,
    "broll_index": BrollIndex,
    "broll_select": BrollSelection,
}

__all__ = [
    "CHECKPOINTS",
    "EDL",
    "SCHEMA_VERSION",
    "AssetPlan",
    "BrollIndex",
    "BrollItem",
    "BrollPlacement",
    "BrollSelection",
    "Candidate",
    "CaptionSettings",
    "Checkpoint",
    "ClipGeometry",
    "ClipInfo",
    "ClipPlacement",
    "ClipTranscript",
    "ClipsManifest",
    "CutAudit",
    "Detection",
    "EDLClip",
    "GeometryMap",
    "Overlay",
    "PlannedSegment",
    "RenderQC",
    "Review",
    "ReviewTile",
    "SceneCut",
    "SceneGeometry",
    "Segment",
    "Sheet",
    "SlotChoice",
    "SheetIndex",
    "Strict",
    "Tile",
    "Transcript",
    "Violation",
    "Word",
]
