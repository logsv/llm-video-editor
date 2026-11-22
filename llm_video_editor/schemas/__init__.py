"""Pydantic v2 schemas for every JSON checkpoint the pipeline writes."""

from __future__ import annotations

from .asset_plan import AssetPlan, BrollPlacement, PlannedSegment
from .base import SCHEMA_VERSION, Checkpoint, Strict
from .broll import BrollIndex, BrollItem, BrollSelection, Candidate, SlotChoice
from .cut_audit import CutAudit, Violation
from .edl import EDL, CaptionSettings, EDLClip, Graphic, MixSettings, MusicTrack, Overlay, Pacing
from .geometry import ClipGeometry, ClipPlacement, Detection, GeometryMap, SceneGeometry
from .manifest import ClipInfo, ClipsManifest, SceneCut
from .media import Credit, Credits, MediaResult, MediaSearch
from .publish import SourceMeta, Thumbnail, UploadPackage
from .render import RenderQC
from .review import GraphicCheck, Review, ReviewTile
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
    "media_search": MediaSearch,
    "credits": Credits,
    "upload": UploadPackage,
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
    "Credit",
    "Credits",
    "Detection",
    "EDLClip",
    "GeometryMap",
    "Graphic",
    "MediaResult",
    "MixSettings",
    "MusicTrack",
    "MediaSearch",
    "Overlay",
    "Pacing",
    "PlannedSegment",
    "RenderQC",
    "GraphicCheck",
    "Review",
    "ReviewTile",
    "SceneCut",
    "SceneGeometry",
    "Segment",
    "Sheet",
    "SourceMeta",
    "SlotChoice",
    "SheetIndex",
    "Strict",
    "Thumbnail",
    "Tile",
    "Transcript",
    "UploadPackage",
    "Violation",
    "Word",
]
