"""Pydantic v2 schemas for every JSON checkpoint the pipeline writes."""

from __future__ import annotations

from .asset_plan import AssetPlan, BrollPlacement, PlannedSegment
from .base import SCHEMA_VERSION, Checkpoint, Strict
from .cut_audit import CutAudit, Violation
from .edl import EDL, CaptionSettings, EDLClip, Overlay
from .manifest import ClipInfo, ClipsManifest, SceneCut
from .render import RenderQC
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
}

__all__ = [
    "CHECKPOINTS",
    "EDL",
    "SCHEMA_VERSION",
    "AssetPlan",
    "BrollPlacement",
    "CaptionSettings",
    "Checkpoint",
    "ClipInfo",
    "ClipTranscript",
    "ClipsManifest",
    "CutAudit",
    "EDLClip",
    "Overlay",
    "PlannedSegment",
    "RenderQC",
    "SceneCut",
    "Segment",
    "Sheet",
    "SheetIndex",
    "Strict",
    "Tile",
    "Transcript",
    "Violation",
    "Word",
]
