"""cut_audit.json: deterministic checks on the rough cut, with the fixes applied."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Checkpoint, Strict

ViolationKind = Literal[
    "unknown_source",
    "out_of_bounds",
    "mid_word_in",
    "mid_word_out",
    "mid_sentence_out",
    "overlap",
    "too_short",
    "duration_budget",
    "overlay_out_of_bounds",
]


class Violation(Strict):
    kind: ViolationKind
    severity: Literal["error", "warning"] = "error"
    clip_id: str | None = None
    source_file: str | None = None
    at: float | None = None
    message: str
    fixed: bool = False
    fix: str | None = None


class CutAudit(Checkpoint):
    passed: bool
    violations: list[Violation] = Field(default_factory=list)
    clips_before: int = 0
    clips_after: int = 0
    duration_before: float = 0.0
    duration_after: float = 0.0
    target_duration: float | None = None
    tolerance: float = 0.05

    @property
    def unresolved(self) -> list[Violation]:
        return [v for v in self.violations if not v.fixed and v.severity == "error"]
