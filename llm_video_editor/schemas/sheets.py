"""sheets/index.json: maps every contact-sheet tile back to a source timestamp."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import Checkpoint, Strict


class Tile(Strict):
    index: int
    t: float  # seconds in the source clip


class Sheet(Strict):
    path: str  # relative to the workspace root
    clip_id: str
    kind: Literal["overview", "dense"] = "overview"
    fps: float
    cols: int
    rows: int
    tile_width: int
    tile_height: int
    start: float
    end: float
    tiles: list[Tile] = Field(default_factory=list)
    est_tokens: int = 0


class SheetIndex(Checkpoint):
    sheets: list[Sheet] = Field(default_factory=list)

    @property
    def est_tokens(self) -> int:
        return sum(s.est_tokens for s in self.sheets)
