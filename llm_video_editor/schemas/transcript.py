"""transcript.json: word-level ASR output for every clip in the batch."""

from __future__ import annotations

from pydantic import Field

from .base import Checkpoint, Strict


class Word(Strict):
    start: float
    end: float
    text: str
    prob: float = 1.0


class Segment(Strict):
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class ClipTranscript(Strict):
    clip_id: str
    source_file: str
    language: str | None = None
    duration: float
    segments: list[Segment] = Field(default_factory=list)

    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]


class Transcript(Checkpoint):
    clips: list[ClipTranscript] = Field(default_factory=list)

    def for_clip(self, clip_id: str) -> ClipTranscript | None:
        return next((c for c in self.clips if c.clip_id == clip_id), None)
