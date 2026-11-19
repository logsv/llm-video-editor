"""Test doubles shared across unit and integration tests."""

from __future__ import annotations

from pathlib import Path

from llm_video_editor.schemas import Segment, Word


def words_for(sentence: str, start: float, step: float = 0.3, gap: float = 0.05) -> list[Word]:
    out, t = [], start
    for w in sentence.split():
        out.append(Word(start=round(t, 3), end=round(t + step - gap, 3), text=w))
        t += step
    return out


def make_segments(sentences: list[tuple[float, str]]) -> list[Segment]:
    segs = []
    for i, (start, text) in enumerate(sentences):
        words = words_for(text, start)
        segs.append(Segment(id=i, start=words[0].start, end=words[-1].end, text=text, words=words))
    return segs


DEFAULT_SCRIPT = [
    (0.2, "Hello and welcome to the show."),
    (2.2, "Today we cover three ideas."),
]


class FakeTranscriber:
    name = "fake-asr"

    def __init__(self, script=None, silent=False):
        self.script = script or DEFAULT_SCRIPT
        self.silent = silent
        self.calls: list[Path] = []

    def transcribe(self, audio_path, language):
        self.calls.append(Path(audio_path))
        return ("en", [] if self.silent else make_segments(self.script))
