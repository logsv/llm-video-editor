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


class FakeProvider:
    """Planner provider double: ``responder(schema, prompt) -> model``."""

    name = "fake-llm"

    def __init__(self, responder):
        self.responder = responder
        self.prompts: list[tuple[str, str, type]] = []

    def complete(self, system, prompt, schema):
        self.prompts.append((system, prompt, schema))
        return self.responder(schema, prompt)


class FakeDetector:
    """Returns the same face box for every frame (normalised source coords)."""

    name = "fake-face"

    def __init__(self, box=(0.05, 0.55, 0.30, 0.95), label="face"):
        self.box, self.label, self.frames = box, label, 0

    def detect(self, rgb):
        self.frames += 1
        assert rgb.ndim == 3 and rgb.shape[2] == 3
        return [(self.label, self.box, 0.9)] if self.box else []

    def close(self):
        pass


class TimelineTranscriber:
    """Chunk-aware ASR double: knows the clip's true word timeline and returns the words
    that lie fully inside the audio it is given, in that file's local time.

    The chunk's offset in the clip is read from the ``chunk_<i>_<start_ms>.wav`` name that
    chunked ASR writes (any other file is treated as the whole clip). Module-level and
    stateless, so it can be pickled into process-pool workers.
    """

    name = "timeline-asr"

    def __init__(self, words: list[Word], per_segment: int = 4):
        self.words, self.per_segment = words, per_segment
        self.calls: list[Path] = []

    def transcribe(self, audio_path, language):
        import re
        import wave

        path = Path(audio_path)
        self.calls.append(path)
        m = re.match(r"chunk_\d+_(\d+)\.wav$", path.name)
        offset = int(m.group(1)) / 1000 if m else 0.0
        with wave.open(str(path), "rb") as r:
            length = r.getnframes() / r.getframerate()
        segs: list[Segment] = []
        group: list[Word] = []
        last_key = None
        for i, w in enumerate(self.words):
            if not (w.start >= offset and w.end <= offset + length):
                continue
            key = i // self.per_segment
            if group and key != last_key:
                segs.append(_segment(len(segs), group))
                group = []
            last_key = key
            group.append(Word(start=round(w.start - offset, 3), end=round(w.end - offset, 3),
                              text=w.text))
        if group:
            segs.append(_segment(len(segs), group))
        return (language or "en", segs)


def _segment(i: int, words: list[Word]) -> Segment:
    return Segment(id=i, start=words[0].start, end=words[-1].end,
                   text=" ".join(w.text for w in words), words=words)
