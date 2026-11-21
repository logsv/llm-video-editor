"""Chunked, optionally parallel transcription for long clips.

The 16 kHz mono WAV from :func:`~.asr.extract_audio` is split into chunks of about
``chunk_seconds`` with ``overlap`` seconds shared between neighbours. Cut points snap to
silences found by ffmpeg ``silencedetect`` where one is close to the nominal boundary.

Each chunk ``i`` owns the half-open *interior* ``[cut_i, cut_{i+1})``; the overlap around
every cut is only context for the model. Results are merged by timestamps, never by text:
a word belongs to the chunk whose interior contains its midpoint, with two guards at each
cut for timestamp jitter between the two transcriptions of the overlap:

* a word that overlaps a recently kept word by more than half of the shorter of the
  two is the same utterance seen twice and is dropped (no duplicates);
* a word whose midpoint falls just before the cut (within ``boundary_tolerance``) and that
  does not collide with a kept word is kept: the left chunk placed its copy on the other
  side of the cut, so neither chunk would claim it otherwise (no drops).

This tolerates the two transcriptions of a cut disagreeing by up to about half a word's
duration; cutting inside silences keeps the cut away from words in the first place.

Segment ids are re-numbered ``0..n-1`` per clip, so ``transcript.json`` is unchanged.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import re
import shutil
import wave
from collections import Counter, deque
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import astuple, dataclass
from pathlib import Path

from ..ffmpeg.runner import FFmpegError, ffmpeg
from ..schemas import Segment, Word
from .asr import Transcriber


@dataclass(frozen=True)
class ChunkConfig:
    """Settings for chunked ASR. Clips shorter than ``min_duration`` are not chunked."""

    min_duration: float = 600.0
    chunk_seconds: float = 300.0
    overlap: float = 2.0
    search_window: float = 30.0  # look this far either side of a nominal cut for silence
    silence_noise_db: float = -35.0
    silence_min: float = 0.3
    boundary_tolerance: float = 0.25
    workers: int = 0  # 0 = auto: 1 on GPU, a few on CPU. Does not affect the output.

    def applies(self, duration: float) -> bool:
        return duration >= self.min_duration

    def cache_key(self) -> tuple:
        """Everything that can change the transcript (``workers`` cannot)."""
        return ("chunked-v1", *astuple(self)[:-1])


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float  # audio extent sent to the model (includes overlap)
    end: float
    lo: float  # interior [lo, hi) owned by this chunk
    hi: float


# --------------------------------------------------------------------------- silence / plan

_SIL_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SIL_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def parse_silences(log: str, duration: float) -> list[tuple[float, float]]:
    """Parse ``silencedetect`` stderr into ``(start, end)`` intervals."""
    out: list[tuple[float, float]] = []
    start: float | None = None
    for line in log.splitlines():
        if m := _SIL_START.search(line):
            start = max(0.0, float(m.group(1)))
        elif (m := _SIL_END.search(line)) and start is not None:
            out.append((start, min(duration, float(m.group(1)))))
            start = None
    if start is not None and start < duration:
        out.append((start, duration))  # silence running to the end of the file
    return out


def detect_silences(wav: Path, duration: float, cfg: ChunkConfig) -> list[tuple[float, float]]:
    proc = ffmpeg(
        "-i", str(wav),
        "-af", f"silencedetect=noise={cfg.silence_noise_db}dB:d={cfg.silence_min}",
        "-f", "null", "-",
        timeout=max(120.0, duration / 4),
    )
    return parse_silences(proc.stderr, duration)


def plan_chunks(duration: float, silences: Sequence[tuple[float, float]],
                cfg: ChunkConfig) -> list[Chunk]:
    """Choose cut points ~``chunk_seconds`` apart, snapped to the best nearby silence."""
    cuts = [0.0]
    min_gap = max(cfg.overlap, 1.0)
    # The tail may run up to chunk_seconds + search_window rather than leave a tiny chunk.
    while duration - cuts[-1] > cfg.chunk_seconds + cfg.search_window:
        target = cuts[-1] + cfg.chunk_seconds
        best: tuple[tuple[float, float], float] | None = None
        for a, b in silences:
            mid = (a + b) / 2
            if abs(mid - target) > cfg.search_window or mid <= cuts[-1] + min_gap:
                continue
            if mid >= duration - min_gap:
                continue
            # Prefer longer pauses (capped, so a long pause doesn't pull the cut far away),
            # then the one nearest the nominal boundary.
            score = (min(b - a, 1.0), -abs(mid - target))
            if best is None or score > best[0]:
                best = (score, mid)
        cuts.append(round(best[1] if best else target, 3))
    cuts.append(duration)
    half = cfg.overlap / 2
    return [
        Chunk(i, max(0.0, lo - half), min(duration, hi + half), lo, hi)
        for i, (lo, hi) in enumerate(zip(cuts, cuts[1:], strict=False))
    ]


# --------------------------------------------------------------------------- merge


def _mid(w: Word) -> float:
    return (w.start + w.end) / 2


def _collides(a: Word, b: Word) -> bool:
    inter = min(a.end, b.end) - max(a.start, b.start)
    shorter = max(min(a.end - a.start, b.end - b.start), 1e-3)
    return inter > 0.5 * shorter


def merge_chunks(results: Sequence[tuple[Chunk, Sequence[Segment]]],
                 boundary_tolerance: float = 0.25) -> list[Segment]:
    """Merge per-chunk segments (already in clip time) into one clip-level list."""
    out: list[Segment] = []
    recent: deque[Word] = deque(maxlen=4)  # last kept words (or wordless segments)
    ordered = sorted(results, key=lambda r: r[0].index)
    for n, (chunk, segs) in enumerate(ordered):
        tol = min(boundary_tolerance, chunk.lo - chunk.start)
        is_last = n == len(ordered) - 1
        for seg in sorted(segs, key=lambda s: s.start):
            units = list(seg.words) or [Word(start=seg.start, end=seg.end, text=seg.text)]
            kept: list[Word] = []
            for w in sorted(units, key=lambda u: u.start):
                m = _mid(w)
                if m >= chunk.hi and not is_last:
                    continue  # the next chunk owns it
                if m < chunk.lo - tol:
                    continue  # the previous chunk owns it
                if any(_collides(w, r) for r in recent):
                    continue  # same word transcribed by both chunks
                prev = recent[-1] if recent else None
                if prev is not None and w.start < prev.end:  # keep times monotonic
                    w = w.model_copy(update={"start": prev.end, "end": max(w.end, prev.end)})
                kept.append(w)
                recent.append(w)
            if not kept:
                continue
            last_end = out[-1].end if out else -math.inf
            if not seg.words:
                start, end, text, words = kept[0].start, kept[0].end, seg.text, []
            elif len(kept) == len(units):
                start = min(max(seg.start, last_end), kept[0].start)
                end, text, words = max(seg.end, kept[-1].end), seg.text, kept
            else:
                start, end = kept[0].start, kept[-1].end
                text, words = " ".join(w.text for w in kept), kept
            out.append(Segment(id=len(out), start=round(start, 3), end=round(end, 3),
                               text=text, words=words))
    return out


def offset_segments(segments: Sequence[Segment], dt: float) -> list[Segment]:
    def sh(t: float) -> float:
        return round(t + dt, 3)

    return [
        s.model_copy(update={
            "start": sh(s.start), "end": sh(s.end),
            "words": [w.model_copy(update={"start": sh(w.start), "end": sh(w.end)})
                      for w in s.words],
        })
        for s in segments
    ]


# --------------------------------------------------------------------------- execution

CHUNK_NAME = "chunk_{index:03d}_{start_ms:09d}.wav"  # start offset in ms, for debugging


def wav_duration(wav: Path) -> float:
    with wave.open(str(wav), "rb") as r:
        return r.getnframes() / r.getframerate()


def write_slices(wav: Path, chunks: Sequence[Chunk], out_dir: Path) -> list[tuple[Chunk, Path]]:
    """Cut sample-exact WAV slices; chunk starts are snapped to the sample grid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = []
    with wave.open(str(wav), "rb") as r:
        rate, total = r.getframerate(), r.getnframes()
        for c in chunks:
            f0 = min(total, round(c.start * rate))
            f1 = min(total, round(c.end * rate))
            r.setpos(f0)
            frames = r.readframes(f1 - f0)
            snapped = Chunk(c.index, f0 / rate, f1 / rate, c.lo, c.hi)
            path = out_dir / CHUNK_NAME.format(index=c.index, start_ms=round(f0 * 1000 / rate))
            with wave.open(str(path), "wb") as w:
                w.setparams(r.getparams())
                w.writeframes(frames)
            result.append((snapped, path))
    return result


def default_workers(transcriber: Transcriber) -> int:
    uses_gpu = getattr(transcriber, "uses_gpu", None)
    if callable(uses_gpu) and uses_gpu():
        return 1
    return max(1, min(4, (os.cpu_count() or 1) // 4))


_WORKER_ASR: Transcriber | None = None


def _worker_init(transcriber: Transcriber) -> None:
    global _WORKER_ASR
    _WORKER_ASR = transcriber  # unpickled copy: each worker loads its own model lazily


def _worker_transcribe(path: str, language: str | None) -> tuple[str | None, list[Segment]]:
    assert _WORKER_ASR is not None
    return _WORKER_ASR.transcribe(Path(path), language)


def transcribe_chunked(transcriber: Transcriber, wav: Path, language: str | None,
                       cfg: ChunkConfig, work_dir: Path) -> tuple[str | None, list[Segment]]:
    duration = wav_duration(wav)
    try:
        silences = detect_silences(wav, duration, cfg)
    except FFmpegError:
        silences = []  # fall back to fixed-length cuts
    chunks = plan_chunks(duration, silences, cfg)
    if len(chunks) == 1:
        return transcriber.transcribe(wav, language)

    workers = min(cfg.workers or default_workers(transcriber), len(chunks))
    try:
        slices = write_slices(wav, chunks, work_dir)
        raw: dict[int, tuple[str | None, list[Segment]]] = {}
        if workers <= 1:
            run = transcriber.transcribe
            for c, p in slices:
                raw[c.index] = run(p, language)
                language = language or raw[c.index][0]
        else:
            # spawn: safe with CUDA and with the threads `watch` is running concurrently.
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                     initializer=_worker_init,
                                     initargs=(transcriber,)) as pool:
                rest = slices
                if language is None:
                    # Detect on the first chunk and force it on the rest, so one clip
                    # doesn't end up transcribed in two languages.
                    (c0, p0), rest = slices[0], slices[1:]
                    raw[c0.index] = pool.submit(_worker_transcribe, str(p0), None).result()
                    language = raw[c0.index][0]
                futs = {c.index: pool.submit(_worker_transcribe, str(p), language)
                        for c, p in rest}
                raw.update({i: f.result() for i, f in futs.items()})
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    langs = Counter(lang for lang, _ in raw.values() if lang)
    merged = merge_chunks(
        [(c, offset_segments(raw[c.index][1], c.start)) for c, _ in slices],
        cfg.boundary_tolerance,
    )
    return (langs.most_common(1)[0][0] if langs else None), merged


__all__ = [
    "Chunk",
    "ChunkConfig",
    "detect_silences",
    "merge_chunks",
    "offset_segments",
    "parse_silences",
    "plan_chunks",
    "transcribe_chunked",
]
