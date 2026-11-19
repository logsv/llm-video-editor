"""Scene cuts and a motion score from ffmpeg's ``scene`` metric (one decode pass)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .runner import ffmpeg

_PTS = re.compile(r"pts_time:([0-9.]+)")
_SCORE = re.compile(r"lavfi\.scene_score=([0-9.]+)")


@dataclass(frozen=True)
class SceneAnalysis:
    cuts: list[float]
    scenes: list[tuple[float, float]]
    motion_score: float  # mean scene score between sampled frames, 0..1
    cuts_per_minute: float


def parse_scene_scores(log: str) -> list[tuple[float, float]]:
    """Parse ``metadata=print`` output into ``(pts_time, scene_score)`` pairs."""
    out: list[tuple[float, float]] = []
    t: float | None = None
    for line in log.splitlines():
        if m := _PTS.search(line):
            t = float(m.group(1))
        elif (m := _SCORE.search(line)) and t is not None:
            out.append((t, float(m.group(1))))
            t = None
    return out


def scenes_from_scores(
    scores: list[tuple[float, float]], duration: float, threshold: float, min_len: float
) -> SceneAnalysis:
    cuts: list[float] = []
    for t, s in scores:
        if s >= threshold and t > min_len and (not cuts or t - cuts[-1] >= min_len):
            cuts.append(round(t, 3))
    if cuts and duration - cuts[-1] < min_len:
        cuts.pop()
    bounds = [0.0, *cuts, round(duration, 3)]
    scenes = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    motion = sum(s for _, s in scores) / len(scores) if scores else 0.0
    cpm = len(cuts) / (duration / 60) if duration > 0 else 0.0
    return SceneAnalysis(cuts, scenes, round(motion, 4), round(cpm, 2))


def analyze_scenes(
    source: str | Path,
    duration: float,
    *,
    sample_fps: float = 4.0,
    threshold: float = 0.3,
    min_len: float = 1.0,
    timeout: float = 1800,
) -> SceneAnalysis:
    proc = ffmpeg(
        "-i", str(source), "-an",
        "-vf", f"fps={sample_fps},scale=160:-2,select='gte(scene,0)',metadata=print",
        "-f", "null", "-",
        timeout=timeout,
    )
    scores = parse_scene_scores(proc.stderr + proc.stdout)
    return scenes_from_scores(scores, duration, threshold, min_len)
