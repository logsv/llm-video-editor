"""qc_report.json: post-render checks (duration, A/V alignment, format, loudness)."""

from __future__ import annotations

from pydantic import Field

from .base import Checkpoint


class RenderQC(Checkpoint):
    output: str
    passed: bool
    encoder: str
    expected_duration: float
    duration: float
    video_duration: float
    audio_duration: float
    av_offset: float  # |video stream duration - audio stream duration|
    width: int
    height: int
    fps: float
    sample_rate: int | None = None
    channels: int | None = None
    integrated_lufs: float | None = None
    true_peak: float | None = None
    target_lufs: float = -14.0
    problems: list[str] = Field(default_factory=list)
