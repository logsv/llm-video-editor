"""Media probing (duration, geometry, frame rate, audio) via ffprobe."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .runner import ffprobe_json


@dataclass(frozen=True)
class ProbeResult:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    video_codec: str | None
    audio_codec: str | None
    sample_rate: int | None
    channels: int | None
    rotation: int = 0


def _rate(value: str | None) -> float:
    if not value or value in {"0/0", "0"}:
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(round(float(sd["rotation"]))) % 360
    tag = stream.get("tags", {}).get("rotate")
    return int(tag) % 360 if tag else 0


def probe(path: str | Path) -> ProbeResult:
    data = ffprobe_json(path)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError(f"No video stream in {path}")

    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0.0)
    rotation = _rotation(video)
    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    if rotation in (90, 270):  # ffmpeg auto-rotates on decode, so report display size
        width, height = height, width
    fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    return ProbeResult(
        path=str(Path(path).resolve()),
        duration=duration,
        width=width,
        height=height,
        fps=round(fps, 3),
        has_audio=audio is not None,
        video_codec=video.get("codec_name"),
        audio_codec=audio.get("codec_name") if audio else None,
        sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        channels=int(audio["channels"]) if audio and audio.get("channels") else None,
        rotation=rotation,
    )


def media_duration(path: str | Path) -> float:
    """Duration of any media file, audio-only included (``probe`` needs a video stream)."""
    from .runner import ffprobe_json

    fmt = ffprobe_json(Path(path)).get("format", {})
    return float(fmt.get("duration") or 0.0)
