"""Pick an H.264 encoder that actually works on this machine.

``ffmpeg -encoders`` lists NVENC/VideoToolbox whenever ffmpeg was built with
them, even with no GPU present, so availability is proven with a tiny test
encode before an encoder is chosen.
"""

from __future__ import annotations

from functools import cache
from typing import Literal

from .runner import FFmpegError, ffmpeg, run

Encoder = Literal["h264_nvenc", "h264_videotoolbox", "libx264"]
Quality = Literal["high", "draft"]
HARDWARE: tuple[Encoder, ...] = ("h264_nvenc", "h264_videotoolbox")


@cache
def _listed() -> str:
    try:
        return run(["ffmpeg", "-hide_banner", "-encoders"], timeout=30).stdout
    except FFmpegError:
        return ""


@cache
def encoder_works(name: str) -> bool:
    if name not in _listed():
        return False
    try:
        ffmpeg("-f", "lavfi", "-i", "color=c=black:s=256x144:r=30:d=0.2", "-frames:v", "3",
               "-c:v", name, "-f", "null", "-", timeout=30)
        return True
    except FFmpegError:
        return False


def detect_encoder(prefer: str = "auto") -> Encoder:
    if prefer != "auto":
        if prefer not in (*HARDWARE, "libx264"):
            raise ValueError(f"Unsupported encoder {prefer!r}")
        if not encoder_works(prefer):
            raise FFmpegError(f"Encoder {prefer} is not usable on this machine")
        return prefer  # type: ignore[return-value]
    for name in HARDWARE:
        if encoder_works(name):
            return name
    return "libx264"


def video_args(encoder: Encoder, fps: int, quality: Quality = "high") -> list[str]:
    draft = quality == "draft"
    if encoder == "h264_nvenc":
        args = ["-c:v", "h264_nvenc", "-preset", "p1" if draft else "p5", "-tune", "hq",
                "-rc", "vbr", "-cq", "28" if draft else "21", "-b:v", "0", "-profile:v", "high"]
    elif encoder == "h264_videotoolbox":
        args = ["-c:v", "h264_videotoolbox", "-b:v", "4M" if draft else "10M",
                "-profile:v", "high", "-allow_sw", "1"]
    else:
        args = ["-c:v", "libx264", "-preset", "ultrafast" if draft else "medium",
                "-crf", "28" if draft else "20", "-profile:v", "high"]
    return [*args, "-pix_fmt", "yuv420p", "-r", str(fps), "-g", str(fps * 2)]


AUDIO_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
MEZZANINE_VIDEO = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p"]
MEZZANINE_AUDIO = ["-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2"]
