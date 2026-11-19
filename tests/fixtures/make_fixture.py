"""Generate small synthetic test clips with ffmpeg's lavfi sources.

Clips can carry A/V sync markers: a white full-frame flash and a 1 kHz beep
that both start at t = k + 0.5 s and last 0.1 s, so a render can be checked
for drift by comparing flash onsets with beep onsets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from llm_video_editor.ffmpeg.runner import ffmpeg

Audio = Literal["sine", "noise", "beeps"] | None

MARKER_PHASE = 0.5
MARKER_LEN = 0.1


def make_clip(
    path: Path,
    *,
    duration: float = 3.0,
    size: tuple[int, int] = (1920, 1080),
    fps: float = 30,
    audio: Audio = "sine",
    markers: bool = False,
    sample_rate: int = 44100,
) -> Path:
    w, h = size
    args: list[str] = []
    if markers:
        args += ["-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r={fps}:d={duration}"]
    else:
        args += ["-f", "lavfi", "-i", f"testsrc2=s={w}x{h}:r={fps}:d={duration}"]
    if audio == "sine":
        args += ["-f", "lavfi", "-i", f"sine=f=440:sample_rate={sample_rate}:d={duration}"]
    elif audio == "noise":
        args += ["-f", "lavfi", "-i",
                 f"anoisesrc=color=pink:amplitude=0.05:sample_rate={sample_rate}:d={duration}"]
    elif audio == "beeps":
        args += ["-f", "lavfi", "-i", f"sine=f=1000:sample_rate={sample_rate}:d={duration}"]

    win = f"between(mod(t,1),{MARKER_PHASE},{MARKER_PHASE + MARKER_LEN - 0.001})"
    if markers:
        args += ["-vf", f"drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='{win}'"]
    if audio == "beeps":
        args += ["-af", f"volume=volume=0:enable='not({win})'"]
    elif audio == "sine":
        args += ["-af", "volume=-12dB"]

    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-g", str(int(fps))]
    if audio:
        args += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(*args, str(path), timeout=120)
    return path
