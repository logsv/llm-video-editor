"""Synthesize narration lines into the workspace, cached by text and voice.

Each line becomes ``<ws>/voice/<hash>.wav``: 48 kHz mono, leading/trailing silence
trimmed, 15 ms fades so cuts never click, and a high-pass that removes rumble. The
level is set at render time (levelled to the measured speech like every track), so
lines are left unnormalised here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..ffmpeg.probe import media_duration
from ..ffmpeg.runner import ffmpeg
from .engines import Engine
from .text import load_lexicon, spoken

CLEAN = ("highpass=f=70,"
         "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.03,"
         "areverse,silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.06,"
         "areverse,afade=t=in:d=0.015,areverse,afade=t=in:d=0.015,areverse,"
         "aresample=48000")


@dataclass(frozen=True)
class Line:
    text: str  # as written (captions)
    spoken: str  # what the engine read
    voice: str
    engine: str
    speed: float
    file: str
    duration: float
    cached: bool

    def as_dict(self) -> dict:
        return asdict(self)


def voice_dir(ws_root: Path) -> Path:
    d = Path(ws_root) / "voice"
    d.mkdir(parents=True, exist_ok=True)
    return d


def lexicon_paths(ws_root: Path) -> list[Path]:
    from ..agent.setup import config_dir

    return [config_dir() / "lexicon.json", voice_dir(ws_root) / "lexicon.json"]


def synthesize(ws_root: Path, engine: Engine, text: str, voice: str, *, speed: float = 1.0,
               lang: str | None = None) -> Line:
    """One clean line of narration (reused when text, voice and settings repeat)."""
    text = text.strip()
    if not text:
        raise ValueError("empty narration line")
    code = lang or next((v.lang_code for v in engine.voices() if v.voice == voice), "en-us")
    say = spoken(text, code, load_lexicon(*lexicon_paths(ws_root)))
    key = hashlib.sha1(json.dumps([engine.model_id(), voice, round(speed, 3), code, say])
                       .encode()).hexdigest()[:16]
    out = voice_dir(ws_root) / f"vo-{key}.wav"
    cached = out.exists()
    if not cached:
        raw = out.with_name(f".{out.stem}.raw.wav")
        engine.synth(say, voice, raw, speed=speed, lang=code)
        ffmpeg("-i", str(raw), "-af", CLEAN, "-ac", "1", "-c:a", "pcm_s16le", str(out),
               timeout=120)
        raw.unlink(missing_ok=True)
    return Line(text=text, spoken=say, voice=voice, engine=engine.name, speed=speed,
                file=str(out), duration=round(media_duration(out), 3), cached=cached)
