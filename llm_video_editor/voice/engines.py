"""Local text-to-speech engines.

* ``kokoro``: Kokoro-82M through ``kokoro-onnx`` (``uv sync --extra voice``); Apache-2.0
  model, natural voices in American/British English, Hindi and more. The model files
  download once from the kokoro-onnx GitHub release into ``~/.cache/llmve/models/kokoro``
  (``$LLMVE_CACHE`` moves it).
* ``say``: macOS system voices, nothing to install; flat and robotic, a fallback.

Every engine writes a WAV file for one line of text. Cloud engines (and copies of the
user's own voice) are a later phase and need the user's explicit setting.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_MODELS = ("kokoro-v1.0.onnx", "kokoro-v1.0.fp16.onnx", "kokoro-v1.0.int8.onnx")
KOKORO_VOICES_FILE = "voices-v1.0.bin"
# voice id prefix -> (language name, espeak language code)
KOKORO_LANGS = {"a": ("English (US)", "en-us"), "b": ("English (UK)", "en-gb"),
                "h": ("Hindi", "hi"), "e": ("Spanish", "es"), "f": ("French", "fr-fr"),
                "i": ("Italian", "it"), "p": ("Portuguese (BR)", "pt-br"),
                "j": ("Japanese", "ja"), "z": ("Chinese", "cmn")}


class EngineUnavailable(RuntimeError):
    """The engine is not installed here (the message says how to get it)."""


@dataclass(frozen=True)
class VoiceInfo:
    engine: str
    voice: str
    language: str
    lang_code: str
    gender: str  # "female" | "male" | "?"
    licence: str
    local: bool = True


class Engine(Protocol):
    name: str

    def voices(self) -> list[VoiceInfo]: ...

    def synth(self, text: str, voice: str, out_wav: Path, *, speed: float = 1.0,
              lang: str | None = None) -> Path: ...

    def model_id(self) -> str: ...


def cache_dir() -> Path:
    d = Path(os.environ.get("LLMVE_CACHE") or Path.home() / ".cache" / "llmve") / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


class KokoroEngine:
    name = "kokoro"
    licence = "Apache-2.0 (Kokoro-82M)"

    def __init__(self, model: str | None = None):
        try:
            import kokoro_onnx  # noqa: F401
        except ImportError as e:
            raise EngineUnavailable("Kokoro needs `uv sync --extra voice`") from e
        self.dir = cache_dir() / "kokoro"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model or os.environ.get("LLMVE_KOKORO_MODEL") or next(
            (m for m in KOKORO_MODELS if (self.dir / m).exists()), KOKORO_MODELS[0])
        self._k = None

    def _download(self, name: str) -> Path:
        import requests

        out = self.dir / name
        if out.exists() and out.stat().st_size > 0:
            return out
        tmp = out.with_suffix(out.suffix + ".part")
        try:
            with requests.get(f"{KOKORO_RELEASE}/{name}", stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        except requests.RequestException as e:
            tmp.unlink(missing_ok=True)
            raise EngineUnavailable(f"could not download the Kokoro model file {name} "
                                    f"({e}); put it in {self.dir} by hand") from e
        tmp.replace(out)
        return out

    def _kokoro(self):
        if self._k is None:
            from kokoro_onnx import Kokoro

            self._k = Kokoro(str(self._download(self.model_name)),
                             str(self._download(KOKORO_VOICES_FILE)))
        return self._k

    def model_id(self) -> str:
        return f"kokoro/{self.model_name}"

    def voices(self) -> list[VoiceInfo]:
        out = []
        for v in sorted(self._kokoro().get_voices()):
            lang, code = KOKORO_LANGS.get(v[0], ("?", "en-us"))
            gender = {"f": "female", "m": "male"}.get(v[1:2], "?")
            out.append(VoiceInfo("kokoro", v, lang, code, gender, self.licence))
        return out

    def synth(self, text: str, voice: str, out_wav: Path, *, speed: float = 1.0,
              lang: str | None = None) -> Path:
        import numpy as np

        k = self._kokoro()
        if voice not in k.get_voices():
            raise ValueError(f"no Kokoro voice {voice!r}; see the voices tool")
        code = lang or KOKORO_LANGS.get(voice[0], ("", "en-us"))[1]
        samples, rate = k.create(text, voice=voice, speed=speed, lang=code)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        _write_wav(out_wav, pcm, rate)
        return out_wav


class SayEngine:
    """macOS ``say``: system voices, robotic; a fallback when Kokoro is unavailable."""

    name = "say"
    licence = "macOS system voice (check Apple's terms before commercial use)"

    def __init__(self):
        if platform.system() != "Darwin" or not shutil.which("say"):
            raise EngineUnavailable("`say` exists only on macOS")

    def model_id(self) -> str:
        return "macos-say"

    def voices(self) -> list[VoiceInfo]:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True,
                             timeout=30).stdout
        vs = []
        for line in out.splitlines():
            m = re.match(r"^(.+?)\s+([a-z]{2}[_-][A-Z]{2})\s+#", line)
            if m:
                vs.append(VoiceInfo("say", m.group(1).strip(), m.group(2), m.group(2), "?",
                                    self.licence))
        return vs

    def synth(self, text: str, voice: str, out_wav: Path, *, speed: float = 1.0,
              lang: str | None = None) -> Path:
        from ..ffmpeg.runner import ffmpeg

        aiff = out_wav.with_suffix(".aiff")
        subprocess.run(["say", "-v", voice, "-r", str(round(175 * speed)), "-o", str(aiff),
                        text], check=True, capture_output=True, timeout=120)
        ffmpeg("-i", str(aiff), "-ar", "48000", "-ac", "1", str(out_wav), timeout=120)
        aiff.unlink(missing_ok=True)
        return out_wav


def _write_wav(path: Path, pcm: bytes, rate: int) -> None:
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


ENGINES = {"kokoro": KokoroEngine, "say": SayEngine}
_LOADED: dict[str, Engine] = {}


def get_engine(name: str = "auto") -> Engine:
    """An engine by name; ``auto`` is Kokoro, then macOS ``say``."""
    order = list(ENGINES) if name == "auto" else [name]
    errors = []
    for n in order:
        if n not in ENGINES:
            raise ValueError(f"unknown engine {n!r}: {', '.join(ENGINES)} or auto")
        if n in _LOADED:
            return _LOADED[n]
        try:
            _LOADED[n] = ENGINES[n]()
            return _LOADED[n]
        except EngineUnavailable as e:
            errors.append(f"{n}: {e}")
    raise EngineUnavailable("no voice engine available: " + "; ".join(errors))


def list_voices(engine: str = "auto", language: str | None = None) -> list[VoiceInfo]:
    eng = get_engine(engine)
    vs = eng.voices()
    if language:
        q = language.lower()
        vs = [v for v in vs if q in v.language.lower() or v.lang_code.lower().startswith(q)]
    return vs
