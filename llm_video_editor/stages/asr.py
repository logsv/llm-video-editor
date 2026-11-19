"""Speech-to-text backends. The default is faster-whisper with word timestamps."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..ffmpeg.runner import ffmpeg
from ..schemas import Segment, Word

DEFAULT_ASR_MODEL = "distil-large-v3"  # English-only; use large-v3-turbo for other languages


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio_path: Path, language: str | None) -> tuple[str | None, list[Segment]]:
        ...


def extract_audio(source: str | Path, out: Path, timeout: float = 1800) -> Path:
    """16 kHz mono PCM, what Whisper models expect."""
    ffmpeg("-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out),
           timeout=timeout)
    return out


class FasterWhisperTranscriber:
    def __init__(self, model: str = DEFAULT_ASR_MODEL, device: str = "auto",
                 compute_type: str = "auto"):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self._model = None

    @property
    def name(self) -> str:
        return f"faster-whisper:{self.model_name}"

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            device = self.device
            if device == "auto":
                try:
                    import ctranslate2

                    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
                except Exception:
                    device = "cpu"
            compute = self.compute_type
            if compute == "auto":
                compute = "float16" if device == "cuda" else "int8"
            self._model = WhisperModel(self.model_name, device=device, compute_type=compute)
        return self._model

    def transcribe(self, audio_path: Path, language: str | None) -> tuple[str | None, list[Segment]]:
        model = self._load()
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        out: list[Segment] = []
        for i, s in enumerate(segments):
            words = [
                Word(start=round(w.start, 3), end=round(w.end, 3), text=w.word.strip(),
                     prob=round(getattr(w, "probability", 1.0), 3))
                for w in (s.words or [])
                if w.word.strip()
            ]
            out.append(Segment(id=i, start=round(s.start, 3), end=round(s.end, 3),
                               text=s.text.strip(), words=words))
        return info.language, out
