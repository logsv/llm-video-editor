"""Two-pass EBU R128 loudness normalisation (default -14 LUFS, -1.5 dBTP, LRA 11)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .runner import ffmpeg


@dataclass(frozen=True)
class LoudnessTarget:
    integrated: float = -14.0
    true_peak: float = -1.5
    lra: float = 11.0

    def base(self) -> str:
        return f"loudnorm=I={self.integrated}:TP={self.true_peak}:LRA={self.lra}"


DEFAULT_TARGET = LoudnessTarget()
_JSON_BLOCK = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.S)


def parse_loudnorm_json(stderr: str) -> dict[str, float]:
    """Return the measurements printed by ``loudnorm=print_format=json``."""
    blocks = _JSON_BLOCK.findall(stderr)
    if not blocks:
        raise ValueError("no loudnorm measurement block in ffmpeg output")
    raw = json.loads(blocks[-1])
    out: dict[str, float] = {}
    for k, v in raw.items():
        if k == "normalization_type":
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = -math.inf
    return out


def measure_filter(target: LoudnessTarget = DEFAULT_TARGET) -> str:
    return f"{target.base()}:print_format=json"


def is_silent(m: dict[str, float]) -> bool:
    return not math.isfinite(m.get("input_i", -math.inf)) or m["input_i"] < -70


def apply_filter(m: dict[str, float], target: LoudnessTarget = DEFAULT_TARGET) -> str:
    """Second-pass filter using first-pass measurements (linear when possible)."""
    if is_silent(m):
        return "anull"
    return (
        f"{target.base()}:measured_I={m['input_i']:.2f}:measured_TP={m['input_tp']:.2f}"
        f":measured_LRA={m['input_lra']:.2f}:measured_thresh={m['input_thresh']:.2f}"
        f":offset={m.get('target_offset', 0.0):.2f}:linear=true:print_format=summary"
    )


def measure_file(path: str | Path, target: LoudnessTarget = DEFAULT_TARGET,
                 timeout: float = 1800) -> dict[str, float]:
    proc = ffmpeg("-i", str(path), "-vn", "-af", measure_filter(target), "-f", "null", "-",
                  timeout=timeout)
    return parse_loudnorm_json(proc.stderr)
