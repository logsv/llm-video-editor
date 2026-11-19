"""Parse human durations: ``10m``, ``90s``, ``1:30``, ``1m30s``, ``2 minutes``."""

from __future__ import annotations

import re

_UNIT = r"(h|hrs?|hours?|m|mins?|minutes?|s|secs?|seconds?)"
_PART = re.compile(rf"(\d+(?:\.\d+)?)\s*{_UNIT}(?![a-z])", re.IGNORECASE)
_CLOCK = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s*$")


def _seconds(value: float, unit: str) -> float:
    u = unit.lower()
    if u.startswith("h"):
        return value * 3600
    if u.startswith("m"):
        return value * 60
    return value


def parse_duration(text: str) -> float:
    """Parse a CLI duration value; raises ValueError if it is not one."""
    text = text.strip()
    if m := _CLOCK.match(text):
        h, mm, ss = m.groups()
        return int(h or 0) * 3600 + int(mm) * 60 + float(ss)
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    parts = list(_PART.finditer(text))
    if not parts or "".join(p.group(0) for p in parts).replace(" ", "") != text.replace(" ", ""):
        raise ValueError(f"Not a duration: {text!r}")
    return sum(_seconds(float(p.group(1)), p.group(2)) for p in parts)


def duration_from_prompt(prompt: str) -> float | None:
    """First explicit duration in free text (``"60s reel: top 3 takeaways"`` -> 60)."""
    m = re.search(rf"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*{_UNIT}(?![a-z])", prompt, re.IGNORECASE)
    return _seconds(float(m.group(1)), m.group(2)) if m else None
