"""YouTube chapters from ``EDLClip.chapter`` markers on the output timeline.

YouTube only shows chapters when the first starts at 0:00, there are at least
three, and each lasts at least 10 seconds. Violations are warnings (the video
still renders); the first chapter is always moved to 0:00.
"""

from __future__ import annotations

import re

from ..schemas import EDL

MIN_CHAPTERS = 3
MIN_SECONDS = 10.0


def chapter_marks(edl: EDL, starts: list[float]) -> list[tuple[float, str]]:
    marks = [(round(t, 3), c.chapter.strip()) for c, t in zip(edl.clips, starts, strict=True)
             if c.chapter and c.chapter.strip()]
    if marks and marks[0][0] > 0:
        marks[0] = (0.0, marks[0][1])
    return marks


def chapter_warnings(marks: list[tuple[float, str]], total: float) -> list[str]:
    if not marks:
        return []
    out = []
    if len(marks) < MIN_CHAPTERS:
        out.append(f"YouTube needs at least {MIN_CHAPTERS} chapters; the cut has {len(marks)}")
    ends = [t for t, _ in marks[1:]] + [total]
    for (t, title), end in zip(marks, ends, strict=True):
        if end - t < MIN_SECONDS:
            out.append(f"chapter {title!r} is {end - t:.1f}s; YouTube needs {MIN_SECONDS:g}s or more")
    return out


def stamp(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def description_block(marks: list[tuple[float, str]]) -> str:
    """The lines to paste into the YouTube description."""
    return "".join(f"{stamp(t)} {title}\n" for t, title in marks)


def ffmetadata(marks: list[tuple[float, str]], total: float) -> str:
    """Chapter atoms for the MP4 (players and YouTube read them too)."""
    def esc(text: str) -> str:
        return re.sub(r"([=;#\\\n])", r"\\\1", text)

    lines = [";FFMETADATA1"]
    ends = [t for t, _ in marks[1:]] + [total]
    for (t, title), end in zip(marks, ends, strict=True):
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(t * 1000)}",
                  f"END={round(end * 1000)}", f"title={esc(title)}"]
    return "\n".join(lines) + "\n"
