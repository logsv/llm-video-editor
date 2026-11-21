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


# -- automatic chapters (llmve auto) ----------------------------------------------------
def suggest_chapters(edl, transcript, provider, *, min_total: float = 120.0) -> dict[str, str]:
    """Ask the planner to name chapters by clip id; return {clip_id: title} that meets
    YouTube's rules (first at 0:00, >= 3, each >= 10 s), or {} when it can't."""
    from pydantic import Field

    from ..llm import structured
    from ..schemas import Strict

    class _Chapter(Strict):
        clip_id: str = Field(description="clip that starts the chapter")
        title: str = Field(description="2-6 words, specific, no numbering")

    class _Chapters(Strict):
        chapters: list[_Chapter]

    starts = edl.output_starts()
    total = edl.total_duration
    if total < min_total or len(edl.clips) < 3:
        return {}
    lines = []
    for clip, t0 in zip(edl.clips, starts, strict=True):
        ct = transcript.for_clip(clip.source_clip_id)
        text = ""
        if ct is not None:
            seg = next((s for s in ct.segments if s.end > clip.start and s.start < clip.end), None)
            text = seg.text if seg else ""
        lines.append(f"{clip.clip_id} @ {stamp(t0)} ({clip.duration:.0f}s): {text[:160]}")
    res = structured(provider, (
        "You split a video into YouTube chapters. Start a chapter where the topic changes. "
        "Titles are short, specific and scannable. Answer strictly in the requested JSON schema."),
        f"VIDEO: {stamp(total)}. Pick 3-{max(3, min(12, int(total // 60)))} chapter starts "
        "by clip id (the first chapter starts at the first clip).\n\n" + "\n".join(lines),
        _Chapters)

    order = {c.clip_id: i for i, c in enumerate(edl.clips)}
    picked: dict[int, str] = {}
    for ch in res.chapters:
        title = re.sub(r"\s+", " ", ch.title.replace("<", "").replace(">", "")).strip()[:60]
        if ch.clip_id in order and title:
            picked.setdefault(order[ch.clip_id], title)
    if not picked:
        return {}
    if 0 not in picked:  # the first chapter must start the video
        first = min(picked)
        picked[0] = picked.pop(first)
    kept: list[int] = []
    for i in sorted(picked):  # drop chapters that would be shorter than 10 s
        if kept and starts[i] - starts[kept[-1]] < MIN_SECONDS:
            continue
        kept.append(i)
    if kept and total - starts[kept[-1]] < MIN_SECONDS:
        kept.pop()
    if len(kept) < MIN_CHAPTERS:
        return {}
    return {edl.clips[i].clip_id: picked[i] for i in kept}
