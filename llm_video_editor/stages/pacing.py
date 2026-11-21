"""Pacing: jump-cut tightening and alternating punch-ins (run inside cut-audit).

``EDL.pacing.tighten`` splits every A-roll clip into speech islands, removing
pauses longer than ``max_gap`` and keeping ``lead``/``tail`` of air around the
words. ``punch_in: alternate`` zooms every second island of a clip so the jump
cut reads as a deliberate camera change instead of a glitch.

Runs after cut-audit's editorial fixes (word/sentence snapping, budget), so the
split points always sit on word edges inside an already audited clip. Island
pieces keep the clip's priority; the first keeps its id and chapter, the rest
get ``<id>.2``, ``<id>.3``..., and overlays are re-anchored to the piece that
now plays their source moment.
"""

from __future__ import annotations

import re
from typing import TypeVar

from ..schemas import EDL, EDLClip, Graphic, MusicTrack, Overlay, Pacing, Word

Anchored = TypeVar("Anchored", Overlay, Graphic, MusicTrack)

FILLERS = {"um", "uh", "erm", "uhm", "umm", "hmm", "mm", "ah", "er"}


def is_filler(word: Word) -> bool:
    return re.sub(r"[^a-z]", "", word.text.lower()) in FILLERS


def speech_islands(start: float, end: float, words: list[Word], p: Pacing
                   ) -> list[tuple[float, float]]:
    """Split [start, end] at pauses longer than ``p.max_gap``.

    A span with no words (demo, room tone, music) is kept whole.
    """
    inside = [w for w in words if w.start < end and w.end > start]
    if p.drop_fillers:
        inside = [w for w in inside if not is_filler(w)]
    if not inside:
        return [(start, end)]
    islands: list[list[float]] = []
    for w in inside:
        if islands and w.start - islands[-1][1] <= p.max_gap:
            islands[-1][1] = max(islands[-1][1], w.end)
        else:
            islands.append([w.start, w.end])
    out: list[tuple[float, float]] = []
    for s, e in islands:
        if e - s < p.min_island:
            continue
        s, e = max(start, s - p.lead), min(end, e + p.tail)
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out or [(start, end)]


def apply_pacing(edl: EDL, clips: list[EDLClip], words_for: dict[str, list[Word]],
                 overlays: list[Overlay], graphics: list[Graphic] | None = None,
                 music: list[MusicTrack] | None = None
                 ) -> tuple[list[EDLClip], list[Overlay], list[Graphic], list[MusicTrack], float]:
    """Return (clips, overlays, graphics, music - all re-anchored - and seconds removed)."""
    p = edl.pacing
    graphics, music = list(graphics or []), list(music or [])
    if not p.tighten and p.punch_in == "off":
        return clips, overlays, graphics, music, 0.0
    out: list[EDLClip] = []
    pieces_of: dict[str, list[EDLClip]] = {}
    removed = 0.0
    for clip in clips:
        words = words_for.get(clip.source_clip_id, []) if clip.role == "a_roll" else []
        spans = speech_islands(clip.start, clip.end, words, p) if p.tighten else [
            (clip.start, clip.end)]
        removed += clip.duration - sum(e - s for s, e in spans)
        pieces = []
        for k, (s, e) in enumerate(spans):
            update = {"start": round(s, 3), "end": round(e, 3)}
            if k:
                update |= {"clip_id": f"{clip.clip_id}.{k + 1}", "chapter": None}
            # only zoom clips the editor left at 1.0, and only A-roll (B-roll framing is fixed)
            if p.punch_in == "alternate" and k % 2 == 1 and clip.zoom == 1.0 \
                    and clip.role == "a_roll":
                update["zoom"] = p.punch_zoom
            pieces.append(clip.model_copy(update=update))
        pieces_of[clip.clip_id] = pieces
        out += pieces

    starts = {c.clip_id: c.start for c in clips}

    def reanchor(item: Anchored) -> Anchored:
        pieces = pieces_of.get(item.anchor_clip_id)
        if not pieces:
            return item  # anchor missing: cut-audit's bounds check drops it
        src_t = starts[item.anchor_clip_id] + item.offset
        target = next((pc for pc in pieces if src_t < pc.end), pieces[-1])
        return item.model_copy(update={"anchor_clip_id": target.clip_id,
                                       "offset": round(max(0.0, src_t - target.start), 3)})

    return (out, [reanchor(o) for o in overlays], [reanchor(g) for g in graphics],
            [reanchor(m) for m in music], round(removed, 3))
