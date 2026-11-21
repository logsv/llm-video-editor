"""
Pure EDL logic for long-form edits (no ffmpeg calls here, so it is unit-testable).

edl.json, written by Claude after reading transcript.md:

{
  "title": "How I built X",
  "tighten": {"enabled": true, "max_gap": 0.6, "pad_before": 0.08, "pad_after": 0.18,
              "drop_fillers": true},
  "punch_in": {"mode": "alternate", "zoom": 1.12},
  "segments": [
    {"in": 1834.20, "out": 1851.70, "chapter": "Cold open", "tighten": false},
    {"in": 12.40,   "out": 95.20,  "chapter": "Intro"},
    {"in": 130.00,  "out": 410.50, "zoom": 1.0, "cx": 0.5, "cy": 0.45},
    {"in": 410.50,  "out": 900.00, "chapter": "Step 1: Setup"}
  ]
}

Times are source seconds. Segments play in the order listed (a cold open taken from
later in the source is fine). "chapter" starts a YouTube chapter at that segment.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

FILLERS = {"um", "uh", "erm", "uhm", "umm", "hmm", "mm", "ah", "er"}

DEFAULT_TIGHTEN = {
    "enabled": False,
    "max_gap": 0.6,      # silences longer than this inside a kept segment are removed
    "pad_before": 0.08,  # keep this much air before the first word of an island
    "pad_after": 0.18,   # and this much after the last word
    "min_island": 0.25,  # islands shorter than this (after filler removal) are dropped
    "drop_fillers": False,
}


@dataclass
class Range:
    """One continuous piece of source that ends up in the output."""
    src_in: float
    src_out: float
    seg_index: int
    zoom: float = 1.0
    cx: float = 0.5
    cy: float = 0.5
    out_start: float = 0.0  # filled in by layout()

    @property
    def duration(self) -> float:
        return self.src_out - self.src_in


@dataclass
class Plan:
    ranges: List[Range]
    chapters: List[Tuple[float, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(r.duration for r in self.ranges)


def _norm(word: str) -> str:
    return re.sub(r"[^a-z']", "", word.lower())


def load_edl(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        edl = json.load(f)
    if not isinstance(edl.get("segments"), list) or not edl["segments"]:
        raise ValueError("edl.json needs a non-empty 'segments' list")
    for i, s in enumerate(edl["segments"]):
        if "in" not in s or "out" not in s:
            raise ValueError(f"segment {i} needs 'in' and 'out'")
        if float(s["out"]) <= float(s["in"]):
            raise ValueError(f"segment {i}: out ({s['out']}) must be after in ({s['in']})")
    return edl


def tighten_range(
    a: float,
    b: float,
    words: Sequence[Dict[str, Any]],
    max_gap: float = 0.6,
    pad_before: float = 0.08,
    pad_after: float = 0.18,
    min_island: float = 0.25,
    drop_fillers: bool = False,
) -> List[Tuple[float, float]]:
    """Split [a, b] into speech islands, removing pauses longer than max_gap.

    A range with no words in it (a demo, b-roll, music) is kept whole.
    """
    inside = [w for w in words if w["s"] < b and w["e"] > a]
    if drop_fillers:
        inside = [w for w in inside if _norm(w["w"]) not in FILLERS]
    if not inside:
        return [(a, b)]

    islands: List[List[float]] = []
    for w in inside:
        if islands and w["s"] - islands[-1][1] <= max_gap:
            islands[-1][1] = max(islands[-1][1], w["e"])
        else:
            islands.append([w["s"], w["e"]])

    out: List[Tuple[float, float]] = []
    for s, e in islands:
        if e - s < min_island:
            continue
        s, e = max(a, s - pad_before), min(b, e + pad_after)
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out or [(a, b)]


def snap(t: float, fps: float) -> float:
    """Snap a time to the nearest frame boundary so audio and video lengths match."""
    return round(t * fps) / fps


def build_plan(
    edl: Dict[str, Any],
    words: Sequence[Dict[str, Any]],
    fps: float,
    source_duration: Optional[float] = None,
) -> Plan:
    tcfg = {**DEFAULT_TIGHTEN, **(edl.get("tighten") or {})}
    punch = edl.get("punch_in") or {}
    punch_mode = punch.get("mode", "off")
    punch_zoom = float(punch.get("zoom", 1.12))
    warnings: List[str] = []

    ranges: List[Range] = []
    chapter_marks: List[Tuple[int, str]] = []  # (index into ranges, title)
    for si, seg in enumerate(edl["segments"]):
        a, b = float(seg["in"]), float(seg["out"])
        if source_duration is not None:
            if a >= source_duration:
                warnings.append(f"segment {si} starts after the end of the source; skipped")
                continue
            if b > source_duration:
                warnings.append(f"segment {si} out {b:.2f} clipped to source end {source_duration:.2f}")
                b = source_duration
        do_tighten = seg.get("tighten", tcfg["enabled"])
        pieces = (
            tighten_range(a, b, words, tcfg["max_gap"], tcfg["pad_before"], tcfg["pad_after"],
                          tcfg["min_island"], tcfg["drop_fillers"])
            if do_tighten else [(a, b)]
        )
        first_index = len(ranges)
        for pi, (s, e) in enumerate(pieces):
            s, e = snap(s, fps), snap(e, fps)
            if e - s < 1.0 / fps:
                continue
            zoom = float(seg.get("zoom", 1.0))
            # Alternate punch-in on consecutive jump cuts within a segment hides the jump.
            if "zoom" not in seg and punch_mode == "alternate" and pi % 2 == 1:
                zoom = punch_zoom
            ranges.append(Range(s, e, si, zoom, float(seg.get("cx", 0.5)), float(seg.get("cy", 0.5))))
        if seg.get("chapter") and len(ranges) > first_index:
            chapter_marks.append((first_index, str(seg["chapter"])))

    layout(ranges)
    chapters = [(ranges[i].out_start, title) for i, title in chapter_marks]
    warnings += validate_chapters(chapters, sum(r.duration for r in ranges))
    if chapters and chapters[0][0] > 0:
        chapters[0] = (0.0, chapters[0][1])
    return Plan(ranges=ranges, chapters=chapters, warnings=warnings)


def layout(ranges: List[Range]) -> None:
    t = 0.0
    for r in ranges:
        r.out_start = t
        t += r.duration


def runs(ranges: Sequence[Range]) -> List[List[Range]]:
    """Group ranges into runs that move forward through the source.

    Each run renders in one ffmpeg pass; a jump backwards (a cold open, a reordered
    section) starts a new run.
    """
    out: List[List[Range]] = []
    for r in ranges:
        if out and r.src_in >= out[-1][-1].src_out - 1e-6:
            out[-1].append(r)
        else:
            out.append([r])
    return out


def validate_chapters(chapters: List[Tuple[float, str]], total: float) -> List[str]:
    """YouTube only shows chapters if: first at 0:00, at least 3, each at least 10s."""
    if not chapters:
        return []
    w = []
    if len(chapters) < 3:
        w.append(f"YouTube needs at least 3 chapters, the EDL has {len(chapters)}")
    if chapters[0][0] > 0.5:
        w.append("first chapter does not start at 0:00; it will be moved to 0:00")
    ends = [c[0] for c in chapters[1:]] + [total]
    for (start, title), end in zip(chapters, ends):
        if end - start < 10:
            w.append(f"chapter '{title}' is {end - start:.1f}s; YouTube needs 10s or more")
    return w


def map_time(t: float, ranges: Sequence[Range]) -> Optional[float]:
    """Source time -> output time (None if that moment was cut). Used for captions."""
    for r in ranges:
        if r.src_in <= t < r.src_out:
            return r.out_start + (t - r.src_in)
    return None


def hms(t: float, millis: bool = False) -> str:
    t = max(0.0, t)
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    if millis:
        ms = int(round((t - int(t)) * 1000))
        if ms == 1000:
            ms = 999
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def chapters_text(chapters: Sequence[Tuple[float, str]]) -> str:
    """The block you paste into the YouTube description."""
    return "\n".join(f"{hms(t)} {title}" for t, title in chapters) + ("\n" if chapters else "")


def ffmetadata(chapters: Sequence[Tuple[float, str]], total: float, title: Optional[str]) -> str:
    def esc(s: str) -> str:
        return re.sub(r"([=;#\\\n])", r"\\\1", s)

    lines = [";FFMETADATA1"]
    if title:
        lines.append(f"title={esc(title)}")
    ends = [c[0] for c in chapters[1:]] + [total]
    for (start, name), end in zip(chapters, ends):
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(start * 1000)}",
                  f"END={int(end * 1000)}", f"title={esc(name)}"]
    return "\n".join(lines) + "\n"


def captions_srt(
    words: Sequence[Dict[str, Any]],
    ranges: Sequence[Range],
    max_chars: int = 42,
    max_dur: float = 4.0,
    max_gap: float = 0.6,
    replacements: Optional[Dict[str, str]] = None,
    drop_fillers: bool = True,
) -> str:
    """Build an SRT for the edited timeline from source word timestamps."""
    reps = {k.lower(): v for k, v in (replacements or {}).items()}
    mapped = []
    for w in words:
        if drop_fillers and _norm(w["w"]) in FILLERS:
            continue
        s = map_time(w["s"], ranges)
        if s is None:
            continue
        e = map_time(max(w["s"], w["e"] - 1e-3), ranges)
        e = e if e is not None and e > s else s + max(0.1, w["e"] - w["s"])
        text = w["w"].strip()
        key = _norm(text)
        if key in reps:
            text = re.sub(re.escape(key), reps[key], text, flags=re.I)
        mapped.append((s, e, text))

    cues: List[Tuple[float, float, str]] = []
    cur: List[Tuple[float, float, str]] = []

    def flush():
        if cur:
            cues.append((cur[0][0], cur[-1][1], " ".join(t for _, _, t in cur)))
            cur.clear()

    for s, e, text in mapped:
        if cur:
            line = " ".join(t for _, _, t in cur) + " " + text
            if (len(line) > max_chars or e - cur[0][0] > max_dur or s - cur[-1][1] > max_gap):
                flush()
        cur.append((s, e, text))
        if text.endswith((".", "?", "!")):
            flush()
    flush()

    out = []
    for i, (s, e, text) in enumerate(cues, 1):
        if i < len(cues):
            e = min(e, cues[i][0])  # never overlap the next cue
        out.append(f"{i}\n{hms(s, True)} --> {hms(e, True)}\n{text}\n")
    return "\n".join(out)
