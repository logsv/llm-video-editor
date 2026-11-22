"""Captions from word timestamps on the output timeline.

Burned-in ASS for social cuts, plus a sidecar SRT for every render (upload it
in YouTube Studio; better than burned-in text for long-form). Standalone
fillers (um/uh) are left out, and ``CaptionSettings.replacements`` fixes ASR
mishearings of names and terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..schemas import EDL, Transcript, Word
from .cut_audit import ends_sentence


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    words: list[tuple[float, float, str]]  # (start, end, text) on the output timeline
    clip_index: int = 0

    @property
    def text(self) -> str:
        return " ".join(w for _, _, w in self.words)


def fix_word(text: str, replacements: dict[str, str]) -> str:
    """Replace a mis-heard word, keeping its punctuation: 'cloud,' -> 'Claude,'."""
    core = re.sub(r"[^\w'-]", "", text).lower()
    if core and core in replacements:
        return re.sub(re.escape(core), replacements[core], text, count=1, flags=re.I)
    return text


def output_words(edl: EDL, transcript: Transcript, starts: list[float],
                 durations: list[float]) -> list[tuple[float, float, str, int]]:
    """Map every word spoken inside an A-roll clip onto the output timeline.

    Returns ``(start, end, text, clip_index)`` tuples.
    """
    from .cut_audit import doubtful
    from .pacing import is_filler

    reps = {k.lower(): v for k, v in edl.captions.replacements.items()}
    out = []
    for i, (clip, t0, dur) in enumerate(zip(edl.clips, starts, durations, strict=True)):
        if clip.role != "a_roll":
            continue
        ct = transcript.for_clip(clip.source_clip_id)
        if ct is None:
            continue
        for w in ct.words():
            if w.end <= clip.start or w.start >= clip.start + dur or is_filler(w):
                continue
            if doubtful(w, ct.language):  # invented over wind or engine noise
                continue
            s = max(0.0, w.start - clip.start)
            e = min(dur, w.end - clip.start)
            if e - s > 0.02:
                out.append((round(t0 + s, 3), round(t0 + e, 3), fix_word(w.text, reps), i))
    out += voiceover_words(edl, starts, sum(durations), reps)
    return sorted(out)


def voiceover_words(edl: EDL, starts: list[float], total: float,
                    reps: dict[str, str] | None = None) -> list[tuple[float, float, str, int]]:
    """Voice-over lines with text, spread over each line's length by word length (TTS
    reads at an even pace, so this lands close to the spoken words)."""
    from ..ffmpeg.probe import media_duration

    index = {c.clip_id: i for i, c in enumerate(edl.clips)}
    out = []
    for m in edl.music:
        if m.role != "voiceover" or not m.text or m.anchor_clip_id not in index:
            continue
        i = index[m.anchor_clip_id]
        at = starts[i] + m.offset
        try:
            length = m.duration or max(0.0, media_duration(Path(m.file).expanduser())
                                       - m.src_start)
        except Exception:  # noqa: BLE001 - a missing file is reported by audit
            continue
        length = min(length, max(0.0, total - at))
        words = m.text.split()
        weights = [len(w) + 1 for w in words]
        t, per = at, length / max(1, sum(weights))
        for w, k in zip(words, weights, strict=True):
            out.append((round(t, 3), round(t + k * per, 3), fix_word(w, reps or {}), i))
            t += k * per
    return out


def group_cues(words: list[tuple[float, float, str, int]], max_words: int = 6,
               max_gap: float = 0.6) -> list[Cue]:
    """Group words into caption cues; never across a cut or a long pause."""
    cues: list[Cue] = []
    cur: list[tuple[float, float, str, int]] = []

    def flush():
        if cur:
            cues.append(Cue(cur[0][0], cur[-1][1], [w[:3] for w in cur], cur[0][3]))
            cur.clear()

    for w in words:
        if cur and (len(cur) >= max_words or w[0] - cur[-1][1] > max_gap or w[3] != cur[-1][3]):
            flush()
        cur.append(w)
        if ends_sentence(Word(start=w[0], end=w[1], text=w[2])):
            flush()
    flush()
    # hold each cue until the next one starts (up to 0.4s) so text does not flicker
    held = []
    for i, c in enumerate(cues):
        nxt = cues[i + 1].start if i + 1 < len(cues) else c.end + 0.4
        held.append(Cue(c.start, min(nxt, c.end + 0.4), c.words, c.clip_index))
    return held


def _ts(t: float) -> str:
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


ALIGN = {"bottom": 2, "auto": 2, "center": 5, "top": 8}


def build_ass(cues: list[Cue], width: int, height: int, *, style: str = "lines",
              position: str = "auto", margin_v: int | None = None,
              clip_positions: dict[int, str] | None = None) -> str:
    size = round(min(width, height) * 0.065)
    vertical = height > width
    margin_v = margin_v if margin_v is not None else round(height * (0.14 if vertical else 0.07))
    head = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {width}\nPlayResY: {height}\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,DejaVu Sans,{size},&H00FFFFFF,&H0000D7FF,&H00000000,&H64000000,"
        f"-1,0,0,0,100,100,0,0,1,{max(2, size // 14)},0,{ALIGN.get(position, 2)},"
        f"{round(width * 0.08)},{round(width * 0.08)},{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = []
    for c in cues:
        if style == "word_highlight":
            parts, t = [], c.start
            for ws, we, w in c.words:
                gap = max(0, round((ws - t) * 100))
                if gap:
                    parts.append(f"{{\\k{gap}}}")
                parts.append(f"{{\\k{max(1, round((we - ws) * 100))}}}{_esc(w)} ")
                t = we
            text = "".join(parts).rstrip()
        else:
            text = _esc(c.text)
        pos = (clip_positions or {}).get(c.clip_index)
        if pos and position == "auto" and ALIGN[pos] != ALIGN[position]:
            text = f"{{\\an{ALIGN[pos]}}}" + text
        lines.append(f"Dialogue: 0,{_ts(c.start)},{_ts(c.end)},Default,,0,0,0,,{text}")
    return head + "\n".join(lines) + "\n"


def write_captions(path: Path, edl: EDL, transcript: Transcript, starts: list[float],
                   durations: list[float]) -> Path | None:
    words = output_words(edl, transcript, starts, durations)
    if not words:
        return None
    cues = group_cues(words, edl.captions.max_words_per_line)
    positions = {i: c.caption_position for i, c in enumerate(edl.clips) if c.caption_position}
    path.write_text(build_ass(cues, edl.width, edl.height, style=edl.captions.style,
                              position=edl.captions.position, clip_positions=positions),
                    encoding="utf-8")
    return path


def _srt_ts(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def build_srt(cues: list[Cue]) -> str:
    return "\n".join(f"{i}\n{_srt_ts(c.start)} --> {_srt_ts(c.end)}\n{c.text}\n"
                     for i, c in enumerate(cues, 1))


def write_srt(path: Path, edl: EDL, transcript: Transcript, starts: list[float],
              durations: list[float], max_words: int = 10) -> Path | None:
    """Sidecar captions: longer lines than burned-in ones (players wrap them)."""
    words = output_words(edl, transcript, starts, durations)
    if not words:
        return None
    path.write_text(build_srt(group_cues(words, max_words)), encoding="utf-8")
    return path
