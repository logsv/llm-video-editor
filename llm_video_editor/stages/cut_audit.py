"""Stage 3 - ``cut-audit``: deterministic checks and fixes on the rough cut.

No LLM. Every A-roll cut is clamped to its own source clip's duration and
snapped to word boundaries from ``transcript.json``; cuts that end mid-sentence
are extended to the sentence end (or trimmed back to the previous one). Then
overlaps are merged, too-short clips dropped, and the duration budget enforced.
Finally ``EDL.pacing`` (jump-cut tightening, punch-ins) is applied; see pacing.py.

Outputs: cut_audit.json, edl_audited.json
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..schemas import (
    EDL,
    ClipsManifest,
    CutAudit,
    EDLClip,
    Graphic,
    MusicTrack,
    Transcript,
    Violation,
    Word,
)
from ..workspace import Workspace
from .pacing import apply_pacing, speech_islands

_SENTENCE_END = re.compile(r"[.!?…][\"'”’)\]]*$")


@dataclass(frozen=True)
class AuditConfig:
    lead: float = 0.08  # seconds of air kept before the first word
    tail: float = 0.15  # seconds of air kept after the last word
    max_extend: float = 2.5  # how far a cut may grow to reach a sentence end
    min_clip: float = 0.6
    tolerance: float = 0.05  # duration budget, fraction of target
    # False (agent mode): editorial issues (word/sentence snapping, overlaps, short
    # clips, duration budget) are reported with a suggested fix and left for the agent
    # to decide; correctness guardrails (unknown source, out of bounds) still apply.
    editorial_fixes: bool = True


def ends_sentence(word: Word) -> bool:
    return bool(_SENTENCE_END.search(word.text.strip()))


class _Recorder:
    def __init__(self):
        self.items: list[Violation] = []

    def add(self, kind, clip: EDLClip | None, message: str, *, at=None, fixed=False, fix=None,
            severity="error"):
        self.items.append(Violation(
            kind=kind, severity=severity, clip_id=clip.clip_id if clip else None,
            source_file=clip.source_file if clip else None,
            at=round(at, 3) if at is not None else None, message=message, fixed=fixed, fix=fix,
        ))


DOUBTFUL_PROB = 0.4
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def doubtful(w: Word, language: str | None = None) -> bool:
    """A word ASR probably invented (wind, engine, crowd noise): low confidence, or
    Chinese/Japanese/Korean characters in a transcript of another language."""
    if w.prob < DOUBTFUL_PROB:
        return True
    return bool(_CJK.search(w.text)) and (language or "")[:2] not in ("zh", "ja", "ko", "yu")


def _word_note(w: Word, language: str | None) -> str:
    return (f" - doubtful word (confidence {w.prob:.2f}): probably noise, not speech"
            if doubtful(w, language) else "")


def _snap(clip: EDLClip, words: list[Word], duration: float, cfg: AuditConfig,
          rec: _Recorder, language: str | None = None) -> EDLClip | None:
    s, e = clip.start, clip.end
    if not words:
        return clip

    # -- in point ---------------------------------------------------------------
    inside = next((w for w in words if w.start + 1e-3 < s < w.end - 1e-3), None)
    if inside is not None:
        rec.add("mid_word_in", clip, f"in point {s:.3f}s cuts the word {inside.text!r}"
                f"{_word_note(inside, language)}",
                at=s, fixed=True, fix=f"moved to word start {inside.start:.3f}s")
        s = inside.start
    first_i = next((i for i, w in enumerate(words) if w.start >= s - 1e-3), None)
    if first_i is None or words[first_i].start >= e:
        return clip  # no speech inside the cut (pure room tone); leave as is
    prev_end = words[first_i - 1].end if first_i > 0 else 0.0
    s = max(prev_end, words[first_i].start - cfg.lead, 0.0) if s <= words[first_i].start else s

    # -- out point --------------------------------------------------------------
    inside = next((w for w in words if w.start + 1e-3 < e < w.end - 1e-3), None)
    if inside is not None:
        rec.add("mid_word_out", clip, f"out point {e:.3f}s cuts the word {inside.text!r}"
                f"{_word_note(inside, language)}",
                at=e, fixed=True, fix=f"moved to word end {inside.end:.3f}s")
        e = inside.end
    last_i = max((i for i, w in enumerate(words) if w.end <= e + 1e-3), default=None)
    if last_i is None or last_i < first_i:
        return EDLClip(**{**clip.model_dump(), "start": round(s, 3), "end": round(e, 3)})

    if not ends_sentence(words[last_i]):
        has_punct = any(ends_sentence(w) for w in words)
        fwd = next((j for j in range(last_i + 1, len(words))
                    if words[j].end - e <= cfg.max_extend and words[j].end <= duration
                    and ends_sentence(words[j])), None)
        if fwd is not None and words[fwd].end - e <= cfg.max_extend:
            rec.add("mid_sentence_out", clip, f"cut ends mid-sentence after {words[last_i].text!r}",
                    at=e, fixed=True, fix=f"extended to sentence end {words[fwd].end:.3f}s")
            last_i = fwd
        else:
            back = next((j for j in range(last_i - 1, first_i - 1, -1) if ends_sentence(words[j])),
                        None)
            keep = words[back].end - s if back is not None else 0.0
            if back is not None and keep >= max(cfg.min_clip, 0.5 * (e - s)):
                rec.add("mid_sentence_out", clip,
                        f"cut ends mid-sentence after {words[last_i].text!r}", at=e, fixed=True,
                        fix=f"trimmed back to sentence end {words[back].end:.3f}s")
                last_i = back
            else:
                rec.add("mid_sentence_out", clip,
                        f"cut ends mid-sentence after {words[last_i].text!r} and no sentence "
                        "end is within reach", at=e,
                        severity="error" if has_punct else "warning")
    next_start = words[last_i + 1].start if last_i + 1 < len(words) else duration
    e = min(words[last_i].end + cfg.tail, next_start, duration)
    return EDLClip(**{**clip.model_dump(), "start": round(s, 3), "end": round(e, 3)})


def _as_suggestion(v: Violation) -> Violation:
    """A fix the classic audit would apply, reported instead (agent mode)."""
    fix = f"suggested: {v.fix}" if v.fix else None
    return v.model_copy(update={"fixed": False, "fix": fix, "severity": "warning"})


def audit(edl: EDL, transcript: Transcript, manifest: ClipsManifest,
          cfg: AuditConfig = AuditConfig()) -> tuple[EDL, CutAudit]:
    rec = _Recorder()
    before_n, before_d = len(edl.clips), edl.total_duration
    clips: list[EDLClip] = []

    for clip in edl.clips:
        info = manifest.get(clip.source_clip_id)
        if info is None or info.source_file != clip.source_file:
            rec.add("unknown_source", clip, "clip references a source not in clips_manifest",
                    fixed=True, fix="dropped clip")
            continue
        s, e = clip.start, clip.end
        if s < 0 or e > info.duration or e <= s:
            ns, ne = max(0.0, s), min(info.duration, e)
            if ne - ns < cfg.min_clip:
                rec.add("out_of_bounds", clip,
                        f"{s:.3f}-{e:.3f}s is outside the {info.duration:.3f}s source",
                        at=s, fixed=True, fix="dropped clip")
                continue
            rec.add("out_of_bounds", clip, f"{s:.3f}-{e:.3f}s exceeds the {info.duration:.3f}s source",
                    at=s if s < 0 else e, fixed=True, fix=f"clamped to {ns:.3f}-{ne:.3f}s")
            clip = EDLClip(**{**clip.model_dump(), "start": ns, "end": ne})
        ct = transcript.for_clip(clip.source_clip_id)
        words = ct.words() if ct and clip.role == "a_roll" else []
        lang = ct.language if ct else None
        if cfg.editorial_fixes:
            snapped = _snap(clip, words, info.duration, cfg, rec, lang)
            if snapped is not None:
                clips.append(snapped)
        else:
            scratch = _Recorder()
            _snap(clip, words, info.duration, cfg, scratch, lang)
            for v in scratch.items:
                rec.items.append(_as_suggestion(v))
            clips.append(clip)

    # -- overlaps between cuts from the same source ---------------------------------
    merged: list[EDLClip] = []
    for clip in clips:
        prev = merged[-1] if merged else None
        if (prev and prev.source_clip_id == clip.source_clip_id and clip.start >= prev.start
                and clip.start < prev.end and clip.end > prev.start):  # forward continuation
            if not cfg.editorial_fixes:
                rec.add("overlap", clip, f"overlaps {prev.clip_id} in the source", at=clip.start,
                        severity="warning", fix=f"suggested: merge into {prev.clip_id}")
                merged.append(clip)
                continue
            rec.add("overlap", clip, f"overlaps {prev.clip_id} in the source", at=clip.start,
                    fixed=True, fix=f"merged into {prev.clip_id}")
            merged[-1] = EDLClip(**{**prev.model_dump(), "start": min(prev.start, clip.start),
                                    "end": max(prev.end, clip.end),
                                    "priority": min(prev.priority, clip.priority)})
            continue
        for other in merged:
            if (other.source_clip_id == clip.source_clip_id
                    and clip.start <= other.start and other.end <= clip.end):
                # an earlier clip replayed inside a later one: a cold open / teaser
                rec.add("overlap", clip, f"replays {other.clip_id} (teaser or cold open)",
                        at=other.start, severity="warning")
                continue
            if (other.source_clip_id == clip.source_clip_id
                    and clip.start < other.end and clip.end > other.start):
                if not cfg.editorial_fixes:
                    rec.add("overlap", clip, f"repeats source material from {other.clip_id}",
                            at=clip.start, severity="warning",
                            fix=f"suggested: start at {other.end:.3f}s")
                    continue
                rec.add("overlap", clip, f"repeats source material from {other.clip_id}",
                        at=clip.start, fixed=True, fix=f"start moved to {other.end:.3f}s")
                clip = EDLClip(**{**clip.model_dump(), "start": other.end})
        merged.append(clip)

    kept = []
    for clip in merged:
        if clip.duration < cfg.min_clip and cfg.editorial_fixes:
            rec.add("too_short", clip, f"only {clip.duration:.2f}s long", fixed=True,
                    fix="dropped clip")
        elif clip.duration < cfg.min_clip:
            rec.add("too_short", clip, f"only {clip.duration:.2f}s long", severity="warning",
                    fix="suggested: remove it or lengthen it")
            kept.append(clip)
        else:
            kept.append(clip)

    # -- duration budget ------------------------------------------------------------
    target = edl.target_duration
    # judge the budget on the length the clips will have *after* jump-cut tightening,
    # otherwise pauses that pacing is about to remove cost whole clips here
    words_by_clip = {ct.clip_id: ct.words() for ct in transcript.clips}

    def length(c: EDLClip) -> float:
        if not edl.pacing.tighten or c.role != "a_roll":
            return c.duration
        spans = speech_islands(c.start, c.end, words_by_clip.get(c.source_clip_id, []),
                               edl.pacing)
        return sum(e - s for s, e in spans)

    if target:
        total = sum(length(c) for c in kept)
        hi, lo = target * (1 + cfg.tolerance), target * (1 - cfg.tolerance)
        if total > hi and not cfg.editorial_fixes:
            rec.add("duration_budget", None,
                    f"{total:.1f}s is over target {target:.1f}s (+{cfg.tolerance:.0%})",
                    severity="warning",
                    fix=f"suggested: remove about {total - target:.0f}s")
        elif total > hi:
            order = sorted(range(len(kept)), key=lambda i: (-kept[i].priority, -i))
            dropped = set()
            for i in order:
                if total <= hi or len(kept) - len(dropped) <= 1:
                    break
                dropped.add(i)
                total -= length(kept[i])
            for i in sorted(dropped):
                rec.add("duration_budget", kept[i], "over the duration budget", fixed=True,
                        fix=f"dropped priority-{kept[i].priority} clip")
            kept = [c for i, c in enumerate(kept) if i not in dropped]
            if total > hi:
                rec.add("duration_budget", None,
                        f"{total:.1f}s still exceeds target {target:.1f}s (+{cfg.tolerance:.0%})")
        elif total < lo:
            rec.add("duration_budget", None,
                    f"{total:.1f}s is under target {target:.1f}s (-{cfg.tolerance:.0%})",
                    severity="warning")

    # -- pacing (after the editorial fixes, so splits land inside audited clips) -----
    words_for = {ct.clip_id: ct.words() for ct in transcript.clips}
    kept, paced_overlays, paced_graphics, paced_music, removed = apply_pacing(
        edl, kept, words_for, list(edl.overlays), list(edl.graphics), list(edl.music))

    # -- overlays ---------------------------------------------------------------------
    by_id = {c.clip_id: c for c in kept}
    overlays = []
    for ov in paced_overlays:
        anchor, src = by_id.get(ov.anchor_clip_id), manifest.get(ov.source_clip_id)
        if anchor is None or src is None:
            rec.add("overlay_out_of_bounds", None,
                    f"overlay {ov.overlay_id}: anchor or source clip no longer exists",
                    fixed=True, fix="dropped overlay")
            continue
        offset = min(max(0.0, ov.offset), max(0.0, anchor.duration - 0.5))
        dur = min(ov.duration, anchor.duration - offset, src.duration)
        src_start = min(max(0.0, ov.src_start), max(0.0, src.duration - dur))
        if dur < 0.5:
            rec.add("overlay_out_of_bounds", None, f"overlay {ov.overlay_id} has no room",
                    fixed=True, fix="dropped overlay")
            continue
        if (offset, dur, src_start) != (ov.offset, ov.duration, ov.src_start):
            rec.add("overlay_out_of_bounds", None, f"overlay {ov.overlay_id} exceeded its bounds",
                    fixed=True, fix=f"offset {offset:.2f}s, {dur:.2f}s from {src_start:.2f}s")
        overlays.append(ov.model_copy(update={
            "offset": round(offset, 3), "duration": round(dur, 3), "src_start": round(src_start, 3)}))

    graphics = _audit_graphics(paced_graphics, kept, rec)
    music = _audit_music(paced_music, kept, rec)
    audited = edl.model_copy(update={"clips": kept, "overlays": overlays, "graphics": graphics,
                                     "music": music})
    report = CutAudit(
        passed=not any(v.severity == "error" and not v.fixed for v in rec.items),
        violations=rec.items, clips_before=before_n, clips_after=len(kept),
        duration_before=round(before_d, 3), duration_after=round(audited.total_duration, 3),
        target_duration=target, tolerance=cfg.tolerance, pauses_removed=removed,
    )
    return audited, report


def _audit_graphics(graphics: list[Graphic], clips: list[EDLClip], rec: _Recorder
                    ) -> list[Graphic]:
    """Drop graphics with a missing anchor or source; keep each inside the timeline."""
    from ..hyperframes.compose import templates

    starts, t = {}, 0.0
    for c in clips:
        starts[c.clip_id] = t
        t += c.duration
    total, known = t, templates()
    out = []
    for g in graphics:
        if g.template is not None and g.template not in known:
            rec.add("unknown_graphic", None, f"graphic {g.graphic_id}: no template {g.template!r} "
                    f"(have {', '.join(sorted(known))})", fixed=True, fix="dropped graphic")
            continue
        if g.composition is not None and not (Path(g.composition).expanduser() / "index.html"
                                              ).exists():
            rec.add("unknown_graphic", None, f"graphic {g.graphic_id}: {g.composition} has no "
                    "index.html", fixed=True, fix="dropped graphic")
            continue
        if g.anchor_clip_id not in starts:
            rec.add("graphic_out_of_bounds", None,
                    f"graphic {g.graphic_id}: anchor {g.anchor_clip_id} is not in the cut",
                    fixed=True, fix="dropped graphic")
            continue
        at = starts[g.anchor_clip_id] + max(0.0, g.offset)
        dur = min(g.duration, total - at)
        if dur < 0.5:
            rec.add("graphic_out_of_bounds", None, f"graphic {g.graphic_id} has no room",
                    fixed=True, fix="dropped graphic")
            continue
        if dur < g.duration or g.offset < 0:
            rec.add("graphic_out_of_bounds", None,
                    f"graphic {g.graphic_id} ran past the end of the cut", fixed=True,
                    fix=f"{dur:.2f}s from {at:.2f}s")
        out.append(g.model_copy(update={"offset": round(max(0.0, g.offset), 3),
                                        "duration": round(dur, 3)}))
    return out


def _audit_music(tracks: list[MusicTrack], clips: list[EDLClip], rec: _Recorder
                 ) -> list[MusicTrack]:
    """Drop tracks with a missing file or anchor; keep each inside the timeline."""
    starts, t = {}, 0.0
    for c in clips:
        starts[c.clip_id] = t
        t += c.duration
    total, out = t, []
    for m in tracks:
        if not Path(m.file).expanduser().exists():
            rec.add("unknown_music", None, f"music {m.track_id}: {m.file} does not exist",
                    fixed=True, fix="dropped track")
            continue
        if m.anchor_clip_id not in starts:
            rec.add("music_out_of_bounds", None,
                    f"music {m.track_id}: anchor {m.anchor_clip_id} is not in the cut",
                    fixed=True, fix="dropped track")
            continue
        at = starts[m.anchor_clip_id] + max(0.0, m.offset)
        room = total - at
        if room < 0.3:
            rec.add("music_out_of_bounds", None, f"music {m.track_id} starts at the end",
                    fixed=True, fix="dropped track")
            continue
        dur = m.duration
        if dur is not None and dur > room:
            rec.add("music_out_of_bounds", None, f"music {m.track_id} ran past the end of the cut",
                    fixed=True, fix=f"{room:.2f}s from {at:.2f}s")
            dur = round(room, 3)
        out.append(m.model_copy(update={"offset": round(max(0.0, m.offset), 3), "duration": dur}))
    return out


def run_cut_audit(ws: Workspace, edl: EDL | None = None, cfg: AuditConfig = AuditConfig()
                  ) -> tuple[EDL, CutAudit]:
    edl = edl or Workspace.read(ws.edl_raw, EDL)
    audited, report = audit(edl, Workspace.read(ws.transcript, Transcript),
                            Workspace.read(ws.clips_manifest, ClipsManifest), cfg)
    Workspace.write(ws.edl_audited, audited)
    Workspace.write(ws.cut_audit, report)
    ws.record("cut_audit", ws.stage_key("cut_audit", edl.model_dump(), cfg),
              [ws.edl_audited, ws.cut_audit])
    return audited, report
