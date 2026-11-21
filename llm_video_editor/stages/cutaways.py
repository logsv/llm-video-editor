"""Cutaways: a few licensed B-roll or funny clips, only where they help.

``suggest`` asks the planner (text only: clip ids, numbered transcript lines) for
cutaway moments and search words. It may answer with none, and the result is
held to a budget locally: at most one cutaway per ``every`` seconds of the cut,
humour at most one per ``humor_every`` seconds, never inside the first seconds of
the video, and at least ``spacing`` seconds apart.

``fetch`` searches licensed video libraries (Pixabay, Pexels, Wikimedia Commons)
for each slot, downloads the best match into a B-roll folder with a credit
sidecar, registers it in ``clips_manifest.json`` and adds a full-frame overlay to
``edl_raw.json`` (other hand edits there are kept). Social platforms and
unlicensed web clips are never searched: their clips are copyrighted.

Outputs: media/cutaways.json, edl_raw.json (overlays), credits.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..ffmpeg.probe import probe
from ..llm import Provider, structured
from ..media import library as lib
from ..media.providers import DEFAULT_FOR
from ..schemas import EDL, ClipInfo, ClipsManifest, MediaResult, Overlay, Strict, Transcript
from ..schemas.base import Checkpoint
from ..workspace import SourceFingerprint, Workspace

SYSTEM = (
    "You are a video editor deciding where a short cutaway would make a cut better: "
    "stock B-roll that shows what is being described, or a funny reaction clip on a "
    "joke or exaggeration. Most lines need nothing. Suggest a cutaway only when it "
    "clearly adds; an empty list is a good answer. Never cover the opening hook or a "
    "step the viewer must watch the speaker explain. Search words must describe generic "
    "footage (no real people, brands, films, shows, memes or copyrighted characters). "
    "Answer strictly in the requested JSON schema."
)


class _Pick(Strict):
    clip_id: str = Field(description="EDL clip id the moment is in")
    segment_id: int = Field(description="transcript segment id where the cutaway starts")
    kind: Literal["illustrate", "humor"]
    query: str = Field(description="2-6 stock-footage search words")
    seconds: float = Field(description="1.5-3 for humor, 2-6 for illustrate")
    reason: str = Field(description="why this moment needs it, <= 15 words")


class _Picks(Strict):
    cutaways: list[_Pick] = Field(default_factory=list)


class CutawaySlot(Strict):
    slot: int
    kind: Literal["illustrate", "humor"]
    anchor_clip_id: str
    offset: float  # seconds into the anchor clip (output time)
    duration: float
    query: str
    reason: str = ""
    chosen: MediaResult | None = None
    file: str | None = None
    overlay_id: str | None = None
    note: str = ""


class CutawayPlan(Checkpoint):
    slots: list[CutawaySlot] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)  # suggestions removed by the budget


@dataclass(frozen=True)
class Budget:
    every: float = 45.0  # at most one cutaway per this many seconds of the cut
    humor_every: float = 120.0  # at most one funny clip per this many seconds
    spacing: float = 20.0  # minimum gap between cutaways
    skip_start: float = 5.0  # never cover the opening seconds (the hook)
    humor: bool = True


def plan_path(ws: Workspace) -> Path:
    return lib.media_dir(ws) / "cutaways.json"


def _material(edl: EDL, transcript: Transcript) -> str:
    lines, t0 = [], 0.0
    for clip in edl.clips:
        ct = transcript.for_clip(clip.source_clip_id)
        if clip.role != "a_roll" or ct is None:
            t0 += clip.duration
            continue
        lines.append(f"## clip {clip.clip_id} (output {t0:.1f}-{t0 + clip.duration:.1f}s)")
        for seg in ct.segments:
            if seg.end > clip.start and seg.start < clip.end:
                lines.append(f"[{seg.id}] {seg.text}")
        t0 += clip.duration
    return "\n".join(lines)


def enforce_budget(picks: list[_Pick], edl: EDL, transcript: Transcript, budget: Budget
                   ) -> tuple[list[CutawaySlot], list[str]]:
    starts = dict(zip([c.clip_id for c in edl.clips], edl.output_starts(), strict=True))
    clips = {c.clip_id: c for c in edl.clips}
    total = edl.total_duration
    max_all = max(1, int(total // budget.every))
    max_humor = int(total // budget.humor_every) if budget.humor else 0
    slots, dropped, times, humor = [], [], [], 0
    for p in picks:
        clip = clips.get(p.clip_id)
        ct = transcript.for_clip(clip.source_clip_id) if clip else None
        seg = next((s for s in ct.segments if s.id == p.segment_id), None) if ct else None
        if clip is None or seg is None or not (clip.start <= seg.start < clip.end):
            dropped.append(f"{p.clip_id}/{p.segment_id}: not in the cut")
            continue
        offset = seg.start - clip.start
        at = starts[clip.clip_id] + offset
        lo, hi = (1.5, 3.0) if p.kind == "humor" else (2.0, 6.0)
        dur = min(max(p.seconds, lo), hi, clip.end - seg.start)
        why = None
        if at < budget.skip_start:
            why = "covers the opening hook"
        elif p.kind == "humor" and humor >= max_humor:
            why = "humor budget used"
        elif len(slots) >= max_all:
            why = "cutaway budget used"
        elif any(abs(at - t) < budget.spacing for t in times):
            why = f"closer than {budget.spacing:g}s to another cutaway"
        elif dur < 1.0:
            why = "no room left in the clip"
        if why:
            dropped.append(f"{p.clip_id}/{p.segment_id} ({p.kind}): {why}")
            continue
        slots.append(CutawaySlot(slot=len(slots) + 1, kind=p.kind, anchor_clip_id=clip.clip_id,
                                 offset=round(offset, 3), duration=round(dur, 3),
                                 query=p.query.strip(), reason=p.reason))
        times.append(at)
        humor += p.kind == "humor"
    return slots, dropped


def suggest(ws: Workspace, provider: Provider, *, budget: Budget = Budget(),
            brief: str = "") -> CutawayPlan:
    edl = Workspace.read(ws.edl_raw, EDL)
    transcript = Workspace.read(ws.transcript, Transcript)
    total = edl.total_duration
    prompt = (
        f"CUT: {total:.0f}s for {edl.target_platform}. {brief}\n"
        f"BUDGET: at most {max(1, int(total // budget.every))} cutaways in total"
        + (f", at most {int(total // budget.humor_every)} of them humor" if budget.humor
           else ", no humor")
        + f"; none in the first {budget.skip_start:g}s.\n\n"
        "Pick moments by clip id and transcript segment id.\n\n" + _material(edl, transcript))
    picks = structured(provider, SYSTEM, prompt, _Picks).cutaways
    if not budget.humor:
        picks = [p for p in picks if p.kind != "humor"]
    slots, dropped = enforce_budget(picks, edl, transcript, budget)
    plan = CutawayPlan(slots=slots, dropped=dropped)
    Workspace.write(plan_path(ws), plan)
    return plan


def _pick(results: list[MediaResult], seconds: float, portrait: bool) -> MediaResult | None:
    def ok(r: MediaResult) -> bool:
        return r.duration is None or r.duration >= seconds

    def orient(r: MediaResult) -> bool:
        return r.width is None or r.height is None or (r.height > r.width) == portrait

    for pool in ([r for r in results if ok(r) and orient(r)], [r for r in results if ok(r)]):
        if pool:
            return pool[0]
    return None


def _register(manifest: ClipsManifest, path: Path, description: str) -> ClipInfo:
    fp = SourceFingerprint.of(path)
    info = manifest.get(fp.clip_id)
    if info is None:
        p = probe(path)
        info = ClipInfo(clip_id=fp.clip_id, source_file=str(path.resolve()),
                        duration=round(p.duration, 3), width=p.width, height=p.height, fps=p.fps,
                        has_audio=p.has_audio, video_codec=p.video_codec,
                        audio_codec=p.audio_codec, kind="broll")
        manifest.clips.append(info)
    return info


def fetch(ws: Workspace, *, library: Path | None = None, providers: tuple[str, ...] | None = None,
          search=lib.search, download=lib.fetch) -> CutawayPlan:
    """Search, download and place every unfilled slot of media/cutaways.json."""
    plan = Workspace.read(plan_path(ws), CutawayPlan)
    edl = Workspace.read(ws.edl_raw, EDL)
    manifest = Workspace.read(ws.clips_manifest, ClipsManifest)
    portrait = edl.height > edl.width
    overlays = list(edl.overlays)
    used = {o.overlay_id for o in overlays}
    slots = []
    for s in plan.slots:
        if s.overlay_id and s.overlay_id in used:
            slots.append(s)  # already placed
            continue
        res = search(s.query, "video", providers=providers or DEFAULT_FOR["video"], limit=8)
        best = _pick(res.results, s.duration, portrait)
        if best is None:
            note = "; ".join(res.errors) or "no licensed clip long enough"
            slots.append(s.model_copy(update={"note": note}))
            continue
        path = download(ws, best, library=library, description=f"{s.query}. {best.title}")
        info = _register(manifest, Path(path), s.query)
        n = 1
        while f"x{n:03d}" in used:
            n += 1
        oid = f"x{n:03d}"
        used.add(oid)
        overlays.append(Overlay(overlay_id=oid, source_clip_id=info.clip_id,
                                source_file=info.source_file, anchor_clip_id=s.anchor_clip_id,
                                offset=s.offset, duration=min(s.duration, info.duration)))
        slots.append(s.model_copy(update={"chosen": best, "file": str(path), "overlay_id": oid,
                                          "note": ""}))
    plan = plan.model_copy(update={"slots": slots})
    Workspace.write(ws.clips_manifest, manifest)
    Workspace.write(ws.edl_raw, edl.model_copy(update={"overlays": overlays}))
    Workspace.write(plan_path(ws), plan)
    return plan
