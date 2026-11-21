"""Stage 2 - ``plan``: map/reduce planning over the whole transcript.

The LLM never invents timestamps. It sees clip ids and numbered transcript
segments and answers with segment-id ranges; this module resolves those to
times from ``transcript.json``. Only text crosses the provider boundary.

Outputs: asset_plan.json, edl_raw.json
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pydantic import Field

from ..llm import Provider, structured
from ..presets.platform_presets import get_platform_specs
from ..schemas import (
    EDL,
    AssetPlan,
    BrollPlacement,
    ClipInfo,
    ClipsManifest,
    ClipTranscript,
    EDLClip,
    Overlay,
    PlannedSegment,
    Segment,
    Strict,
    Transcript,
)
from ..workspace import Workspace
from .durations import duration_from_prompt

DIRECT_LIMIT_CHARS = 24_000  # below this the whole transcript goes to one reduce call
CHUNK_SECONDS = 60.0
MAP_BATCH_CHARS = 12_000


# -- LLM response schemas -----------------------------------------------------
class ChunkScore(Strict):
    chunk_id: str
    score: int = Field(description="0-10: how well this chunk serves the request")
    summary: str = Field(description="<= 25 words")
    best_segments: list[int] = Field(description="segment ids in this chunk worth keeping")


class MapResult(Strict):
    chunks: list[ChunkScore]


class Selection(Strict):
    clip_id: str
    first_segment: int
    last_segment: int
    purpose: str = Field(description="why this part is in the edit, <= 12 words")
    priority: int = Field(description="1 = essential, 2 = important, 3 = nice to have")


class BrollPick(Strict):
    over_selection: int = Field(description="index into selections the B-roll covers")
    clip_id: str | None = Field(description="a B-roll clip id from the catalog, or null")
    query: str = Field(description="what the B-roll should show")
    seconds: float


class ReduceResult(Strict):
    selections: list[Selection]
    broll: list[BrollPick] = Field(default_factory=list)
    notes: str = ""


SYSTEM = (
    "You are a senior video editor planning a cut from transcripts. You only see text: "
    "clip ids, numbered transcript segments with times, and a B-roll catalog. "
    "Choose whole sentences, keep the narrative coherent, and hit the duration target. "
    "Answer strictly in the requested JSON schema."
)


# -- prompt building ------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    clip_id: str
    start: float
    end: float
    lines: list[str]

    @property
    def text(self) -> str:
        head = f"## chunk {self.chunk_id} (clip {self.clip_id}, {self.start:.1f}-{self.end:.1f}s)"
        return "\n".join([head, *self.lines])


def _opening(text: str, n: int = 4) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]", "", text.lower()).split()[:n])


def retake_ids(segments: list[Segment], lookahead: int = 3) -> set[int]:
    """Segments whose opening words are repeated by one of the next few segments.

    People restart a line until it lands; the earlier attempts are usually cut and
    the last complete take kept. A hint for the planner, not a verdict.
    """
    out = set()
    for i, seg in enumerate(segments):
        key = _opening(seg.text)
        if len(key.split()) >= 3 and any(_opening(n.text) == key
                                         for n in segments[i + 1:i + 1 + lookahead]):
            out.add(seg.id)
    return out


def _seg_line(seg: Segment, retake: bool = False) -> str:
    return f"[{seg.id}] {seg.start:.1f}-{seg.end:.1f} {seg.text}" + (" (retake?)" if retake else "")


def format_transcript(ct: ClipTranscript, min_pause: float = 2.0) -> str:
    """Compact, human/LLM-readable view of one clip: ids, times, pauses, retakes."""
    retakes = retake_ids(ct.segments)
    lines = [f"## clip {ct.clip_id} ({ct.duration:.1f}s)"]
    prev_end = 0.0
    for seg in ct.segments:
        if seg.start - prev_end >= min_pause:
            lines.append(f"-- pause {seg.start - prev_end:.1f}s --")
        unclear = sum(1 for w in seg.words if w.prob < 0.5)
        lines.append(_seg_line(seg, seg.id in retakes) + (f" (unclear {unclear})" if unclear else ""))
        prev_end = seg.end
    return "\n".join(lines)


def chunk_transcript(ct: ClipTranscript, seconds: float = CHUNK_SECONDS) -> list[Chunk]:
    """Group a clip's segments into ~``seconds`` windows on segment boundaries."""
    retakes = retake_ids(ct.segments)
    groups: list[list[Segment]] = []
    for seg in ct.segments:
        if groups and seg.end - groups[-1][0].start <= seconds:
            groups[-1].append(seg)
        else:
            groups.append([seg])
    return [
        Chunk(f"{ct.clip_id}#{i}", ct.clip_id, g[0].start, g[-1].end,
              [_seg_line(s, s.id in retakes) for s in g])
        for i, g in enumerate(groups)
    ]


def broll_catalog(manifest: ClipsManifest) -> str:
    rows = [
        f"- {c.clip_id}: {c.duration:.1f}s {c.orientation}, motion {c.motion_score:.2f}, "
        f"{len(c.scenes)} scene(s)"
        for c in manifest.clips if c.kind == "broll"
    ]
    return "\n".join(rows) if rows else "(no B-roll clips)"


def _batches(chunks: list[Chunk], limit: int) -> list[list[Chunk]]:
    out, cur, size = [], [], 0
    for ch in chunks:
        n = len(ch.text)
        if cur and size + n > limit:
            out.append(cur)
            cur, size = [], 0
        cur.append(ch)
        size += n
    if cur:
        out.append(cur)
    return out


def map_chunks(provider: Provider, prompt: str, chunks: list[Chunk], workers: int = 4
               ) -> dict[str, ChunkScore]:
    def run(batch: list[Chunk]) -> MapResult:
        body = "\n\n".join(c.text for c in batch)
        return structured(provider, SYSTEM, (
            f"EDIT REQUEST: {prompt}\n\nScore every chunk below for this request. "
            f"Return one entry per chunk id.\n\n{body}"
        ), MapResult)

    scores: dict[str, ChunkScore] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(run, _batches(chunks, MAP_BATCH_CHARS)):
            for c in res.chunks:
                scores[c.chunk_id] = c
    return scores


def build_reduce_prompt(prompt: str, platform: str, target: float, material: str,
                        catalog: str) -> str:
    return (
        f"EDIT REQUEST: {prompt}\n"
        f"PLATFORM: {platform}\nTARGET DURATION: {target:.0f} seconds (stay within 5%)\n\n"
        "Pick transcript ranges for the edit in final playback order. Each selection is "
        "clip_id + first_segment..last_segment (inclusive segment ids from the same clip). "
        "Start and end on complete sentences. Durations come from the segment times shown. "
        "Lines marked (retake?) were restarted: keep the last complete take, skip the false starts. "
        "Optionally add B-roll cutaways from the catalog over selections (or clip_id null "
        "with a query when nothing fits).\n\n"
        f"MATERIAL:\n{material}\n\nB-ROLL CATALOG:\n{catalog}"
    )


# -- resolution -------------------------------------------------------------------
def resolve(result: ReduceResult, transcript: Transcript, manifest: ClipsManifest,
            prompt: str, platform: str, target: float) -> AssetPlan:
    segments: list[PlannedSegment] = []
    index_map: dict[int, int] = {}
    notes = [result.notes] if result.notes else []
    for i, sel in enumerate(result.selections):
        ct, clip = transcript.for_clip(sel.clip_id), manifest.get(sel.clip_id)
        if ct is None or clip is None:
            notes.append(f"dropped selection {i}: unknown clip {sel.clip_id}")
            continue
        by_id = {s.id: s for s in ct.segments}
        a, b = sorted((sel.first_segment, sel.last_segment))
        if a not in by_id or b not in by_id:
            notes.append(f"dropped selection {i}: segment ids {a}-{b} not in {sel.clip_id}")
            continue
        index_map[i] = len(segments)
        segments.append(PlannedSegment(
            clip_id=clip.clip_id, source_file=clip.source_file,
            start=by_id[a].start, end=by_id[b].end, first_segment_id=a, last_segment_id=b,
            purpose=sel.purpose, priority=min(3, max(1, sel.priority)),
        ))
    broll = []
    for pick in result.broll:
        if pick.over_selection not in index_map:
            continue
        clip = manifest.get(pick.clip_id) if pick.clip_id else None
        if clip is not None and clip.kind != "broll":
            clip = None
        broll.append(BrollPlacement(
            over_segment=index_map[pick.over_selection], duration=max(0.5, pick.seconds),
            query=pick.query, clip_id=clip.clip_id if clip else None,
            source_file=clip.source_file if clip else None,
            src_start=_best_broll_start(clip) if clip else 0.0,
        ))
    return AssetPlan(prompt=prompt, target_platform=platform, target_duration=target,
                     segments=segments, broll=broll, notes="; ".join(notes))


def _best_broll_start(clip: ClipInfo) -> float:
    """Start of the longest scene: usually the steadiest usable shot."""
    if not clip.scenes:
        return 0.0
    best = max(clip.scenes, key=lambda s: s.end - s.start)
    return round(best.start, 3)


def montage_plan(manifest: ClipsManifest, prompt: str, platform: str, target: float) -> AssetPlan:
    """No speech anywhere: deterministic B-roll montage, most dynamic clips first."""
    clips = sorted((c for c in manifest.clips), key=lambda c: -c.motion_score)
    per = max(1.5, min(4.0, target / max(1, len(clips))))
    segs, total = [], 0.0
    for c in clips * 8:
        if total >= target:
            break
        start = _best_broll_start(c)
        dur = min(per, c.duration - start, target - total)
        if dur < 0.5:
            continue
        segs.append(PlannedSegment(clip_id=c.clip_id, source_file=c.source_file, start=start,
                                   end=round(start + dur, 3), purpose="montage", priority=2))
        total += dur
    return AssetPlan(prompt=prompt, target_platform=platform, target_duration=target,
                     segments=segs, notes="montage: no speech detected in any clip")


def build_edl(plan: AssetPlan, manifest: ClipsManifest) -> EDL:
    spec = get_platform_specs(plan.target_platform)
    width, height = map(int, spec["resolution"].split("x"))
    clips = [
        EDLClip(clip_id=f"c{i + 1:03d}", source_clip_id=s.clip_id, source_file=s.source_file,
                start=s.start, end=s.end, priority=s.priority, notes=s.purpose)
        for i, s in enumerate(plan.segments)
    ]
    overlays = []
    for j, b in enumerate(plan.broll):
        if b.clip_id is None or b.source_file is None or b.over_segment >= len(clips):
            continue
        anchor = clips[b.over_segment]
        offset = min(0.5, anchor.duration / 4)
        overlays.append(Overlay(
            overlay_id=f"o{j + 1:03d}", source_clip_id=b.clip_id, source_file=b.source_file,
            src_start=b.src_start, anchor_clip_id=anchor.clip_id, offset=round(offset, 3),
            duration=round(min(b.duration, anchor.duration - offset), 3),
        ))
    return EDL(target_platform=plan.target_platform, width=width, height=height,
               target_duration=plan.target_duration, clips=clips, overlays=overlays)


# -- stage entry ---------------------------------------------------------------------
SPEECH_AIR = 1.15  # cut-audit keeps lead/tail air around every sentence


def choose_target(prompt: str, platform: str, explicit: float | None, available: float) -> float:
    """Requested length, capped by the platform and by what the footage can fill.

    ``available`` is pure speech time; a cut that keeps all of it is ~15% longer
    once sentences get their lead/tail air, so the cap allows for that instead of
    making cut-audit drop clips from a "use everything" edit.
    """
    spec = get_platform_specs(platform)
    target = explicit or duration_from_prompt(prompt) or float(spec["typical_duration"])
    return round(min(target, float(spec["max_duration"]), available * SPEECH_AIR), 3)


def plan(ws: Workspace, prompt: str, provider: Provider, *, target_platform: str = "youtube",
         target_duration: float | None = None) -> tuple[AssetPlan, EDL]:
    manifest = Workspace.read(ws.clips_manifest, ClipsManifest)
    transcript = Workspace.read(ws.transcript, Transcript)
    talking = [transcript.for_clip(c.clip_id) for c in manifest.clips if c.kind == "talking_head"]
    talking = [t for t in talking if t is not None and t.segments]
    available = sum(s.end - s.start for t in talking for s in t.segments) or manifest.total_duration
    target = choose_target(prompt, target_platform, target_duration, available)

    if not talking:
        asset_plan = montage_plan(manifest, prompt, target_platform, target)
    else:
        chunks = [ch for t in talking for ch in chunk_transcript(t)]
        full = "\n\n".join(ch.text for ch in chunks)
        if len(full) <= DIRECT_LIMIT_CHARS:
            material = full
        else:
            scores = map_chunks(provider, prompt, chunks)
            material = "\n\n".join(_summarised(ch, scores.get(ch.chunk_id)) for ch in chunks)
        result = structured(provider, SYSTEM, build_reduce_prompt(
            prompt, target_platform, target, material, broll_catalog(manifest)), ReduceResult)
        asset_plan = resolve(result, transcript, manifest, prompt, target_platform, target)
        if not asset_plan.segments:
            raise RuntimeError(f"Planner returned no usable selections: {asset_plan.notes}")

    edl = build_edl(asset_plan, manifest)
    Workspace.write(ws.asset_plan, asset_plan)
    Workspace.write(ws.edl_raw, edl)
    ws.record("plan", ws.stage_key("plan", prompt, target_platform, target, provider.name),
              [ws.asset_plan, ws.edl_raw])
    return asset_plan, edl


def _summarised(ch: Chunk, score: ChunkScore | None) -> str:
    head = f"## chunk {ch.chunk_id} (clip {ch.clip_id}, {ch.start:.1f}-{ch.end:.1f}s)"
    if score is None:
        return head + "\n(unscored)\n" + "\n".join(ch.lines[:3])
    keep = set(score.best_segments)
    lines = [ln for ln in ch.lines if int(ln[1:ln.index("]")]) in keep]
    return f"{head} score {score.score}/10: {score.summary}\n" + "\n".join(lines)
