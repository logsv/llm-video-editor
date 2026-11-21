"""Stage 2b - ``b-roll-select``: a persistent local B-roll index and matching.

Index (kept in the library folder, rescanned incrementally by path/mtime/size):
  <library>/.llmve_broll_index.json   machine-readable catalogue
  <library>/broll_index.md            human/agent-readable table
  <library>/.llmve_broll/thumbs/      one mid-clip thumbnail per item
  <library>/.llmve_broll/sheet_*.png  labelled thumbnail contact sheets

Descriptions come from file/folder names, optional sidecars (``clip.txt`` or
``clip.json`` with ``description``/``tags``), or ``describe()`` (e.g. written
by an agent after looking at the thumbnail sheet). Pixels never go to the
planner: matching is BM25 over text, with an optional text-only LLM re-rank.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path

from pydantic import Field

from ..ffmpeg.probe import probe
from ..ffmpeg.runner import ffmpeg
from ..ffmpeg.scenes import analyze_scenes
from ..llm import Provider, structured
from ..schemas import (
    AssetPlan,
    BrollIndex,
    BrollItem,
    BrollPlacement,
    BrollSelection,
    Candidate,
    ClipInfo,
    ClipsManifest,
    SceneCut,
    SlotChoice,
    Strict,
    Transcript,
)
from ..workspace import SourceFingerprint, Workspace, resolve_inputs
from .plan import build_edl

INDEX_NAME = ".llmve_broll_index.json"
MD_NAME = "broll_index.md"
DATA_DIR = ".llmve_broll"

_CAMERA_JUNK = re.compile(
    r"^(c\d+|dji|img|mvi|dsc|dscf|gopr|gh\d+|gx\d+|pxl|vid|clip|final|take\d*|v\d+|"
    r"copy|edit|export|raw|mp4|mov|mkv|\d+|[a-z]\d{3,})$")
_STOP = set(["a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with", "from", "by", "is", "are", "be", "this", "that", "it", "as", "we", "you", "i", "our", "your", "they", "them", "their", "was", "were", "will", "just", "so", "very", "can", "into", "about", "over", "under", "then", "than", "there", "here", "what", "when", "where", "how", "why", "who", "all", "any", "some", "more", "most", "other", "such", "no", "not", "only", "own", "same", "too", "s", "t", "don", "now", "get", "got", "let", "lets", "like", "really", "thing", "things"])


# -- text helpers -------------------------------------------------------------------
def _split_name(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [t.lower() for t in re.split(r"[^A-Za-z0-9]+", text) if t]


def filename_tags(rel_path: str) -> list[str]:
    parts = Path(rel_path).with_suffix("").parts
    tags: list[str] = []
    for part in parts:
        for tok in _split_name(part):
            if len(tok) > 2 and not _CAMERA_JUNK.match(tok) and tok not in tags:
                tags.append(tok)
    return tags


def _stem(tok: str) -> str:
    for suf in ("ing", "es", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in _STOP and len(t) > 2]


_MEDIA_TOKEN = re.compile(r"\S*\.(mp4|mov|mkv|avi|m4v|webm|mxf|wav|mp3)\b", re.IGNORECASE)


def _clean(text: str) -> str:
    return _MEDIA_TOKEN.sub("", text).strip()


# -- index ------------------------------------------------------------------------------
def _sidecar(path: Path) -> tuple[str, list[str]] | None:
    js, txt = path.with_suffix(".json"), path.with_suffix(".txt")
    if js.exists():
        data = json.loads(js.read_text())
        return _clean(str(data.get("description", ""))), [str(t).lower() for t in data.get("tags", [])]
    if txt.exists():
        return _clean(txt.read_text().strip()), []
    return None


def _thumb(src: Path, out: Path, t: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg("-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
           "-vf", "scale=320:180:force_original_aspect_ratio=decrease,"
                  "pad=320:180:(ow-iw)/2:(oh-ih)/2:color=black",
           "-q:v", "4", str(out), timeout=60)


def index_path(library: Path) -> Path:
    return Path(library) / INDEX_NAME


def load_index(library: Path) -> BrollIndex:
    p = index_path(library)
    if p.exists():
        return Workspace.read(p, BrollIndex)
    return BrollIndex(root=str(Path(library).resolve()))


def index_library(library: Path, *, rescan: bool = False) -> tuple[BrollIndex, int]:
    """(Re)index ``library``; returns the index and how many files were (re)scanned."""
    library = Path(library).resolve()
    old = {} if rescan else {i.rel_path: i for i in load_index(library).items}
    items, scanned = [], 0
    files = [f for f in resolve_inputs(library) if DATA_DIR not in f.parts]
    for f in files:
        rel = str(f.relative_to(library))
        st = f.stat()
        prev = old.get(rel)
        side = _sidecar(f)
        if prev and (prev.mtime_ns, prev.size) == (st.st_mtime_ns, st.st_size):
            if side and prev.described_by != "manual":
                prev = prev.model_copy(update={"description": side[0], "described_by": "sidecar",
                                               "tags": side[1] or prev.tags})
            items.append(prev)
            continue
        scanned += 1
        info = probe(f)
        sa = analyze_scenes(f, info.duration)
        item_id = SourceFingerprint.of(f).clip_id
        thumb = Path(DATA_DIR) / "thumbs" / f"{item_id}.jpg"
        _thumb(f, library / thumb, info.duration / 2)
        tags = filename_tags(rel)
        desc, by = " ".join(tags), "filename"
        if side:
            desc, by = side[0] or desc, "sidecar"
            tags = side[1] or tags
        if prev and prev.described_by == "manual":
            desc, tags, by = prev.description, prev.tags, "manual"
        orient = ("landscape" if info.width > info.height else
                  "portrait" if info.height > info.width else "square")
        items.append(BrollItem(
            item_id=item_id, rel_path=rel, mtime_ns=st.st_mtime_ns, size=st.st_size,
            duration=round(info.duration, 3), width=info.width, height=info.height, fps=info.fps,
            has_audio=info.has_audio, orientation=orient, motion_score=sa.motion_score,
            scenes=[SceneCut(start=a, end=b) for a, b in sa.scenes], thumbnail=str(thumb),
            description=desc, tags=tags, described_by=by))
    index = BrollIndex(root=str(library), updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       items=items)
    save_index(library, index)
    return index, scanned


def save_index(library: Path, index: BrollIndex) -> None:
    Workspace.write(index_path(library), index)
    (Path(library) / MD_NAME).write_text(render_markdown(index), encoding="utf-8")


def render_markdown(index: BrollIndex) -> str:
    lines = [
        "# B-roll index", "",
        f"{len(index.items)} clips, updated {index.updated_at}. Generated by `llmve broll index`; "
        "edit descriptions with `llmve broll describe`, not by hand.", "",
        "| id | file | duration | orientation | motion | description | tags | source |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i in index.items:
        lines.append(f"| {i.item_id} | {i.rel_path} | {i.duration:.1f}s | {i.orientation} | "
                     f"{i.motion_score:.2f} | {i.description.replace('|', '/')} | "
                     f"{', '.join(i.tags)} | {i.described_by} |")
    return "\n".join(lines) + "\n"


def describe(library: Path, item_id: str, description: str, tags: list[str] | None = None
             ) -> BrollItem:
    index = load_index(library)
    item = index.get(item_id)
    if item is None:
        raise KeyError(f"No item {item_id} in {index_path(library)}")
    updated = item.model_copy(update={"description": _clean(description),
                                      "tags": [t.lower() for t in (tags or item.tags)],
                                      "described_by": "manual"})
    index.items = [updated if i.item_id == item_id else i for i in index.items]
    save_index(library, index)
    return updated


def thumbnail_sheets(library: Path, cols: int = 5, rows: int = 4) -> list[Path]:
    """Labelled thumbnail sheets an agent can look at to write descriptions."""
    library = Path(library)
    index = load_index(library)
    work = library / DATA_DIR
    for old in [*work.glob("label_*.png"), *work.glob("sheet_*.png")]:
        old.unlink()
    for n, item in enumerate(index.items):
        ffmpeg("-i", str(library / item.thumbnail), "-vf",
               f"drawtext=text='{item.item_id}':x=6:y=h-th-6:fontsize=18:fontcolor=yellow:"
               "box=1:boxcolor=black@0.7:boxborderw=4", str(work / f"label_{n:03d}.png"),
               timeout=60)
    if not index.items:
        return []
    ffmpeg("-framerate", "1", "-start_number", "0", "-i", str(work / "label_%03d.png"),
           "-vf", f"tile={cols}x{rows}:padding=4:margin=4", "-start_number", "0",
           str(work / "sheet_%03d.png"), timeout=120)
    for old in work.glob("label_*.png"):
        old.unlink()
    return sorted(work.glob("sheet_*.png"))


# -- matching -------------------------------------------------------------------------
class _Doc:
    def __init__(self, item_id: str, text: str, duration: float, orientation: str,
                 scenes: list[SceneCut], source: Path | None, info: ClipInfo | None = None):
        self.item_id, self.duration, self.orientation = item_id, duration, orientation
        self.scenes, self.source, self.info = scenes, source, info
        self.tokens = tokenize(text)


class _BM25:
    def __init__(self, docs: list[_Doc], k1: float = 1.4, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avg = sum(len(d.tokens) for d in docs) / max(1, len(docs)) or 1.0
        df = Counter(t for d in docs for t in set(d.tokens))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query: list[str], doc: _Doc) -> float:
        tf = Counter(doc.tokens)
        s = 0.0
        for q in set(query):
            if q not in tf:
                continue
            f = tf[q]
            s += self.idf[q] * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * len(doc.tokens) / self.avg))
        return s


class _Choice(Strict):
    slot: int
    item_id: str | None = Field(description="one of the slot's candidate ids, or null")
    reason: str = Field(description="<= 12 words")


class _Choices(Strict):
    choices: list[_Choice]


RERANK_SYSTEM = ("You match B-roll cutaways to what is being said in a video edit. You see "
                 "only text descriptions. Pick the candidate that best illustrates each slot, "
                 "or null when none fits. Answer in the JSON schema.")


def _auto_slots(plan: AssetPlan, transcript: Transcript) -> list[BrollPlacement]:
    """One cutaway per ~20 s of programme on the longest segments (>= 4 s)."""
    want = max(1, math.ceil((plan.target_duration or 60) / 20))
    ranked = sorted(range(len(plan.segments)), key=lambda i: -plan.segments[i].duration)
    slots = []
    for i in ranked[:want]:
        seg = plan.segments[i]
        if seg.duration < 4:
            continue
        ct = transcript.for_clip(seg.clip_id)
        text = " ".join(s.text for s in (ct.segments if ct else [])
                        if s.start < seg.end and s.end > seg.start)
        slots.append(BrollPlacement(over_segment=i, offset=1.0,
                                    duration=round(min(3.0, seg.duration - 1.5), 3),
                                    query=text[:300] or seg.purpose))
    return slots


def _best_start(scenes: list[SceneCut]) -> float:
    return round(max(scenes, key=lambda s: s.end - s.start).start, 3) if scenes else 0.0


def match_broll(ws: Workspace, library: Path | None = None, *, provider: Provider | None = None,
                top_k: int = 5, min_score: float = 0.5, auto: bool = True,
                generator=None) -> BrollSelection:
    """Fill B-roll slots in asset_plan.json and rebuild edl_raw.json."""
    plan = Workspace.read(ws.asset_plan, AssetPlan)
    manifest = Workspace.read(ws.clips_manifest, ClipsManifest)
    transcript = Workspace.read(ws.transcript, Transcript)
    edl_orientation = "portrait" if plan.target_platform in ("reels", "tiktok") else "landscape"

    docs: list[_Doc] = []
    index = load_index(library) if library else None
    for item in (index.items if index else []):
        docs.append(_Doc(item.item_id, f"{item.description} {' '.join(item.tags * 2)}",
                         item.duration, item.orientation, item.scenes,
                         Path(index.root) / item.rel_path))  # type: ignore[union-attr]
    known = {d.item_id for d in docs}
    for c in manifest.clips:  # B-roll shot during the same session is a candidate too
        if c.kind == "broll" and c.clip_id not in known:
            docs.append(_Doc(c.clip_id, " ".join(filename_tags(Path(c.source_file).name)),
                             c.duration, c.orientation, c.scenes, None, c))
    bm25 = _BM25(docs)

    placements = list(plan.broll)
    if auto and not placements:
        placements = _auto_slots(plan, transcript)
    slots: list[SlotChoice] = []
    used: set[str] = set()
    for n, pl in enumerate(placements):
        if pl.clip_id:  # already resolved by the planner from the session's own B-roll
            slots.append(SlotChoice(slot=n, over_segment=pl.over_segment, query=pl.query,
                                    duration=pl.duration, chosen=pl.clip_id, reason="planner"))
            used.add(pl.clip_id)
            continue
        q = tokenize(pl.query)
        scored = []
        for d in docs:
            if d.duration < 1.0:
                continue
            s = bm25.score(q, d)
            s += 0.3 if d.orientation == edl_orientation else 0.0
            s -= 0.5 if d.duration < pl.duration else 0.0
            s -= 1.0 if d.item_id in used else 0.0
            scored.append(Candidate(item_id=d.item_id, score=round(s, 3)))
        scored.sort(key=lambda c: -c.score)
        top = scored[:top_k]
        chosen = top[0].item_id if top and top[0].score >= min_score else None
        slots.append(SlotChoice(slot=n, over_segment=pl.over_segment, query=pl.query,
                                duration=pl.duration, candidates=top, chosen=chosen,
                                reason="bm25" if chosen else "no candidate above threshold"))
        if chosen:
            used.add(chosen)

    by_id = {d.item_id: d for d in docs}
    if provider is not None and any(s.candidates for s in slots):
        lines = []
        for s in slots:
            if not s.candidates:
                continue
            lines.append(f"## slot {s.slot}: {_clean(s.query)[:300]}")
            for c in s.candidates:
                d = by_id[c.item_id]
                desc = " ".join(d.tokens[:40])
                lines.append(f"- {c.item_id} ({d.duration:.1f}s {d.orientation}): {desc}")
        res = structured(provider, RERANK_SYSTEM, "\n".join(lines), _Choices)
        for ch in res.choices:
            slot = next((s for s in slots if s.slot == ch.slot), None)
            if slot is None:
                continue
            ids = {c.item_id for c in slot.candidates}
            if ch.item_id is None or ch.item_id in ids:
                slot.chosen, slot.reason = ch.item_id, f"llm: {ch.reason}"

    if generator is not None:
        for s in slots:
            if s.chosen is None:
                path = generator(s.query, s.duration, edl_orientation)
                info = probe(path)
                fp = SourceFingerprint.of(Path(path))
                by_id[fp.clip_id] = _Doc(fp.clip_id, s.query, info.duration,
                                         "portrait" if info.height > info.width else "landscape",
                                         [], Path(path))
                s.chosen, s.generated, s.reason = fp.clip_id, True, "generated"

    # register library items in the manifest so cut-audit/render can resolve them
    resolved: list[BrollPlacement] = []
    for s, pl in zip(slots, placements, strict=True):
        if s.chosen is None:
            resolved.append(pl.model_copy(update={"clip_id": None, "source_file": None}))
            continue
        info = manifest.get(s.chosen)
        if info is None:
            d = by_id[s.chosen]
            assert d.source is not None
            p = probe(d.source)
            info = ClipInfo(clip_id=s.chosen, source_file=str(d.source.resolve()),
                            duration=round(p.duration, 3), width=p.width, height=p.height,
                            fps=p.fps, has_audio=p.has_audio, video_codec=p.video_codec,
                            audio_codec=p.audio_codec, kind="broll", scenes=d.scenes)
            manifest.clips.append(info)
        start = _best_start(info.scenes)
        resolved.append(pl.model_copy(update={
            "clip_id": info.clip_id, "source_file": info.source_file, "src_start": start,
            "duration": round(min(pl.duration, info.duration - start), 3)}))

    plan = plan.model_copy(update={"broll": resolved})
    selection = BrollSelection(library=str(library) if library else None,
                               reranked_by=provider.name if provider else None, slots=slots)
    Workspace.write(ws.clips_manifest, manifest)
    Workspace.write(ws.asset_plan, plan)
    Workspace.write(ws.edl_raw, build_edl(plan, manifest))
    Workspace.write(ws.broll_select, selection)
    ws.record("broll_select", ws.stage_key("broll", plan.model_dump(), str(library)),
              [ws.broll_select, ws.asset_plan, ws.edl_raw])
    return selection
