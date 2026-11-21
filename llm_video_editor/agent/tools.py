"""The editor agent's instruments: typed tools over the existing tested stages.

Tools execute and report; they never make editorial choices. Groups:

* project  - open footage (optionally only the first N clips), analyse it
* look     - overview, transcript, speech map, frames, edit state, history
* edit     - atomic operations on the versioned edit (each with a reason)
* check    - audit (report only), review sheet, render, publish, media search

Every tool takes a pydantic params model and returns a JSON-safe dict. Errors are
returned as ``{"error": "..."}`` with a message the agent can act on. The same
registry is served over MCP (``llmve mcp``) and can be called from Python.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..schemas import EDL, ClipsManifest, Transcript
from ..workspace import Workspace, check_output_path
from . import jobs, ops
from .edits import EditConflict, EditError, EditStore
from .footage import group_recordings, inspect, select_inputs

Privacy = Literal["stills", "text-only"]


class P(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WsP(P):
    workspace: str = Field(description="workspace directory returned by open_project")


class EditP(WsP):
    reason: str = Field(min_length=3, description="why: recorded in the decision log")
    expected_version: int | None = Field(
        None, description="version you based this on; rejected if the edit moved on")


@dataclass
class Context:
    """Per-server settings and injectable dependencies (tests swap the transcriber)."""

    privacy: Privacy = "stills"
    home: Path | None = None
    transcriber_factory: Callable[[str], Any] | None = None
    images_out: list[str] = field(default_factory=list)  # images the last call returned
    # long tools (analyze, review, render): "off" runs them in the call (terminal use);
    # "process"/"thread" run them as background jobs polled with job_status (MCP)
    jobs: Literal["off", "thread", "process"] = "off"
    job_wait: float = 20.0  # seconds a long call waits before returning a job id


@dataclass(frozen=True)
class Tool:
    name: str
    kind: Literal["project", "look", "edit", "check"]
    description: str
    params: type[BaseModel]
    run: Callable[[Context, Any], dict]

    def schema(self) -> dict:
        return self.params.model_json_schema()


REGISTRY: dict[str, Tool] = {}


def tool(name: str, kind: str, params: type[BaseModel]):
    def deco(fn: Callable[[Context, Any], dict]) -> Callable[[Context, Any], dict]:
        REGISTRY[name] = Tool(name, kind, (fn.__doc__ or "").strip(), params, fn)  # type: ignore[arg-type]
        return fn
    return deco


def call(name: str, args: dict[str, Any], ctx: Context | None = None) -> dict:
    """Validate ``args`` and run tool ``name``; errors come back as {"error": ...}."""
    ctx = ctx or Context()
    ctx.images_out = []
    t = REGISTRY.get(name)
    if t is None:
        return {"error": f"unknown tool {name!r}; tools: {', '.join(sorted(REGISTRY))}"}
    try:
        params = t.params.model_validate(args)
    except ValidationError as e:
        err = e.errors()[0]
        return {"error": f"bad arguments for {name}: {err['msg']} at "
                         f"{'.'.join(map(str, err['loc'])) or '(root)'}"}
    try:
        if ctx.jobs != "off" and name in jobs.LONG_TOOLS:
            ws = _ws(params.workspace)  # type: ignore[attr-defined]
            job = jobs.start(name, params.model_dump(mode="json"), ctx, ws.root, ctx.job_wait)
            ctx.images_out = job.get("images", []) if job["status"] == "done" else []
            return jobs.summary(job)
        return t.run(ctx, params)
    except (EditError, EditConflict, ValueError, FileNotFoundError, KeyError) as e:
        return {"error": str(e).strip("'\"")}


# -- shared helpers -------------------------------------------------------------------------------
def _ws(path: str) -> Workspace:
    root = Path(path).expanduser()
    if not (root / "manifest.json").exists() and not (root / "sources.json").exists():
        raise FileNotFoundError(f"{root} is not a workspace: call open_project first")
    return Workspace.load(root)


def _manifest(ws: Workspace) -> ClipsManifest:
    if not ws.clips_manifest.exists():
        raise FileNotFoundError("footage not analysed yet: call analyze first")
    return Workspace.read(ws.clips_manifest, ClipsManifest)


def _footage(ws: Workspace) -> dict[str, dict]:
    p = ws.root / "footage.json"
    return json.loads(p.read_text()) if p.exists() else {}


# =============================================================================== project
class OpenP(P):
    inputs: list[str] = Field(description="video files, folders or globs")
    first: int | None = Field(None, ge=1, description="only the first N clips (testing)")
    order: Literal["name", "time"] = Field("name", description="order for `first`")


@tool("open_project", "project", OpenP)
def open_project(ctx: Context, p: OpenP) -> dict:
    """Open footage as a project: inspects every file (HDR, frame rate, audio, split
    recordings, device, time, GPS) without changing anything. Returns the workspace."""
    paths = select_inputs(p.inputs, first=p.first, order=p.order)
    infos = [inspect(x) for x in paths]
    usable = [Path(i.path) for i in infos if i.ok]
    if not usable:
        raise ValueError("no readable video files: " + "; ".join(f"{i.name}: {i.error}"
                                                                 for i in infos))
    ws = Workspace.open([str(x) for x in usable], home=ctx.home)
    groups = group_recordings([i for i in infos if i.ok])
    facts = {i.path: i.as_dict() for i in infos}
    (ws.root / "footage.json").write_text(json.dumps(facts, indent=2))
    return {"workspace": str(ws.root), "clips": len(usable), "privacy": ctx.privacy,
            "skipped": [{"name": i.name, "reason": i.error} for i in infos if not i.ok],
            "split_recordings": groups,
            "files": [{"name": i.name, "ok": i.ok, "duration": i.duration,
                       "size": f"{i.width}x{i.height}", "fps": i.fps, "issues": i.issues}
                      for i in infos],
            "next": "call analyze to transcribe and score scenes"}


class AnalyzeP(WsP):
    asr_model: str = Field("distil-large-v3", description="faster-whisper model; "
                           "large-v3-turbo for Hindi or other languages")
    language: str | None = Field(None, description="force a language, e.g. hi, en")
    chunk_over_minutes: float | None = Field(10.0, description="parallel ASR above this length")


@tool("analyze", "project", AnalyzeP)
def analyze(ctx: Context, p: AnalyzeP) -> dict:
    """Transcribe (word timings), score scenes and motion, build overview contact sheets."""
    from ..stages.asr import FasterWhisperTranscriber
    from ..stages.chunked_asr import ChunkConfig
    from ..stages.watch import watch

    ws = _ws(p.workspace)
    factory = ctx.transcriber_factory or FasterWhisperTranscriber
    chunk = ChunkConfig(min_duration=p.chunk_over_minutes * 60) if p.chunk_over_minutes else None
    res = watch(ws, transcriber=factory(p.asr_model), language=p.language, chunking=chunk)
    return {"clips": [{"clip_id": c.clip_id, "file": Path(c.source_file).name,
                       "duration": c.duration, "kind": c.kind,
                       "speech_seconds": c.speech_seconds, "scenes": len(c.scenes),
                       "motion": c.motion_score, "sheets": c.overview_sheets}
                      for c in res.manifest.clips],
            "cached": res.cached}


# =============================================================================== look
@tool("overview", "look", WsP)
def overview(ctx: Context, p: WsP) -> dict:
    """All clips with duration, speech, scenes, motion, device, time, GPS and issues."""
    ws = _ws(p.workspace)
    facts = _footage(ws)
    out = []
    for c in _manifest(ws).clips:
        f = facts.get(c.source_file, {})
        out.append({"clip_id": c.clip_id, "file": Path(c.source_file).name,
                    "duration": c.duration, "size": f"{c.width}x{c.height}", "fps": c.fps,
                    "kind": c.kind, "speech_ratio": c.speech_ratio,
                    "scenes": [[round(s.start, 2), round(s.end, 2)] for s in c.scenes][:40],
                    "motion": c.motion_score, "recorded_at": f.get("recorded_at"),
                    "device": f.get("device"), "gps": [f.get("latitude"), f.get("longitude")]
                    if f.get("latitude") is not None else None,
                    "hdr": f.get("hdr"), "issues": f.get("issues", [])})
    return {"clips": out, "total_seconds": round(sum(c["duration"] for c in out), 1)}


class TranscriptP(WsP):
    clip_id: str | None = None
    start: float | None = None
    end: float | None = None
    max_lines: int = Field(400, ge=1, le=5000)


@tool("transcript", "look", TranscriptP)
def transcript(ctx: Context, p: TranscriptP) -> dict:
    """Transcript segments (ids, source times, text, retake and low-confidence hints)."""
    from ..stages.plan import retake_ids

    ws = _ws(p.workspace)
    tr = Workspace.read(ws.transcript, Transcript)
    lines = []
    for ct in tr.clips:
        if p.clip_id and ct.clip_id != p.clip_id:
            continue
        retakes = retake_ids(ct.segments)
        for s in ct.segments:
            if (p.start is not None and s.end < p.start) or (p.end is not None and s.start > p.end):
                continue
            lines.append({"clip_id": ct.clip_id, "id": s.id, "start": s.start, "end": s.end,
                          "text": s.text, "retake": s.id in retakes,
                          "unclear_words": sum(w.prob < 0.5 for w in s.words)})
    return {"segments": lines[:p.max_lines], "truncated": len(lines) > p.max_lines,
            "language": {c.clip_id: c.language for c in tr.clips}}


class SpeechMapP(WsP):
    clip_id: str
    start: float = 0.0
    end: float | None = None
    min_pause: float = Field(0.3, description="report pauses at least this long (s)")


@tool("speech_map", "look", SpeechMapP)
def speech_map(ctx: Context, p: SpeechMapP) -> dict:
    """Words with times and the pauses between them, for precise cut points."""
    ws = _ws(p.workspace)
    ct = Workspace.read(ws.transcript, Transcript).for_clip(p.clip_id)
    if ct is None:
        raise KeyError(f"no transcript for clip {p.clip_id}")
    end = p.end if p.end is not None else ct.duration
    words = [w for w in ct.words() if w.end > p.start and w.start < end]
    pauses, prev = [], p.start
    for w in words:
        if w.start - prev >= p.min_pause:
            pauses.append({"start": round(prev, 3), "end": round(w.start, 3),
                           "seconds": round(w.start - prev, 3)})
        prev = w.end
    if end - prev >= p.min_pause:
        pauses.append({"start": round(prev, 3), "end": round(end, 3),
                       "seconds": round(end - prev, 3)})
    return {"words": [{"start": w.start, "end": w.end, "text": w.text, "prob": w.prob}
                      for w in words][:3000],
            "pauses": pauses}


class FramesP(WsP):
    clip_id: str
    start: float
    end: float
    fps: float = Field(1.0, gt=0, le=4)


@tool("frames", "look", FramesP)
def frames(ctx: Context, p: FramesP) -> dict:
    """Contact sheets of a clip window (images with timestamps burned in). Disabled in
    text-only privacy mode. Keep windows short: 30 s at 1 fps is two sheets."""
    from ..ffmpeg.sheets import SheetSpec
    from ..stages.watch import watch_window

    if ctx.privacy == "text-only":
        raise ValueError("privacy is text-only: no images go to the model; use overview, "
                         "transcript and measurements instead")
    if p.end - p.start > 120:
        raise ValueError("window too long (max 120 s): look at shorter windows")
    ws = _ws(p.workspace)
    sheets = watch_window(ws, p.clip_id, p.start, p.end, spec=SheetSpec(fps=p.fps))
    paths = [str(ws.root / s.path) for s in sheets]
    ctx.images_out = paths
    return {"sheets": [{"path": path, "start": s.start, "end": s.end, "tokens": s.est_tokens}
                       for path, s in zip(paths, sheets, strict=True)]}


@tool("edit_state", "look", WsP)
def edit_state(ctx: Context, p: WsP) -> dict:
    """The current edit: version, output settings, clips with output times, and items."""
    ws = _ws(p.workspace)
    store = EditStore(ws)
    try:
        edl = store.current()
    except EditError:
        return {"version": 0, "edit": None, "next": "call new_edit, then add_clip"}
    starts = edl.output_starts()
    return {
        "version": store.head(), "platform": edl.target_platform,
        "size": f"{edl.width}x{edl.height}", "target_duration": edl.target_duration,
        "total_duration": round(edl.total_duration, 3), "fit_mode": edl.fit_mode,
        "clips": [{"clip_id": c.clip_id, "source": c.source_clip_id, "start": c.start,
                   "end": c.end, "duration": round(c.duration, 3), "out_start": round(t, 3),
                   "role": c.role, "zoom": c.zoom, "crop_center": c.crop_center,
                   "chapter": c.chapter, "notes": c.notes}
                  for c, t in zip(edl.clips, starts, strict=True)],
        "pacing": edl.pacing.model_dump(), "captions": edl.captions.model_dump(),
        "mix": edl.mix.model_dump(),
        "overlays": [o.model_dump() for o in edl.overlays],
        "graphics": [g.model_dump() for g in edl.graphics],
        "music": [m.model_dump() for m in edl.music],
    }


class HistoryP(WsP):
    limit: int = Field(30, ge=1, le=500)
    kind: Literal["edit", "check"] | None = None


@tool("history", "look", HistoryP)
def history(ctx: Context, p: HistoryP) -> dict:
    """Decision log: tool calls with reasons and versions (newest last)."""
    return {"entries": EditStore(_ws(p.workspace)).history(p.limit, p.kind)}


# =============================================================================== edit
def _edit(p: EditP, op: Callable[..., EDL], params: dict, name: str,
          need_manifest: bool = True) -> dict:
    ws = _ws(p.workspace)
    store = EditStore(ws)
    manifest = _manifest(ws) if need_manifest else None
    ch = store.apply(name, params, p.reason, lambda e: op(e, manifest, **params),
                     expected_version=p.expected_version)
    return {"version": ch.version, "changed": ch.changed, "summary": ch.summary}


class NewEditP(EditP):
    target_platform: str = Field(description="youtube | shorts | reels | tiktok | twitter | linkedin")
    target_duration: float | None = Field(None, description="seconds, or none")
    replace: bool = Field(False, description="start over even if an edit exists")


@tool("new_edit", "edit", NewEditP)
def new_edit(ctx: Context, p: NewEditP) -> dict:
    """Start an empty edit for a platform (frame size from the platform)."""
    ws = _ws(p.workspace)
    store = EditStore(ws)
    if store.head() and not p.replace:
        raise EditError("an edit already exists: set replace=true to start over "
                        "(the old versions stay in the history)")
    edl = ops.new_edit(None, _manifest(ws), target_platform=p.target_platform,
                       target_duration=p.target_duration)
    if store.head() == 0:
        Workspace.write(ws.edl_raw, edl)
        store.current()  # imports as version 1
        return {"version": store.head(), "changed": True, "summary": "new empty edit"}
    ch = store.apply("new_edit", {"target_platform": p.target_platform}, p.reason, lambda _: edl,
                     expected_version=p.expected_version)
    return {"version": ch.version, "changed": ch.changed, "summary": "new empty edit"}


class AddClipP(EditP):
    source_clip_id: str
    start: float = Field(description="source seconds")
    end: float = Field(description="source seconds")
    position: int | None = Field(None, description="index in the timeline (0 = first)")
    after: str | None = Field(None, description="insert after this clip id")
    role: Literal["a_roll", "broll"] = "a_roll"
    chapter: str | None = None
    zoom: float = 1.0
    crop_center: tuple[float, float] | None = None
    notes: str = ""


def _fields(p: BaseModel, skip: tuple[str, ...] = ("workspace", "reason",
                                                     "expected_version")) -> dict:
    return {k: v for k, v in p.model_dump().items() if k not in skip}


@tool("add_clip", "edit", AddClipP)
def add_clip(ctx: Context, p: AddClipP) -> dict:
    """Add a piece of a source clip to the timeline."""
    return _edit(p, ops.add_clip, _fields(p), "add_clip")


class ClipIdP(EditP):
    clip_id: str


@tool("remove_clip", "edit", ClipIdP)
def remove_clip(ctx: Context, p: ClipIdP) -> dict:
    """Remove a clip from the timeline."""
    return _edit(p, ops.remove_clip, _fields(p), "remove_clip")


class TrimP(ClipIdP):
    start: float | None = None
    end: float | None = None


@tool("trim_clip", "edit", TrimP)
def trim_clip(ctx: Context, p: TrimP) -> dict:
    """Move a clip's in and/or out point (source seconds)."""
    return _edit(p, ops.trim_clip, _fields(p), "trim_clip")


class SplitP(ClipIdP):
    at: float = Field(description="source seconds inside the clip")


@tool("split_clip", "edit", SplitP)
def split_clip(ctx: Context, p: SplitP) -> dict:
    """Split a clip in two at a source time."""
    return _edit(p, ops.split_clip, _fields(p), "split_clip")


class MoveP(ClipIdP):
    position: int | None = None
    after: str | None = None


@tool("move_clip", "edit", MoveP)
def move_clip(ctx: Context, p: MoveP) -> dict:
    """Reorder: move a clip to a position or after another clip."""
    return _edit(p, ops.move_clip, _fields(p), "move_clip")


class SetClipP(ClipIdP):
    fields: dict[str, Any] = Field(description="zoom, crop_center, chapter (\"\" clears), "
                                   "role, priority, notes")


@tool("set_clip", "edit", SetClipP)
def set_clip(ctx: Context, p: SetClipP) -> dict:
    """Change a clip's framing, chapter, role or notes."""
    return _edit(p, ops.set_clip, {"clip_id": p.clip_id, **p.fields}, "set_clip")


class FieldsP(EditP):
    fields: dict[str, Any]


@tool("set_pacing", "edit", FieldsP)
def set_pacing(ctx: Context, p: FieldsP) -> dict:
    """Jump-cut settings: tighten, max_gap, lead, tail, min_island, drop_fillers,
    punch_in (off | alternate), punch_zoom."""
    return _edit(p, ops.set_pacing, p.fields, "set_pacing")


@tool("set_captions", "edit", FieldsP)
def set_captions(ctx: Context, p: FieldsP) -> dict:
    """Captions: enabled, style (lines | word_highlight), max_words_per_line, position,
    replacements (ASR fixes)."""
    return _edit(p, ops.set_captions, p.fields, "set_captions")


@tool("set_mix", "edit", FieldsP)
def set_mix(ctx: Context, p: FieldsP) -> dict:
    """Ducking: duck_depth_db, attack_ms, release_ms."""
    return _edit(p, ops.set_mix, p.fields, "set_mix")


class OutputP(EditP):
    target_duration: float | None = None
    clear_target: bool = False
    fit_mode: Literal["cover", "contain"] | None = None


@tool("set_output", "edit", OutputP)
def set_output(ctx: Context, p: OutputP) -> dict:
    """Target length and fit mode (cover crops, contain letterboxes)."""
    return _edit(p, ops.set_output, _fields(p), "set_output")


class AddItemP(EditP):
    kind: Literal["overlays", "graphics", "music"]
    fields: dict[str, Any] = Field(description="overlay: source_clip_id, anchor_clip_id, "
                                   "offset, duration, src_start, zone; graphic: template or "
                                   "composition, variables, anchor_clip_id, offset, duration, "
                                   "box; music: file, role (music|sfx|voiceover), "
                                   "anchor_clip_id, offset, duration, volume_db, duck, loop")


@tool("add_item", "edit", AddItemP)
def add_item(ctx: Context, p: AddItemP) -> dict:
    """Add a B-roll overlay, a motion graphic, or music / sound effect / voice-over."""
    return _edit(p, ops.add_item, {"kind": p.kind, **p.fields}, "add_item")


class UpdateItemP(EditP):
    kind: Literal["overlays", "graphics", "music"]
    item_id: str
    fields: dict[str, Any]


@tool("update_item", "edit", UpdateItemP)
def update_item(ctx: Context, p: UpdateItemP) -> dict:
    """Change an overlay, graphic or music track."""
    return _edit(p, ops.update_item, {"kind": p.kind, "item_id": p.item_id, **p.fields},
                 "update_item")


class RemoveItemP(EditP):
    kind: Literal["overlays", "graphics", "music"]
    item_id: str


@tool("remove_item", "edit", RemoveItemP)
def remove_item(ctx: Context, p: RemoveItemP) -> dict:
    """Remove an overlay, graphic or music track."""
    return _edit(p, ops.remove_item, {"kind": p.kind, "item_id": p.item_id}, "remove_item")


_BATCHABLE = {"add_clip": ops.add_clip, "remove_clip": ops.remove_clip,
              "trim_clip": ops.trim_clip, "split_clip": ops.split_clip,
              "move_clip": ops.move_clip, "set_clip": ops.set_clip,
              "set_pacing": ops.set_pacing, "set_captions": ops.set_captions,
              "set_mix": ops.set_mix, "set_output": ops.set_output,
              "add_item": ops.add_item, "update_item": ops.update_item,
              "remove_item": ops.remove_item}


# per-call keys of the standalone tools that a batch sets once for all its operations
_BATCH_LEVEL = ("workspace", "reason", "expected_version")


def _op_params(op: str, params: dict[str, Any]) -> dict[str, Any]:
    """Accept an operation's params in the same shape as the standalone tool: tools that
    take a nested ``fields`` dict (set_clip, set_pacing, add_item, ...) may pass it nested
    or flat. Per-call keys (workspace, reason) are ignored: the batch carries them."""
    out = {k: v for k, v in params.items() if k not in _BATCH_LEVEL}
    nested = out.pop("fields", None)
    if nested is not None:
        if not isinstance(nested, dict):
            raise EditError(f"{op}: `fields` must be an object")
        clash = sorted(set(nested) & set(out))
        if clash:
            raise EditError(f"{op}: {', '.join(clash)} given both inside and outside `fields`")
        out.update(nested)
    return out


class Op(P):
    op: str = Field(description=f"one of {', '.join(_BATCHABLE)}")
    params: dict[str, Any] = Field(
        default_factory=dict, description="the same arguments as the standalone tool "
        "(nested `fields` or flat both work); workspace and reason come from the batch")


class BatchP(EditP):
    operations: list[Op] = Field(min_length=1, max_length=500)


@tool("batch", "edit", BatchP)
def batch(ctx: Context, p: BatchP) -> dict:
    """Several operations as one version (e.g. building a first cut); all or nothing."""
    ws = _ws(p.workspace)
    manifest = _manifest(ws)

    def run_all(edl: EDL) -> EDL:
        for n, o in enumerate(p.operations):
            fn = _BATCHABLE.get(o.op)
            if fn is None:
                raise EditError(f"operation {n}: unknown op {o.op!r}")
            try:
                edl = fn(edl, manifest, **_op_params(o.op, o.params))
            except TypeError as e:
                raise EditError(f"operation {n} ({o.op}): {e}") from e
            except EditError as e:
                raise EditError(f"operation {n} ({o.op}): {e}") from e
        return edl

    ch = EditStore(ws).apply("batch", {"operations": [o.model_dump() for o in p.operations]},
                             p.reason, run_all, expected_version=p.expected_version)
    return {"version": ch.version, "changed": ch.changed, "summary": ch.summary}


class UndoP(EditP):
    to_version: int = Field(ge=1)


@tool("undo", "edit", UndoP)
def undo(ctx: Context, p: UndoP) -> dict:
    """Restore an earlier version (as a new version; history is kept)."""
    ch = EditStore(_ws(p.workspace)).undo(p.to_version, p.reason)
    return {"version": ch.version, "changed": ch.changed, "summary": ch.summary}


# =============================================================================== check
def _agent_audit(ws: Workspace, write: bool):
    from ..stages.cut_audit import AuditConfig, audit, run_cut_audit

    cfg = AuditConfig(editorial_fixes=False)
    if write:
        return run_cut_audit(ws, EditStore(ws).current(), cfg)
    return audit(EditStore(ws).current(), Workspace.read(ws.transcript, Transcript),
                 _manifest(ws), cfg)


@tool("audit", "check", WsP)
def audit_tool(ctx: Context, p: WsP) -> dict:
    """Check the current edit. Reports problems with suggested fixes; changes nothing."""
    ws = _ws(p.workspace)
    _, rep = _agent_audit(ws, write=False)
    EditStore(ws).note("check", "audit", {}, "", f"{len(rep.violations)} findings, "
                       f"{'ok' if rep.passed else 'blocking errors'}")
    return {"passed": rep.passed, "duration_after_pacing": rep.duration_after,
            "pauses_removed": rep.pauses_removed,
            "findings": [v.model_dump(exclude_none=True) for v in rep.violations]}


class ReviewP(WsP):
    detector: str = Field("auto", description="auto | mediapipe | yolo | none")


@tool("review", "check", ReviewP)
def review(ctx: Context, p: ReviewP) -> dict:
    """Prepare the edit for render (pacing applied, guardrails only), find faces for
    crops/captions, and build the review sheet (one frame per cut, overlay, graphic)."""
    from ..stages.detectors import get_detector
    from ..stages.geometry import run_geometry
    from ..stages.review import build_review

    ws = _ws(p.workspace)
    _, rep = _agent_audit(ws, write=True)
    if not rep.passed:
        return {"error": "the edit has blocking problems: "
                         + "; ".join(v.message for v in rep.unresolved)}
    det = get_detector(p.detector)
    try:
        run_geometry(ws, detector=det)
    finally:
        det.close()
    rv = build_review(ws)
    sheets = [str(ws.root / s) for s in rv.sheets]
    if ctx.privacy == "stills":
        ctx.images_out = sheets
    EditStore(ws).note("check", "review", {}, "", f"{len(rv.tiles)} tiles, "
                       f"{len(rv.warnings)} warnings")
    return {"sheets": sheets, "warnings": rv.warnings, "tiles": len(rv.tiles),
            "duration": rep.duration_after}


class RenderP(WsP):
    out: str = Field(description="output .mp4 path, outside the footage folder")
    reason: str = Field(min_length=3, description="why the edit is ready (your approval)")
    draft: bool = Field(False, description="fast low-quality encode for checking")
    encoder: str = "auto"


@tool("render", "check", RenderP)
def render_tool(ctx: Context, p: RenderP) -> dict:
    """Approve the current review as the editor and render (two-pass -14 LUFS, chapters,
    captions .srt). Call review first; any edit after review needs a new review."""
    from ..stages.render import render
    from ..stages.review import ReviewRequired, decide

    ws = _ws(p.workspace)
    out = Path(p.out).expanduser().resolve()
    check_output_path(out, ws.sources)  # before approving: nothing changes on refusal
    if not (ws.review_dir / "review.json").exists():
        raise ValueError("no review yet: call review first")
    decide(ws, True, f"approved by the editor agent: {p.reason}")
    try:
        res = render(ws, out=out, encoder=p.encoder, quality="draft" if p.draft else "high",
                     require_review=True)
    except ReviewRequired as e:
        raise ValueError(f"{e} (call review again after editing)") from e
    q = res.qc
    EditStore(ws).note("check", "render", {"out": str(out), "draft": p.draft}, p.reason,
                       f"{q.duration:.1f}s, {q.integrated_lufs} LUFS, QC "
                       f"{'passed' if q.passed else 'failed'}")
    return {"output": str(res.output), "qc": q.model_dump(exclude={"schema_version"})}


class PublishP(WsP):
    titles: list[str] = Field(min_length=1, description="best first")
    hook: str
    body: str
    tags: list[str]
    hashtags: list[str]
    category: str
    pinned_comment: str = ""
    thumbnails: list[dict[str, Any]] = Field(
        default_factory=list, description="up to 3: {clip_id, segment_id, text, style "
        "(bold|question|reveal), why}")
    video: str | None = None
    geocode: bool = False
    use_location: bool = False


@tool("publish", "check", PublishP)
def publish_tool(ctx: Context, p: PublishP) -> dict:
    """Write <video>.upload.md/.json and thumbnails from copy you wrote (limits enforced)."""
    from ..stages.publish import UploadCopy, publish

    ws = _ws(p.workspace)
    copy = UploadCopy.model_validate({"titles": p.titles, "hook": p.hook, "body": p.body,
                                      "tags": p.tags, "hashtags": p.hashtags,
                                      "category": p.category,
                                      "pinned_comment": p.pinned_comment,
                                      "thumbnails": p.thumbnails})
    pkg = publish(ws, None, video=Path(p.video) if p.video else None, copy=copy,
                  geocode=p.geocode, use_location=p.use_location)
    EditStore(ws).note("check", "publish", {}, "", f"{Path(pkg.video).name}: "
                       f"{pkg.titles[0]!r}, {len(pkg.thumbnails)} thumbnails")
    if ctx.privacy == "stills":
        ctx.images_out = [t.path for t in pkg.thumbnails]
    return {"upload_md": str(Path(pkg.video).with_suffix(".upload.md")),
            "titles": pkg.titles, "thumbnails": [t.model_dump() for t in pkg.thumbnails],
            "warnings": pkg.warnings}


class MediaSearchP(WsP):
    query: str
    kind: Literal["music", "sfx", "video"]
    providers: list[str] | None = None
    limit: int = Field(10, ge=1, le=50)
    length: Literal["short", "medium", "long"] | None = None


@tool("media_search", "check", MediaSearchP)
def media_search(ctx: Context, p: MediaSearchP) -> dict:
    """Search licensed music, sound effects or stock video (commercial use + editing)."""
    from ..media.library import search

    res = search(p.query, p.kind, providers=tuple(p.providers) if p.providers else None,
                 limit=p.limit, length=p.length, ws=_ws(p.workspace))
    return {"results": [r.model_dump() for r in res.results], "errors": res.errors}


class MediaGetP(WsP):
    result_id: str
    library: str | None = None


@tool("media_get", "check", MediaGetP)
def media_get(ctx: Context, p: MediaGetP) -> dict:
    """Download a search result (credit recorded). Video is registered as a source clip
    so it can be used with add_clip or as an overlay."""
    from ..media.library import fetch, find_result
    from ..stages.cutaways import _register

    ws = _ws(p.workspace)
    hit = find_result(ws, p.result_id)  # any earlier search, not only the last
    path = fetch(ws, hit, library=Path(p.library) if p.library else None)
    out = {"file": str(path), "license": hit.license, "credit": hit.attribution}
    if hit.kind == "video":
        manifest = _manifest(ws)
        info = _register(manifest, Path(path), hit.title)
        Workspace.write(ws.clips_manifest, manifest)
        out["source_clip_id"] = info.clip_id
    return out


class JobStatusP(WsP):
    job_id: str
    wait_seconds: float = Field(30, ge=0, le=jobs.MAX_WAIT,
                                description="wait up to this long for the job to finish")


@tool("job_status", "check", JobStatusP)
def job_status(ctx: Context, p: JobStatusP) -> dict:
    """Progress or result of a long tool (analyze, review, render) that returned a
    job_id. Waits up to wait_seconds; returns the tool's result once it is done."""
    ws = _ws(p.workspace)
    job = jobs.wait_for(ws.root, p.job_id, p.wait_seconds)
    if job["status"] == "done" and ctx.privacy == "stills":
        ctx.images_out = job.get("images", [])
    return jobs.summary(job)


# =============================================================================== profile
class NoP(P):
    pass


@tool("profile", "look", NoP)
def profile_get(ctx: Context, p: NoP) -> dict:
    """The user's style profile (channel, audience, languages, look, likes, dislikes) and
    this machine's capabilities. Read it before planning an edit."""
    import json as _json

    from .setup import config_dir, load_profile

    machine = config_dir() / "machine.json"
    return {"profile": load_profile().model_dump(),
            "machine": _json.loads(machine.read_text()) if machine.exists() else
            "unknown: run `llmve setup`"}


class RememberP(P):
    reason: str = Field(min_length=3, description="the note or decision this comes from")
    add_likes: list[str] = Field(default_factory=list)
    add_dislikes: list[str] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)
    set_fields: dict[str, Any] = Field(default_factory=dict,
                                       description="channel, audience, languages, default_look")


@tool("remember", "edit", RememberP)
def remember(ctx: Context, p: RememberP) -> dict:
    """Update the style profile from a note the user accepted (e.g. dislike: film grain).
    Privacy cannot be changed from here."""
    from .setup import update_profile

    if "privacy" in p.set_fields:
        raise ValueError("privacy is the user's setting: change it with `llmve profile set`")
    prof = update_profile(set_fields=p.set_fields, add_likes=p.add_likes,
                          add_dislikes=p.add_dislikes, remove=p.remove)
    return {"profile": prof.model_dump()}


def describe() -> list[dict]:
    """Name, group, description and JSON schema of every tool."""
    return [{"name": t.name, "kind": t.kind, "description": t.description,
             "input_schema": t.schema()} for t in REGISTRY.values()]
