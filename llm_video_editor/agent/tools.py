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
from ..stages.cut_audit import doubtful
from ..workspace import Workspace, check_output_path
from . import jobs, ops
from .edits import EditConflict, EditError, EditStore
from .footage import group_recordings, inspect, select_inputs

Privacy = Literal["stills", "text-only"]


class P(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WsP(P):
    workspace: str = Field(description="workspace directory returned by open_project")


class NoArgsP(P):
    workspace: str | None = Field(None, description="not needed")


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
    object_detector_factory: Callable[..., Any] | None = None  # tests inject a fake
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
    """Transcript segments (ids, source times, text, retake and low-confidence hints).
    doubtful_words counts words ASR probably invented over noise (wind, engine)."""
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
                          "unclear_words": sum(w.prob < 0.5 for w in s.words),
                          "doubtful_words": sum(doubtful(w, ct.language) for w in s.words)})
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
    return {"words": [{"start": w.start, "end": w.end, "text": w.text, "prob": w.prob,
                       **({"doubtful": True} if doubtful(w, ct.language) else {})}
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


class PlacesP(WsP):
    lookup: bool = Field(False, description="name each stop (town, nearest temple/fort/"
                         "landmark, road, state) via OpenStreetMap; only if the user allowed "
                         "location_lookup in their profile")
    stop_radius_km: float = Field(1.0, gt=0, le=50,
                                  description="clips this close together form one stop")
    nearby_km: float = Field(0, ge=0, le=5, description="with lookup: also list named "
                             "temples, forts, forests, parks, rivers and highways within this "
                             "radius of each stop (0 = off)")


@tool("places", "look", PlacesP)
def places_tool(ctx: Context, p: PlacesP) -> dict:
    """Where and when each clip was filmed: the route in recording order from the
    cameras' GPS and clock, distances, and stops (clips filmed close together). With
    lookup, each stop gets place names for titles; without, coordinates only."""
    from .places import name_stops, nearby_stops, route
    from .setup import load_profile

    ws = _ws(p.workspace)
    ids = {}
    if ws.clips_manifest.exists():
        ids = {str(Path(c.source_file).resolve()): c.clip_id for c in _manifest(ws).clips}
    res = route(_footage(ws), ids, stop_radius_km=p.stop_radius_km)
    if not res["clips_with_gps"]:
        res["note"] = ("no clip has GPS: ask the user where the footage was filmed, or read "
                       "the place from signs in frames and the transcript")
        return res
    if p.lookup:
        if not load_profile().location_lookup:
            res["note"] = ("place names need the user's permission (coordinates go to "
                           "OpenStreetMap): they can run `llmve profile --set "
                           "location_lookup=true`; meanwhile ask them or use the transcript")
        else:
            res["stops"], res["errors"] = name_stops(res["stops"], ws.root / "places.json")
            if p.nearby_km:
                res["stops"], errs = nearby_stops(res["stops"], ws.root / "places_nearby.json",
                                                  radius_km=p.nearby_km)
                res["errors"] += errs
            EditStore(ws).note("check", "places", {"lookup": True}, "",
                               f"named {sum('place' in s for s in res['stops'])} stops")
    for s in res["stops"]:
        s.pop("key", None)
    return res


class ObjectsP(WsP):
    clip_id: str = Field(description="source clip id (from analyze/overview)")
    start: float = Field(0, ge=0)
    end: float | None = Field(None, description="default: end of clip (max 600 s window)")
    every: float = Field(2.0, ge=0.25, le=30, description="seconds between sampled frames")
    min_confidence: float = Field(0.35, ge=0.05, le=0.95)
    labels: list[str] | None = Field(None, description="only these, e.g. [\"cow\", \"dog\"]")


@tool("objects", "look", ObjectsP)
def objects_tool(ctx: Context, p: ObjectsP) -> dict:
    """Everyday objects in a clip, detected locally (nothing is uploaded): animals (cow,
    dog, horse, bird, elephant, ...), vehicles, people and other COCO classes, with when
    they are visible and where in the frame (x, y of the box centre, 0-1, for a label
    graphic). Temples, forests and scenery are not classes: look at frames for those."""
    from ..stages.geometry import grab_frame

    ws = _ws(p.workspace)
    clip = _manifest(ws).get(p.clip_id)
    if clip is None:
        raise KeyError(f"no source clip {p.clip_id!r}: see overview")
    end = min(p.end if p.end is not None else clip.duration, clip.duration)
    if end - p.start > 600:
        raise ValueError("window too long (max 600 s): look at shorter windows")
    factory = ctx.object_detector_factory
    if factory is None:
        try:
            from ..stages.detectors import YoloObjectDetector
        except ImportError as e:  # pragma: no cover - import is lazy inside
            raise ValueError(str(e)) from e
        factory = YoloObjectDetector
    try:
        det = factory(min_confidence=p.min_confidence)
    except ImportError as e:
        raise ValueError("object detection needs ultralytics: `uv sync --extra pro` "
                         f"({e})") from e
    want = {x.lower() for x in p.labels} if p.labels else None
    seen: dict[str, dict] = {}
    t, frames = p.start, 0
    try:
        while t < end - 0.04:  # a frame exactly at the end does not decode
            frames += 1
            for label, box, score in det.detect(grab_frame(clip.source_file, t)):
                if want and label.lower() not in want:
                    continue
                area = (box[2] - box[0]) * (box[3] - box[1])
                s = seen.setdefault(label, {"label": label, "frames": 0, "first": t,
                                            "last": t, "best": None})
                s["frames"] += 1
                s["last"] = t
                if s["best"] is None or score * (0.5 + area) > s["best"]["rank"]:
                    s["best"] = {"t": round(t, 2), "score": round(score, 2),
                                 "x": round((box[0] + box[2]) / 2, 3),
                                 "y": round((box[1] + box[3]) / 2, 3),
                                 "box": [round(v, 3) for v in box],
                                 "rank": score * (0.5 + area)}
            t += p.every
    finally:
        det.close()
    found = sorted(seen.values(), key=lambda s: -s["frames"])
    for s in found:
        s["best"].pop("rank")
        s["first"], s["last"] = round(s["first"], 2), round(s["last"], 2)
    return {"clip_id": p.clip_id, "frames_checked": frames, "objects": found,
            "note": "x, y are in the source frame; a label graphic stays put, so use it "
                    "while the object holds still or for a short time"}


@tool("graphics_templates", "look", NoArgsP)
def graphics_templates(ctx: Context, p: NoArgsP) -> dict:
    """Built-in motion graphics (for add_item kind=graphics, `template`) and their
    variables. A custom HyperFrames project folder can be used as `composition`."""
    from ..hyperframes.compose import templates

    return {"templates": [{"name": t.name, "description": t.description,
                           "variables": {v["id"]: v.get("default") for v in t.variables}}
                          for t in templates().values()]}


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


@tool("set_grade", "edit", FieldsP)
def set_grade(ctx: Context, p: FieldsP) -> dict:
    """One look for the whole video: look (none, natural, warm, golden-hour, teal-orange,
    cinematic, vivid, matte, bw), strength 0-1 (0.4-0.6 reads premium), or lut (a .cube
    file). Applied to the footage only, never to graphics."""
    return _edit(p, ops.set_grade, p.fields, "set_grade")


class ColorP(EditP):
    clip_id: str
    fields: dict[str, Any] = Field(default_factory=dict, description="exposure (EV), "
                                   "contrast, saturation, warmth, tint, tonemap (HDR); "
                                   "{} resets the clip")


@tool("set_color", "edit", ColorP)
def set_color(ctx: Context, p: ColorP) -> dict:
    """Per-clip correction so clips match each other (use color_report's suggestions);
    looks belong on set_grade."""
    return _edit(p, ops.set_color, {"clip_id": p.clip_id, **p.fields}, "set_color")


class ColorReportP(WsP):
    samples: int = Field(3, ge=1, le=8, description="frames measured per clip")


@tool("color_report", "look", ColorReportP)
def color_report(ctx: Context, p: ColorReportP) -> dict:
    """Measure every clip of the current edit (exposure, contrast, saturation, colour
    cast, clipping), how consistent they are, suggested per-clip corrections towards
    the median clip, HDR clips that need tone mapping, and warnings about the current
    grade. Measures the ungraded source."""
    import statistics

    from ..stages.grade import color_warnings, measure_clip, suggest_matches

    ws = _ws(p.workspace)
    edl = EditStore(ws).current()
    facts = {Path(k).resolve(): v for k, v in _footage(ws).items()}
    measured, hdr = {}, []
    for c in edl.clips:
        measured[c.clip_id] = measure_clip(c.source_file, c.start, c.end, p.samples)
        f = facts.get(Path(c.source_file).resolve(), {})
        if f.get("hdr") and not (c.color and c.color.tonemap):
            hdr.append(c.clip_id)
    sug = suggest_matches(measured)
    spread = {k: round(statistics.pstdev([m[k] for m in measured.values()]), 3)
              for k in ("mid", "saturation", "warm")} if len(measured) > 1 else {}
    return {"clips": [{"clip_id": cid, **m, **sug[cid]} for cid, m in measured.items()],
            "spread": spread, "hdr_needs_tonemap": hdr,
            "grade": edl.grade.model_dump(), "warnings": color_warnings(edl),
            "note": "mid ~0.35-0.5 is a normal exposure; spread above ~0.08 (mid) or 0.06 "
                    "(warm) shows as clips not matching"}


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
              "remove_item": ops.remove_item, "set_grade": ops.set_grade,
              "set_color": ops.set_color}


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
    from ..stages.grade import color_warnings

    vo = _vo_report(EditStore(ws).current(), ws)
    colour = color_warnings(EditStore(ws).current())
    return {"passed": rep.passed, "duration_after_pacing": rep.duration_after,
            "pauses_removed": rep.pauses_removed,
            "findings": [v.model_dump(exclude_none=True) for v in rep.violations],
            "voiceover": [r for r in vo if r["issues"]] or "no voice-over issues",
            "color": colour or "no colour issues"}


class ReviewP(WsP):
    detector: str = Field("auto", description="auto | mediapipe | yolo | none")


@tool("review", "check", ReviewP)
def review(ctx: Context, p: ReviewP) -> dict:
    """Prepare the edit for render (pacing applied, guardrails only), find faces for
    crops/captions, and build the review sheet (one frame per cut, overlay, graphic).
    Each graphic is measured on its shot: contrast, busy background, size, safe area,
    face/caption overlap, the shot's palette and suggested colours (accent "auto" uses
    them)."""
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
    text = _on_screen_text(Workspace.read(ws.edl_audited, EDL))  # as it will render
    EditStore(ws).note("check", "review", {}, "", f"{len(rv.tiles)} tiles, "
                       f"{len(rv.warnings)} warnings, {text['graphics']} graphics")
    return {"sheets": sheets, "warnings": rv.warnings, "tiles": len(rv.tiles),
            "duration": rep.duration_after, "on_screen_text": text,
            "graphic_checks": [c.model_dump(exclude_none=True) for c in rv.graphic_checks]}


def _on_screen_text(edl: EDL) -> dict:
    """Facts about titles and text in the edit (a report; the agent decides)."""
    starts = dict(zip((c.clip_id for c in edl.clips), edl.output_starts(), strict=True))
    spans = sorted((starts[g.anchor_clip_id] + g.offset,
                    starts[g.anchor_clip_id] + g.offset + g.duration)
                   for g in edl.graphics if g.anchor_clip_id in starts)
    total = edl.total_duration
    gaps, t = [], 0.0
    for a, b in spans:
        if a > t:
            gaps.append((t, a))
        t = max(t, b)
    if total > t:
        gaps.append((t, total))
    longest = max(gaps, key=lambda g: g[1] - g[0], default=(0.0, 0.0))
    chapters = [(c.chapter, starts[c.clip_id]) for c in edl.clips if c.chapter]
    untitled = [name for name, at in chapters
                if not any(a - 1.0 <= at <= b for a, b in spans)]
    out = {"graphics": len(spans),
           "templates": sorted({g.template or "custom" for g in edl.graphics}),
           "seconds_with_graphics": round(sum(b - a for a, b in spans), 1),
           "longest_without_graphics": {"from": round(longest[0], 1),
                                        "to": round(longest[1], 1),
                                        "seconds": round(longest[1] - longest[0], 1)},
           "captions": edl.captions.enabled,
           "chapters_without_a_title_graphic": untitled}
    if not spans:
        out["note"] = ("no titles or text graphics in the video. Place names, chapter "
                       "cards and callouts are add_item kind=graphics (see "
                       "graphics_templates; `places` tells where clips were filmed)")
    return out


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
        default_factory=list, description="up to 3: {clip_id, segment_id or at (source "
        "seconds in the clip, for footage without speech), text, style "
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


# =============================================================================== voice-over
class VoicesP(P):
    engine: str = Field("auto", description="kokoro | say | auto")
    language: str | None = Field(None, description="filter, e.g. Hindi, en-gb")


@tool("voices", "look", VoicesP)
def voices_tool(ctx: Context, p: VoicesP) -> dict:
    """Voices for narration on this machine (local engines only; nothing is uploaded):
    engine, voice id, language, gender, licence."""
    from ..voice import EngineUnavailable, list_voices

    try:
        vs = list_voices(p.engine, p.language)
    except EngineUnavailable as e:
        raise ValueError(str(e)) from e
    return {"voices": [vars(v) for v in vs][:200],
            "note": "kokoro voice ids: a=US English, b=UK English, h=Hindi; f/m = female/male "
                    "(af_heart, am_michael, bf_emma, bm_george, hf_alpha, hm_omega, ...)"}


class VoLine(P):
    text: str = Field(min_length=1, max_length=400,
                      description="the line as it should be heard and captioned")
    anchor_clip_id: str = Field(description="edit clip the line starts on")
    offset: float = Field(0.0, ge=0, description="seconds after that clip starts")
    voice: str | None = None
    speed: float | None = Field(None, ge=0.5, le=1.6)
    volume_db: float | None = Field(None, ge=-30, le=12,
                                    description="relative to the speech; default 0")


class VoiceoverP(EditP):
    lines: list[VoLine] = Field(min_length=1, max_length=60)
    engine: str = Field("auto", description="kokoro | say | auto")
    voice: str = Field("af_heart", description="default voice for lines without one")
    speed: float = Field(1.0, ge=0.5, le=1.6)
    lang: str | None = Field(None, description="override the voice's language, e.g. hi")
    audition: bool = Field(False, description="only synthesize and report lengths; the "
                           "edit is not changed")


def _vo_report(edl: EDL, ws: Workspace) -> list[dict]:
    """Where each voice-over sits and what it collides with (a report)."""
    from ..ffmpeg.probe import media_duration
    from ..stages.captions import output_words

    starts = edl.output_starts()
    index = {c.clip_id: i for i, c in enumerate(edl.clips)}
    total = edl.total_duration
    durations = [c.duration for c in edl.clips]
    speech = output_words(edl.model_copy(update={"music": []}),
                          Workspace.read(ws.transcript, Transcript), starts, durations) \
        if ws.transcript.exists() else []
    spans = []
    for m in edl.music:
        if m.role != "voiceover" or m.anchor_clip_id not in index:
            continue
        a = starts[index[m.anchor_clip_id]] + m.offset
        try:
            length = m.duration or media_duration(Path(m.file).expanduser()) - m.src_start
        except Exception:  # noqa: BLE001
            length = 0.0
        spans.append((m, a, a + length))
    out = []
    for m, a, b in spans:
        talk = sum(max(0.0, min(b, we) - max(a, ws_)) for ws_, we, _, _ in speech)
        others = [o.track_id for o, oa, ob in spans if o is not m and oa < b and ob > a]
        issues = []
        if talk > 0.15:
            issues.append(f"over {talk:.1f}s of speech in the clips")
        if others:
            issues.append(f"overlaps voice-over {', '.join(others)}")
        if b > total + 0.05:
            issues.append(f"runs {b - total:.1f}s past the end of the video")
        if not Path(m.file).expanduser().exists():
            issues.append("audio file missing")
        out.append({"track_id": m.track_id, "text": m.text, "from": round(a, 2),
                    "to": round(b, 2), "issues": issues})
    return out


@tool("voiceover", "edit", VoiceoverP)
def voiceover_tool(ctx: Context, p: VoiceoverP) -> dict:
    """Narration lines you wrote, spoken by a local voice and added as voice-over tracks
    (one version). Each line is clean 48 kHz audio, levelled to the speech at render;
    music ducks under it and the clips' own sound dips (mix.clip_duck_db). Captions use
    the text. Returns each line's exact length and what it collides with."""
    from ..voice import EngineUnavailable, get_engine, synthesize

    ws = _ws(p.workspace)
    try:
        eng = get_engine(p.engine)
    except EngineUnavailable as e:
        raise ValueError(str(e)) from e
    made = []
    for ln in p.lines:
        voice = ln.voice or p.voice
        if eng.name == "say" and voice == "af_heart":
            voice = "Samantha"
        made.append((ln, synthesize(ws.root, eng, ln.text, voice, speed=ln.speed or p.speed,
                                    lang=p.lang)))
    lines = [{"text": line.text, "spoken": line.spoken, "seconds": line.duration,
              "voice": line.voice, "file": line.file} for _, line in made]
    if p.audition:
        return {"lines": lines, "note": "audition only: the edit is unchanged"}

    def add_all(edl: EDL) -> EDL:
        manifest = _manifest(ws)
        for ln, line in made:
            fields = {"file": line.file, "role": "voiceover", "anchor_clip_id": ln.anchor_clip_id,
                      "offset": ln.offset, "text": line.text,
                      "generator": f"{line.engine}:{line.voice}"}
            if ln.volume_db is not None:
                fields["volume_db"] = ln.volume_db
            edl = ops.add_item(edl, manifest, kind="music", **fields)
        return edl

    store = EditStore(ws)
    ch = store.apply("voiceover", {"lines": [ln.model_dump() for ln in p.lines],
                                   "engine": eng.name, "voice": p.voice}, p.reason, add_all,
                     expected_version=p.expected_version)
    report = _vo_report(store.current(), ws)
    new_ids = {r["track_id"] for r in report[-len(made):]}
    return {"version": ch.version, "summary": ch.summary, "lines": lines,
            "fit": [r for r in report if r["track_id"] in new_ids]}


class PronounceP(P):
    reason: str = Field(min_length=3)
    say_as: dict[str, str] = Field(description="word -> respelling, e.g. "
                                   "{\"Khajuraho\": \"Kha-joo-raa-ho\"}")
    scope: Literal["project", "all"] = Field("project", description="this workspace, or "
                                             "every project (your config folder)")
    workspace: str | None = None


@tool("pronounce", "edit", PronounceP)
def pronounce_tool(ctx: Context, p: PronounceP) -> dict:
    """How the narrator says names (places, people): respellings applied before speech;
    captions keep the real spelling. Re-run voiceover for lines that use them."""
    from ..voice.synth import voice_dir
    from .setup import config_dir

    if p.scope == "project":
        if not p.workspace:
            raise ValueError("scope project needs the workspace")
        path = voice_dir(_ws(p.workspace).root) / "lexicon.json"
    else:
        path = config_dir() / "lexicon.json"
    lex = json.loads(path.read_text()) if path.exists() else {}
    lex.update(p.say_as)
    path.write_text(json.dumps(lex, indent=1, ensure_ascii=False))
    return {"lexicon": lex, "file": str(path)}


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
    Privacy and location lookup cannot be changed from here."""
    from .setup import update_profile

    for k in ("privacy", "location_lookup"):
        if k in p.set_fields:
            raise ValueError(f"{k} is the user's setting: they change it with "
                             f"`llmve profile --set {k}=...`")
    prof = update_profile(set_fields=p.set_fields, add_likes=p.add_likes,
                          add_dislikes=p.add_dislikes, remove=p.remove)
    return {"profile": prof.model_dump()}


def describe() -> list[dict]:
    """Name, group, description and JSON schema of every tool."""
    return [{"name": t.name, "kind": t.kind, "description": t.description,
             "input_schema": t.schema()} for t in REGISTRY.values()]
