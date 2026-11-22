"""``llmve``: stage-by-stage CLI over the checkpointed pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from typer.core import TyperGroup

from ..schemas import CHECKPOINTS
from ..workspace import Workspace

app = typer.Typer(no_args_is_help=True, help="Checkpointed, prompt-driven video editor.")
console = Console()

HOME = typer.Option(None, "--home", help="Workspace root (default: $LLMVE_HOME or ./.llmve)")


def _open(inputs: list[str], home: Path | None) -> Workspace:
    return Workspace.open(inputs if len(inputs) > 1 else inputs[0], home=home)


@app.command()
def schema(name: str = typer.Argument(..., help=f"One of: {', '.join(CHECKPOINTS)}")) -> None:
    """Print the JSON Schema for a checkpoint."""
    if name not in CHECKPOINTS:
        raise typer.BadParameter(f"unknown schema {name!r}")
    typer.echo(json.dumps(CHECKPOINTS[name].model_json_schema(), indent=2))


def _chunking(over_minutes: float | None, workers: int):
    from ..stages.chunked_asr import ChunkConfig

    if over_minutes is None:
        return None
    return ChunkConfig(min_duration=over_minutes * 60, workers=workers)


@app.command()
def watch(
    inputs: list[str] = typer.Argument(..., help="Video file(s), folder(s) or glob(s)"),
    asr_model: str = typer.Option("distil-large-v3", help="faster-whisper model"),
    language: str | None = typer.Option(None, help="Force ASR language (e.g. en)"),
    chunk_asr_over: float | None = typer.Option(
        None, help="Chunked/parallel ASR for clips longer than this many minutes (e.g. 10)"),
    asr_workers: int = typer.Option(0, help="ASR worker processes for chunked ASR (0 = auto)"),
    home: Path | None = HOME,
) -> None:
    """Transcribe, score scenes, classify A-roll/B-roll and build overview sheets."""
    from ..stages.asr import FasterWhisperTranscriber
    from ..stages.watch import watch as run_watch

    ws = _open(inputs, home)
    res = run_watch(ws, transcriber=FasterWhisperTranscriber(asr_model), language=language,
                    chunking=_chunking(chunk_asr_over, asr_workers))
    table = Table(title=f"Workspace {ws.root}" + (" (cached)" if res.cached else ""))
    for col in ("clip_id", "kind", "duration", "size", "fps", "speech", "motion", "sheets"):
        table.add_column(col)
    for c in res.manifest.clips:
        table.add_row(c.clip_id, c.kind, f"{c.duration:.1f}s", f"{c.width}x{c.height}",
                      f"{c.fps:g}", f"{c.speech_ratio:.0%}", f"{c.motion_score:.3f}",
                      str(len(c.overview_sheets)))
    console.print(table)
    console.print(f"Overview sheets: ~{res.sheets.est_tokens} vision tokens total")
    typer.echo(str(ws.root))


@app.command("watch-window")
def watch_window_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory printed by `watch`"),
    clip_id: str = typer.Argument(...),
    start: float = typer.Argument(...),
    end: float = typer.Argument(...),
    fps: float = typer.Option(2.0, help="Sampling rate for the dense sheet"),
) -> None:
    """Dense contact sheets for one candidate window."""
    from ..ffmpeg.sheets import SheetSpec
    from ..stages.watch import watch_window

    sheets = watch_window(Workspace.load(workspace), clip_id, start, end, spec=SheetSpec(fps=fps))
    for s in sheets:
        typer.echo(f"{workspace / s.path}\t{s.start:.2f}-{s.end:.2f}s\t~{s.est_tokens} tokens")


@app.command("transcript")
def transcript_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory printed by `watch`"),
    clip_id: str | None = typer.Option(None, "--clip", help="Only this clip"),
) -> None:
    """Write transcript.md: segment ids, times, pauses and (retake?) markers per clip."""
    from ..schemas import Transcript
    from ..stages.plan import format_transcript

    ws = Workspace.load(workspace)
    tr = Workspace.read(ws.transcript, Transcript)
    clips = [c for c in tr.clips if clip_id in (None, c.clip_id)]
    if not clips:
        raise typer.BadParameter(f"no transcript for clip {clip_id!r}")
    out = ws.root / "transcript.md"
    out.write_text("\n\n".join(format_transcript(c) for c in clips) + "\n", encoding="utf-8")
    typer.echo(str(out))


TARGETS = "youtube | shorts | reels | tiktok | twitter | linkedin"
PROVIDER = typer.Option("claude-cli", help="Planner: claude-cli | ollama | anthropic | openai")


@app.command("plan")
def plan_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory printed by `watch`"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="What the edit should be"),
    target: str = typer.Option("youtube", "--target", "-t", help=TARGETS),
    target_duration: str | None = typer.Option(None, help="e.g. 10m, 90s, 1:30"),
    provider: str = PROVIDER,
    model: str | None = typer.Option(None, help="Provider-specific model name"),
) -> None:
    """Map/reduce planning -> asset_plan.json + edl_raw.json (text only leaves the machine)."""
    from ..llm import get_provider
    from ..stages.durations import parse_duration
    from ..stages.plan import plan

    dur = parse_duration(target_duration) if target_duration else None
    asset_plan, edl = plan(Workspace.load(workspace), prompt, get_provider(provider, model),
                           target_platform=target, target_duration=dur)
    console.print(f"{len(edl.clips)} clips, {edl.total_duration:.1f}s of {asset_plan.target_duration:.1f}s "
                  f"target, {len(edl.overlays)} B-roll overlays")
    if asset_plan.notes:
        console.print(f"[yellow]notes:[/yellow] {asset_plan.notes}")


@app.command("cut-audit")
def cut_audit_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory printed by `watch`"),
) -> None:
    """Snap cuts to words/sentences, clamp, merge, enforce budget -> edl_audited.json."""
    from ..stages.cut_audit import run_cut_audit

    edl, report = run_cut_audit(Workspace.load(workspace))
    for v in report.violations:
        mark = "fixed" if v.fixed else v.severity.upper()
        console.print(f"[{mark}] {v.kind} {v.clip_id or ''} {v.message}" + (f" -> {v.fix}" if v.fix else ""))
    console.print(f"{'PASSED' if report.passed else 'FAILED'}: {report.clips_after} clips, "
                  f"{report.duration_after:.1f}s"
                  + (f" ({report.pauses_removed:.1f}s of pauses tightened)"
                     if report.pauses_removed else ""))
    if not report.passed:
        raise typer.Exit(1)


broll_app = typer.Typer(help="Local B-roll library: index, describe, match.",
                        no_args_is_help=True)
app.add_typer(broll_app, name="broll")


@broll_app.command("index")
def broll_index_cmd(
    library: Path = typer.Argument(..., help="B-roll folder"),
    rescan: bool = typer.Option(False, help="Re-probe every file"),
    sheets: bool = typer.Option(True, help="Write labelled thumbnail sheets"),
) -> None:
    """Incrementally index a B-roll folder -> .llmve_broll_index.json + broll_index.md."""
    from ..stages.broll import MD_NAME, index_library, thumbnail_sheets

    index, scanned = index_library(library, rescan=rescan)
    console.print(f"{len(index.items)} clips indexed ({scanned} scanned): {library / MD_NAME}")
    if sheets:
        for p in thumbnail_sheets(library):
            typer.echo(str(p))


@broll_app.command("describe")
def broll_describe_cmd(
    library: Path = typer.Argument(...),
    item_id: str = typer.Argument(...),
    description: str = typer.Argument(..., help="What the clip shows"),
    tags: str = typer.Option("", help="Comma-separated tags"),
) -> None:
    """Set a manual description (kept across rescans)."""
    from ..stages.broll import describe

    t = [x.strip() for x in tags.split(",") if x.strip()] or None
    item = describe(library, item_id, description, t)
    console.print(f"{item.item_id}: {item.description} [{', '.join(item.tags)}]")


@broll_app.command("match")
def broll_match_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory"),
    library: Path | None = typer.Option(None, "--library", "-l", help="Indexed B-roll folder"),
    provider: str | None = typer.Option(None, help="Re-rank top candidates with an LLM"),
    model: str | None = typer.Option(None),
    generate_missing: bool = typer.Option(False, help="Generate B-roll for empty slots (gen extra)"),
) -> None:
    """Fill B-roll slots in asset_plan.json from the library; rebuilds edl_raw.json."""
    from ..stages.broll import match_broll

    sel = match_broll(Workspace.load(workspace), library,
                      provider=_provider(provider, model) if provider else None,
                      generator=_generator(workspace) if generate_missing else None)
    for s in sel.slots:
        console.print(f"slot {s.slot}: {s.chosen or '-'} ({s.reason}) <- {s.query[:60]!r}")


def _provider(name: str, model: str | None):
    from ..llm import get_provider

    return get_provider(name, model)


def _generator(workspace: Path):
    def gen(query: str, seconds: float, orientation: str) -> Path:
        from ..generative.broll import BrollGenerator, BrollRequest

        res = "576x1024" if orientation == "portrait" else "1024x576"
        return BrollGenerator(workspace / "generated").generate(
            BrollRequest(mode="t2v", prompt=query, seconds=max(2, round(seconds)), resolution=res))

    return gen


@app.command("geometry")
def geometry_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory printed by `watch`"),
    detector: str = typer.Option("auto", help="auto | mediapipe | yolo | none"),
    samples: int = typer.Option(3, help="Keyframes analysed per used scene"),
) -> None:
    """Face/subject boxes per used scene -> crop centres and caption/PiP safe zones."""
    from ..stages.detectors import get_detector
    from ..stages.geometry import run_geometry

    det = get_detector(detector)
    try:
        gmap = run_geometry(Workspace.load(workspace), detector=det, samples=samples)
    finally:
        det.close()
    console.print(f"detector: {gmap.detector}")
    for p in gmap.placements:
        console.print(f"{p.clip_id}: crop {p.crop_center}, subject {p.subject_box_out}, "
                      f"captions {p.caption_position}, PiP {p.pip_corner}")


graphics_app = typer.Typer(help="HyperFrames motion graphics: templates, add, chapter cards.",
                           no_args_is_help=True)
app.add_typer(graphics_app, name="graphics")


def _vars(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise typer.BadParameter(f"--var expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v
    return out


@graphics_app.command("templates")
def graphics_templates_cmd() -> None:
    """List the built-in templates and their variables."""
    from ..hyperframes.compose import templates

    for t in templates().values():
        console.print(f"[bold]{t.name}[/bold]: {t.description}")
        for v in t.variables:
            console.print(f"  --var {v['id']}=...  {v.get('label', '')} "
                          f"(default {v.get('default')!r})")


@graphics_app.command("add")
def graphics_add_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory"),
    at: str = typer.Option(..., "--at", help="Clip id in edl_raw.json the graphic starts on"),
    template: str | None = typer.Option(None, "--template", "-t", help="Built-in template"),
    composition: Path | None = typer.Option(None, "--composition", "-c",
                                            help="HyperFrames project folder (has index.html)"),
    offset: float = typer.Option(0.0, help="Seconds after the clip starts"),
    duration: float = typer.Option(4.0, help="Seconds on screen"),
    var: list[str] = typer.Option([], "--var", help="Template variable, key=value (repeat)"),
    box: str | None = typer.Option(None, help="x,y,w,h in 0-1 (default: full frame)"),
) -> None:
    """Add a graphic to edl_raw.json; run cut-audit afterwards."""
    from ..stages.graphics import add_graphic

    if (template is None) == (composition is None):
        raise typer.BadParameter("pass exactly one of --template or --composition")
    b = tuple(float(x) for x in box.split(",")) if box else None
    if b is not None and len(b) != 4:
        raise typer.BadParameter("--box needs four numbers: x,y,w,h")
    g = add_graphic(Workspace.load(workspace), anchor=at, duration=duration, template=template,
                    composition=str(composition) if composition else None, offset=offset,
                    variables=_vars(var), box=b)
    console.print(f"{g.graphic_id}: {g.template or g.composition} on {g.anchor_clip_id} "
                  f"+{g.offset:g}s for {g.duration:g}s. Next: `llmve cut-audit {workspace}`")


@graphics_app.command("chapters")
def graphics_chapters_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory"),
    template: str = typer.Option("title-card", "--template", "-t"),
    duration: float = typer.Option(2.5, help="Seconds per card"),
    var: list[str] = typer.Option([], "--var", help="Extra variable for every card, key=value"),
    numbered: bool = typer.Option(True, help='Add a "Chapter N" kicker'),
) -> None:
    """One title card per chapter marker in edl_raw.json (replaces earlier cards)."""
    from ..stages.graphics import chapter_cards

    cards = chapter_cards(Workspace.load(workspace), template=template, duration=duration,
                          variables=_vars(var), numbered=numbered)
    for g in cards:
        console.print(f"{g.graphic_id}: {g.variables.get('title')!r} on {g.anchor_clip_id}")
    if not cards:
        console.print("[yellow]no clips with a chapter in edl_raw.json[/yellow]")


@graphics_app.command("render")
def graphics_render_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory"),
    draft: bool = typer.Option(True, help="Draft quality (fast); --no-draft for final quality"),
) -> None:
    """Render every graphic in edl_audited.json to transparent WebM (cached)."""
    from ..schemas import EDL, ClipsManifest
    from ..stages.graphics import graphic_specs
    from ..stages.render import timeline

    ws = Workspace.load(workspace)
    edl = Workspace.read(ws.edl_audited, EDL)
    _, starts = timeline(edl, Workspace.read(ws.clips_manifest, ClipsManifest))
    for spec in graphic_specs(ws, edl, starts, quality="draft" if draft else "high"):
        typer.echo(f"{spec.path}\t{spec.at:.2f}s +{spec.duration:.2f}s")


@graphics_app.command("doctor")
def graphics_doctor_cmd(
    install: bool = typer.Option(False, help="Install the pinned toolchain if missing"),
) -> None:
    """Check Node 22+, npm, the pinned HyperFrames toolchain and ffmpeg's libvpx-vp9."""
    from ..hyperframes import doctor, ensure

    if install:
        ensure()
    for k, v in doctor().items():
        console.print(f"{k}: {v}")


media_app = typer.Typer(help="Royalty-free music, sound effects and stock video, with credits.",
                        no_args_is_help=True)
app.add_typer(media_app, name="media")
KIND = typer.Option("music", "--kind", "-k", help="music | sfx | video")


@media_app.command("search")
def media_search_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory"),
    query: str = typer.Argument(..., help='What you need, e.g. "upbeat acoustic"'),
    kind: str = KIND,
    provider: list[str] = typer.Option([], "--provider", "-p",
                                       help="openverse | pixabay | pexels | wikimedia (repeat)"),
    limit: int = typer.Option(10, help="Results per provider"),
    length: str | None = typer.Option(None, help="Audio length: short | medium | long"),
) -> None:
    """Search licensed libraries (commercial use + editing allowed) -> media/search.json."""
    from ..media.library import search

    if kind not in ("music", "sfx", "video"):
        raise typer.BadParameter("--kind is music, sfx or video")
    res = search(query, kind, providers=tuple(provider) or None, limit=limit, length=length,
                 ws=Workspace.load(workspace))
    table = Table(title=f"{len(res.results)} {kind} results for {query!r}")
    for col in ("result_id", "title", "creator", "length", "license", "credit needed"):
        table.add_column(col)
    for r in res.results:
        table.add_row(r.result_id, r.title[:40], r.creator[:24],
                      f"{r.duration:.0f}s" if r.duration else "?", r.license,
                      "yes" if r.attribution_required else "no")
    console.print(table)
    for e in res.errors:
        console.print(f"[yellow]{e}[/yellow]")


@media_app.command("get")
def media_get_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory"),
    result_id: str = typer.Argument(..., help="result_id from `llmve media search`"),
    library: Path | None = typer.Option(None, help="B-roll folder for video (default: <ws>/media/broll)"),
) -> None:
    """Download one search result and record its credit."""
    from ..media.library import fetch, find_result

    ws = Workspace.load(workspace)
    try:
        hit = find_result(ws, result_id)
    except KeyError as e:
        raise typer.BadParameter(str(e).strip("'\"")) from e
    path = fetch(ws, hit, library=library)
    console.print(f"{path}\n  {hit.license}; credit: {hit.attribution or '(not required)'}")
    if hit.kind == "video":
        console.print(f"  next: `llmve broll index {path.parent}` or `llmve media fetch` slots")
    else:
        console.print(f"  next: `llmve music add {workspace} {path} --at <clip_id>`")


@media_app.command("add")
def media_add_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory"),
    file: Path = typer.Argument(..., exists=True, help="Audio/video you already have"),
    title: str = typer.Option(..., help="Track or clip title"),
    license: str = typer.Option(..., help='e.g. "YouTube Audio Library", "CC BY 4.0"'),
    kind: str = KIND,
    provider: str = typer.Option("manual", help="youtube-audio-library | manual | ..."),
    creator: str = typer.Option("", help="Artist / author"),
    attribution: str = typer.Option("", help="Credit line the licence asks for (exact text)"),
    license_url: str = typer.Option("", help="Licence or track page"),
) -> None:
    """Register a file from elsewhere (e.g. YouTube Audio Library) with its credit."""
    from ..media.library import register

    c = register(Workspace.load(workspace), file, kind=kind, title=title, license=license,
                 provider=provider, creator=creator, attribution=attribution,
                 license_url=license_url)
    console.print(f"{c.file}\n  credit: {c.attribution or '(none required)'}")


@media_app.command("credits")
def media_credits_cmd(workspace: Path = typer.Argument(..., help="Workspace directory")) -> None:
    """List every registered file and its credit line."""
    from ..media.library import load_credits

    for c in load_credits(Workspace.load(workspace)).items:
        need = "credit required" if c.attribution_required else "no credit required"
        console.print(f"[bold]{c.title}[/bold] ({c.kind}, {c.license}, {need})\n  {c.file}"
                      + (f"\n  {c.attribution}" if c.attribution else ""))


@media_app.command("suggest")
def media_suggest_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory"),
    humor: bool = typer.Option(True, help="Allow funny reaction clips (--no-humor: B-roll only)"),
    every: float = typer.Option(45.0, help="At most one cutaway per this many seconds"),
    brief: str = typer.Option("", help="Tone notes, e.g. 'playful, for a young audience'"),
    provider: str = PROVIDER,
    model: str | None = typer.Option(None),
) -> None:
    """Suggest a few cutaway moments (may be none) -> media/cutaways.json."""
    from ..stages.cutaways import Budget, suggest

    plan = suggest(Workspace.load(workspace), _provider(provider, model),
                   budget=Budget(every=every, humor=humor), brief=brief)
    for sl in plan.slots:
        console.print(f"slot {sl.slot} [{sl.kind}] {sl.anchor_clip_id} +{sl.offset:.1f}s "
                      f"{sl.duration:.1f}s: {sl.query!r} ({sl.reason})")
    for d in plan.dropped:
        console.print(f"[yellow]dropped:[/yellow] {d}")
    if not plan.slots:
        console.print("No cutaways needed.")


@media_app.command("fetch")
def media_fetch_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory"),
    library: Path | None = typer.Option(None, help="B-roll folder (default: <ws>/media/broll)"),
    provider: list[str] = typer.Option([], "--provider", "-p", help="pixabay | pexels | wikimedia"),
) -> None:
    """Download a licensed clip for every suggested slot and add it to edl_raw.json."""
    from ..stages.cutaways import fetch

    plan = fetch(Workspace.load(workspace), library=library, providers=tuple(provider) or None)
    for sl in plan.slots:
        if sl.overlay_id:
            console.print(f"slot {sl.slot} -> {sl.overlay_id}: {sl.chosen.title} "
                          f"({sl.chosen.provider}, {sl.chosen.license})")
        else:
            console.print(f"[yellow]slot {sl.slot}: {sl.note}[/yellow]")
    console.print(f"Next: `llmve cut-audit {workspace}`")


@media_app.command("place")
def media_place_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory"),
    file: Path = typer.Argument(..., exists=True, help="Video clip (downloaded or your own)"),
    at: str = typer.Option(..., "--at", help="Clip id in edl_raw.json the cutaway sits on"),
    offset: float = typer.Option(0.0, help="Seconds after the clip starts"),
    duration: float = typer.Option(3.0, help="Seconds on screen"),
    src_start: float = typer.Option(0.0, help="Start this far into the cutaway clip"),
    pip: bool = typer.Option(False, help="Picture-in-picture instead of full frame"),
) -> None:
    """Put one cutaway clip at an exact moment (adds an overlay to edl_raw.json)."""
    from ..schemas import EDL, ClipsManifest, Overlay
    from ..stages.cutaways import _register

    ws = Workspace.load(workspace)
    edl = Workspace.read(ws.edl_raw, EDL)
    if at not in {c.clip_id for c in edl.clips}:
        raise typer.BadParameter(f"no clip {at!r} in edl_raw.json")
    manifest = Workspace.read(ws.clips_manifest, ClipsManifest)
    info = _register(manifest, file.resolve(), file.stem)
    used = {o.overlay_id for o in edl.overlays}
    n = 1
    while f"p{n:03d}" in used:
        n += 1
    ov = Overlay(overlay_id=f"p{n:03d}", source_clip_id=info.clip_id,
                 source_file=info.source_file, src_start=src_start, anchor_clip_id=at,
                 offset=offset, duration=min(duration, info.duration - src_start),
                 zone="pip" if pip else "full")
    Workspace.write(ws.clips_manifest, manifest)
    Workspace.write(ws.edl_raw, edl.model_copy(update={"overlays": [*edl.overlays, ov]}))
    console.print(f"{ov.overlay_id}: {file.name} on {at} +{offset:g}s for {ov.duration:g}s. "
                  f"Next: `llmve finish {workspace}`")


music_app = typer.Typer(help="Background music, sound effects and voice-over.",
                        no_args_is_help=True)
app.add_typer(music_app, name="music")


@music_app.command("add")
def music_add_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory"),
    file: Path = typer.Argument(..., exists=True, help="Audio file"),
    at: str = typer.Option(..., "--at", help="Clip id in edl_raw.json the track starts on"),
    role: str = typer.Option("music", help="music | sfx | voiceover"),
    offset: float = typer.Option(0.0, help="Seconds after the clip starts"),
    duration: float | None = typer.Option(None, help="Seconds (default: to the end / whole file)"),
    volume: float | None = typer.Option(None, help="dB relative to the speech (music -10, sfx -4, voice-over 0)"),
    duck: bool = typer.Option(True, help="Music dips under speech (sfx and voice-over never do)"),
    loop: bool = typer.Option(False, help="Repeat a short bed to fill the duration"),
    src_start: float = typer.Option(0.0, help="Start this far into the file"),
) -> None:
    """Add a music bed, sound effect or voice-over to edl_raw.json; run cut-audit after."""
    from ..schemas import EDL, MusicTrack

    if role not in ("music", "sfx", "voiceover"):
        raise typer.BadParameter("--role is music, sfx or voiceover")
    ws = Workspace.load(workspace)
    edl = Workspace.read(ws.edl_raw, EDL)
    if at not in {c.clip_id for c in edl.clips}:
        raise typer.BadParameter(f"no clip {at!r} in edl_raw.json")
    used = {m.track_id for m in edl.music}
    n = len(used) + 1
    while f"a{n:03d}" in used:
        n += 1
    kw = {"volume_db": volume} if volume is not None else {}  # else the role's default
    track = MusicTrack(track_id=f"a{n:03d}", file=str(file.resolve()), role=role,
                       anchor_clip_id=at, offset=offset, duration=duration, src_start=src_start,
                       duck=duck and role == "music",
                       loop=loop, **kw)
    Workspace.write(ws.edl_raw, edl.model_copy(update={"music": [*edl.music, track]}))
    console.print(f"{track.track_id}: {role} {file.name} on {at} +{offset:g}s at "
                  f"{track.volume_db:g} dB{' (ducked)' if track.duck else ''}. "
                  f"Next: `llmve cut-audit {workspace}`")


@app.command("publish")
def publish_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory"),
    video: Path | None = typer.Option(None, "--video", help="Rendered file (default: last render)"),
    brief: str = typer.Option("", help="Channel, audience or keywords to aim for"),
    geocode: bool = typer.Option(False, help="Look up the place name for the GPS position "
                                             "(sends the coordinates to OpenStreetMap)"),
    use_location: bool = typer.Option(False, help="Let the copy mention the place (needs --geocode)"),
    thumbnails: bool = typer.Option(True, help="Make three thumbnails"),
    accent: str = typer.Option("#FFD400", help="Thumbnail accent colour"),
    provider: str = PROVIDER,
    model: str | None = typer.Option(None),
) -> None:
    """Write <video>.upload.md (+ .upload.json, 3 thumbnails) next to the rendered video."""
    from ..stages.publish import publish

    pkg = publish(Workspace.load(workspace), _provider(provider, model), video=video,
                  brief=brief, geocode=geocode, use_location=use_location,
                  thumbnails=thumbnails, accent=accent)
    console.print(f"[bold]{pkg.titles[0]}[/bold]")
    for t in pkg.thumbnails:
        console.print(f"  thumbnail {t.style}: {t.path} ({t.text!r}, {t.renderer})")
    for w in pkg.warnings:
        console.print(f"  [yellow]warning:[/yellow] {w}")
    typer.echo(str(Path(pkg.video).with_suffix(".upload.md")))


class _ReviewGroup(TyperGroup):
    """`llmve review <ws>` builds the sheet; `llmve review approve|reject <ws>` decides.

    A group-level positional argument would swallow the subcommand name
    (`review approve <ws>` parsed "approve" as the workspace), so a bare
    workspace is routed to the hidden `build` subcommand instead.
    """

    def parse_args(self, ctx, args):
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["build", *args]
        return super().parse_args(ctx, args)


review_app = typer.Typer(help="Pre-render review gate: `llmve review <ws>`, then "
                              "`llmve review approve <ws>` or `llmve review reject <ws>`.",
                         cls=_ReviewGroup, no_args_is_help=True)
app.add_typer(review_app, name="review")


@review_app.command("build", hidden=True)
def review_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory"),
) -> None:
    """Build the review contact sheet for the current cut (status: pending)."""
    from ..stages.review import build_review

    ws = Workspace.load(workspace)
    review = build_review(ws)
    for sheet in review.sheets:
        typer.echo(str(ws.root / sheet))
    for w in review.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")
    console.print("Look at the sheet(s), then `llmve review approve <workspace>` "
                  "or `llmve review reject <workspace> --notes ...`.")


@review_app.command("approve")
def review_approve(workspace: Path, notes: str = typer.Option("", help="Reviewer notes")) -> None:
    """Approve the current review so `render` may run."""
    from ..stages.review import decide

    console.print(f"review {decide(Workspace.load(workspace), True, notes).status}")


@review_app.command("reject")
def review_reject(workspace: Path, notes: str = typer.Option("", help="What to change")) -> None:
    """Reject the current review; re-plan or edit, then review again."""
    from ..stages.review import decide

    console.print(f"review {decide(Workspace.load(workspace), False, notes).status}")


ENCODER = typer.Option("auto", help="auto | h264_nvenc | h264_videotoolbox | libx264")


@app.command("render")
def render_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory printed by `watch`"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output .mp4 path"),
    encoder: str = ENCODER,
    draft: bool = typer.Option(False, help="Fast low-quality encode"),
    captions: bool = typer.Option(True, help="Burn captions from the transcript"),
    skip_review: bool = typer.Option(False, help="Render without an approved review"),
) -> None:
    """Render the approved cut in one encode (two-pass -14 LUFS) and write qc_report.json."""
    from ..stages.render import render
    from ..stages.review import ReviewRequired

    try:
        res = render(Workspace.load(workspace), out=out, encoder=encoder,
                     quality="draft" if draft else "high", burn_captions=captions,
                     require_review=not skip_review)
    except ReviewRequired as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2) from e
    _print_qc(res.qc)
    typer.echo(str(res.output))
    if not res.qc.passed:
        raise typer.Exit(1)


def _print_qc(qc) -> None:
    status = "[green]QC PASSED[/green]" if qc.passed else "[red]QC FAILED[/red]"
    console.print(f"{status} {qc.width}x{qc.height}@{qc.fps:g} {qc.duration:.2f}s "
                  f"A/V offset {qc.av_offset * 1000:.0f} ms, {qc.integrated_lufs} LUFS ({qc.encoder})")
    for p in qc.problems:
        console.print(f"  - {p}")
    for w in qc.warnings:
        console.print(f"  [yellow]warning:[/yellow] {w}")
    if qc.chapters:
        console.print(f"  {len(qc.chapters)} chapters (description block: "
                      f"{Path(qc.output).with_suffix('.chapters.txt')})")
    if qc.captions_srt:
        console.print(f"  captions: {qc.captions_srt}")


@app.command("auto")
def auto_cmd(
    inputs: list[str] = typer.Argument(..., help="Video file(s), folder(s) or glob(s)"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="What the edit should be"),
    target: str = typer.Option("youtube", "--target", "-t", help=TARGETS),
    target_duration: str | None = typer.Option(None, help="e.g. 10m, 45s (default: preset)"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output .mp4 (default: ./<input>-<target>.mp4)"),
    pacing: str = typer.Option("auto", help="auto | tight | natural | off"),
    captions: str = typer.Option("auto", help="auto (burn for vertical, .srt for YouTube) | burn | srt"),
    chapters: bool = typer.Option(True, help="Name YouTube chapters automatically"),
    music: str = typer.Option("auto", help="auto (licensed search) | off | path/to/track.mp3"),
    music_query: str | None = typer.Option(None, help='Mood for the music search, e.g. "lofi chill"'),
    cutaways: bool = typer.Option(True, help="Add a few licensed B-roll/funny clips where they help"),
    humor: bool = typer.Option(True, help="Allow funny cutaways"),
    graphics: bool = typer.Option(True, help="Chapter cards / lower third (needs Node 22+)"),
    look: str = typer.Option("none", help="One colour look: none | natural | warm | golden-hour "
                             "| teal-orange | cinematic | vivid | matte | bw"),
    look_strength: float = typer.Option(0.5, min=0.0, max=1.0, help="0.4-0.6 reads premium"),
    match_color: bool = typer.Option(False, help="Correct clips towards each other (exposure, "
                                     "white balance, saturation) so they look like one video"),
    name: str | None = typer.Option(None, help='Lower third, e.g. "Jane Doe|Founder"'),
    broll_library: Path | None = typer.Option(None, help="Your own B-roll folder to use first"),
    review: bool = typer.Option(False, "--review", help="Stop at the review sheet instead of auto-approving"),
    publish: bool = typer.Option(True, help="Write <out>.upload.md and three thumbnails"),
    geocode: bool = typer.Option(False, help="Place name from GPS (sends coordinates to OpenStreetMap)"),
    use_location: bool = typer.Option(False, help="Let the upload copy mention the place"),
    draft: bool = typer.Option(False, help="Fast low-quality encode"),
    encoder: str = ENCODER,
    detector: str = typer.Option("auto", help="auto | mediapipe | yolo | none"),
    asr_model: str = typer.Option("distil-large-v3", help="faster-whisper model"),
    language: str | None = typer.Option(None, help="Force ASR language (e.g. hi)"),
    chunk_asr_over: float = typer.Option(10.0, help="Chunked ASR above this many minutes (0 = off)"),
    provider: str = PROVIDER,
    model: str | None = typer.Option(None),
    home: Path | None = HOME,
) -> None:
    """The whole edit in one command: decides, auto-approves, renders and writes the upload file."""
    from ..stages.auto import AutoOptions, StepFailed, auto_edit
    from ..stages.durations import parse_duration

    if pacing not in ("auto", "tight", "natural", "off"):
        raise typer.BadParameter("--pacing is auto, tight, natural or off")
    if captions not in ("auto", "burn", "srt"):
        raise typer.BadParameter("--captions is auto, burn or srt")
    opts = AutoOptions(
        target=target, target_duration=parse_duration(target_duration) if target_duration else None,
        out=out, pacing=pacing, captions=captions, chapters=chapters, music=music,
        music_query=music_query, cutaways=cutaways, humor=humor, graphics=graphics, name=name,
        look=look, look_strength=look_strength, match_color=match_color,
        broll_library=broll_library, approve=not review, publish=publish, geocode=geocode,
        use_location=use_location, draft=draft, encoder=encoder, detector=detector,
        asr_model=asr_model, language=language, chunk_asr_over=chunk_asr_over or None)
    icons = {"done": "[green]done[/green]", "skipped": "[yellow]skip[/yellow]",
             "failed": "[red]fail[/red]"}

    def show(rep):
        console.print(f"{icons[rep.status]} {rep.step:<14} {rep.detail} ({rep.seconds:g}s)")

    try:
        rep = auto_edit(inputs, prompt, _provider(provider, model), opts, home=home, on_step=show)
    except StepFailed as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    for w in rep.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")
    if not rep.approved:
        console.print(f"Review the sheet, then `llmve review approve {rep.workspace}` and "
                      f"`llmve render {rep.workspace}`.")
        raise typer.Exit(2)
    console.print(f"[bold]{rep.output}[/bold]" + (f"\nupload file: {rep.upload}" if rep.upload else ""))


@app.command("finish")
def finish_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory (from `llmve auto`)"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output .mp4 (default: the last one)"),
    target: str | None = typer.Option(None, "--target", "-t", help="Default: the EDL's platform"),
    review: bool = typer.Option(False, "--review", help="Stop at the review sheet"),
    publish: bool = typer.Option(True, help="Rewrite <out>.upload.md and thumbnails"),
    draft: bool = typer.Option(False, help="Fast low-quality encode"),
    encoder: str = ENCODER,
    detector: str = typer.Option("auto", help="auto | mediapipe | yolo | none"),
    provider: str = PROVIDER,
    model: str | None = typer.Option(None),
) -> None:
    """After editing edl_raw.json (or `llmve music/graphics/media` changes): audit, review,
    render and publish again, auto-approved. No new transcription or planning."""
    from ..schemas import EDL
    from ..stages.auto import AutoOptions, StepFailed, finish_edit

    ws = Workspace.load(workspace)
    platform = target or Workspace.read(ws.edl_raw, EDL).target_platform
    opts = AutoOptions(target=platform, out=out, approve=not review, publish=publish,
                       draft=draft, encoder=encoder, detector=detector)

    def show(rep):
        icon = {"done": "[green]done[/green]", "skipped": "[yellow]skip[/yellow]",
                "failed": "[red]fail[/red]"}[rep.status]
        console.print(f"{icon} {rep.step:<14} {rep.detail} ({rep.seconds:g}s)")

    try:
        rep = finish_edit(ws, _provider(provider, model), opts, on_step=show)
    except StepFailed as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    for w in rep.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")
    if not rep.approved:
        raise typer.Exit(2)
    console.print(f"[bold]{rep.output}[/bold]" + (f"\nupload file: {rep.upload}" if rep.upload else ""))


def _privacy(value: str | None) -> str:
    from ..agent.setup import load_profile

    privacy = value or load_profile().privacy
    if privacy not in ("stills", "text-only"):
        raise typer.BadParameter("--privacy is stills or text-only")
    return privacy


@app.command("mcp")
def mcp_cmd(
    privacy: str | None = typer.Option(None, help="stills | text-only (default: profile)"),
    home: Path | None = HOME,
) -> None:
    """Serve the editor agent's tools over MCP (stdio) for Claude Code / `claude -p`."""
    from ..agent.mcp_server import serve
    from ..agent.tools import Context

    serve(Context(privacy=_privacy(privacy), home=home, jobs="process"))


@app.command("setup")
def agent_setup_cmd() -> None:
    """Record what this machine can do (ffmpeg, encoders, Node, MCP, Claude CLI, disk)."""
    from ..agent.setup import config_dir, machine_record

    rec = machine_record()
    for k, v in rec.items():
        if k != "warnings":
            console.print(f"{k}: {v}")
    for w in rec["warnings"]:
        console.print(f"[yellow]warning:[/yellow] {w}")
    console.print(f"saved {config_dir() / 'machine.json'}")


@app.command("profile")
def agent_profile_cmd(
    set_: list[str] = typer.Option([], "--set", help="key=value (channel, audience, "
                                   "languages=en,hi, default_look, privacy, location_lookup=true, "
                                   "footage_dir, output_dir)"),
    like: list[str] = typer.Option([], "--like"),
    dislike: list[str] = typer.Option([], "--dislike"),
    forget: list[str] = typer.Option([], "--forget", help="remove a like/dislike"),
) -> None:
    """Show or update your style profile (the agent reads it before every edit)."""
    from ..agent.setup import profile_path, update_profile

    fields = {}
    for item in set_:
        if "=" not in item:
            raise typer.BadParameter(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        fields[k.strip()] = [x.strip() for x in v.split(",")] if k.strip() == "languages" else v
    try:
        prof = update_profile(set_fields=fields, add_likes=like, add_dislikes=dislike,
                              remove=forget)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    for k, v in prof.model_dump().items():
        console.print(f"{k}: {v}")
    console.print(f"({profile_path()})")


@app.command("mcp-config")
def agent_mcp_config_cmd(
    privacy: str | None = typer.Option(None, help="stills | text-only"),
) -> None:
    """Print the .mcp.json that connects Claude Code (or claude -p) to the tools."""
    from ..agent.setup import mcp_config

    repo = Path(__file__).resolve().parents[2]
    typer.echo(json.dumps(mcp_config(repo, privacy), indent=2))


@app.command("tools")
def agent_tools_cmd() -> None:
    """List the agent's tools."""
    from ..agent.tools import REGISTRY

    for t in REGISTRY.values():
        console.print(f"[bold]{t.name}[/bold] ({t.kind}): {' '.join(t.description.split())}")


@app.command("tool")
def agent_call_cmd(
    name: str = typer.Argument(..., help="tool name"),
    args: str = typer.Argument("{}", help="JSON arguments"),
    privacy: str | None = typer.Option(None, help="stills | text-only"),
    home: Path | None = HOME,
) -> None:
    """Call one tool from the terminal (JSON in, JSON out) - same tools as over MCP."""
    from ..agent.tools import Context, call

    try:
        parsed = json.loads(args)
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"args must be JSON: {e}") from e
    res = call(name, parsed, Context(privacy=_privacy(privacy), home=home))
    typer.echo(json.dumps(res, indent=2, default=str))
    if "error" in res:
        raise typer.Exit(1)


@app.command("run")
def run_cmd(
    inputs: list[str] = typer.Argument(..., help="Video file(s), folder(s) or glob(s)"),
    prompt: str = typer.Option(..., "--prompt", "-p"),
    target: str = typer.Option("youtube", "--target", "-t", help=TARGETS),
    target_duration: str | None = typer.Option(None, help="e.g. 10m, 90s, 1:30"),
    provider: str = PROVIDER,
    model: str | None = typer.Option(None),
    asr_model: str = typer.Option("distil-large-v3"),
    language: str | None = typer.Option(None),
    chunk_asr_over: float | None = typer.Option(
        None, help="Chunked/parallel ASR for clips longer than this many minutes (e.g. 10)"),
    asr_workers: int = typer.Option(0, help="ASR worker processes for chunked ASR (0 = auto)"),
    encoder: str = ENCODER,
    draft: bool = typer.Option(False),
    out: Path | None = typer.Option(None, "--out", "-o"),
    detector: str = typer.Option("auto", help="auto | mediapipe | yolo | none"),
    broll_library: Path | None = typer.Option(None, help="Indexed B-roll folder to match from"),
    approve: bool = typer.Option(False, "--approve", help="Approve the review sheet and render"),
    home: Path | None = HOME,
) -> None:
    """watch -> plan -> b-roll -> cut-audit -> geometry -> review [-> render with --approve]."""
    from ..llm import get_provider
    from ..stages.asr import FasterWhisperTranscriber
    from ..stages.cut_audit import run_cut_audit
    from ..stages.durations import parse_duration
    from ..stages.plan import plan
    from ..stages.render import render
    from ..stages.watch import watch as run_watch

    ws = _open(inputs, home)
    console.print(f"Workspace: {ws.root}")
    run_watch(ws, transcriber=FasterWhisperTranscriber(asr_model), language=language,
              chunking=_chunking(chunk_asr_over, asr_workers))
    dur = parse_duration(target_duration) if target_duration else None
    llm = get_provider(provider, model)
    plan(ws, prompt, llm, target_platform=target, target_duration=dur)
    from ..stages.broll import index_library, match_broll

    if broll_library is not None:
        index_library(broll_library)
    match_broll(ws, broll_library, provider=llm if broll_library else None)
    _, report = run_cut_audit(ws)
    if not report.passed:
        for v in report.unresolved:
            console.print(f"[red]{v.kind}[/red] {v.clip_id or ''} {v.message}")
        console.print("Cut audit failed; fix edl_raw.json or re-plan, then run `llmve cut-audit`.")
        raise typer.Exit(1)
    from ..stages.detectors import get_detector
    from ..stages.geometry import run_geometry
    from ..stages.review import build_review, decide

    det = get_detector(detector)
    try:
        run_geometry(ws, detector=det)
    finally:
        det.close()
    review = build_review(ws)
    for sheet in review.sheets:
        console.print(f"review sheet: {ws.root / sheet}")
    for w in review.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")
    if not approve:
        console.print(f"Review the sheet, then `llmve review approve {ws.root}` and "
                      f"`llmve render {ws.root}`.")
        raise typer.Exit(2)
    decide(ws, True, "approved via llmve run --approve")
    res = render(ws, out=out, encoder=encoder, quality="draft" if draft else "high",
                 require_review=True)
    _print_qc(res.qc)
    typer.echo(str(res.output))


if __name__ == "__main__":
    app()
