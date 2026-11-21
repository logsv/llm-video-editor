"""``llmve``: stage-by-stage CLI over the checkpointed pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from typer.core import TyperGroup
from rich.table import Table

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
