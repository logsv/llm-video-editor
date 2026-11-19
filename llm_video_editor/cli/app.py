"""``llmve``: stage-by-stage CLI over the checkpointed pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
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


@app.command()
def watch(
    inputs: list[str] = typer.Argument(..., help="Video file(s), folder(s) or glob(s)"),
    asr_model: str = typer.Option("distil-large-v3", help="faster-whisper model"),
    language: str | None = typer.Option(None, help="Force ASR language (e.g. en)"),
    home: Path | None = HOME,
) -> None:
    """Transcribe, score scenes, classify A-roll/B-roll and build overview sheets."""
    from ..stages.asr import FasterWhisperTranscriber
    from ..stages.watch import watch as run_watch

    ws = _open(inputs, home)
    res = run_watch(ws, transcriber=FasterWhisperTranscriber(asr_model), language=language)
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


PROVIDER = typer.Option("claude-cli", help="Planner: claude-cli | ollama | anthropic | openai")


@app.command("plan")
def plan_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory printed by `watch`"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="What the edit should be"),
    target: str = typer.Option("youtube", "--target", "-t", help="youtube | reels | tiktok ..."),
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
                  f"{report.duration_after:.1f}s")
    if not report.passed:
        raise typer.Exit(1)


ENCODER = typer.Option("auto", help="auto | h264_nvenc | h264_videotoolbox | libx264")


@app.command("render")
def render_cmd(
    workspace: Path = typer.Argument(..., help="Workspace directory printed by `watch`"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output .mp4 path"),
    encoder: str = ENCODER,
    draft: bool = typer.Option(False, help="Fast low-quality encode"),
    captions: bool = typer.Option(True, help="Burn captions from the transcript"),
) -> None:
    """Render edl_audited.json in one encode (two-pass -14 LUFS) and write qc_report.json."""
    from ..stages.render import render

    res = render(Workspace.load(workspace), out=out, encoder=encoder,
                 quality="draft" if draft else "high", burn_captions=captions)
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


@app.command("run")
def run_cmd(
    inputs: list[str] = typer.Argument(..., help="Video file(s), folder(s) or glob(s)"),
    prompt: str = typer.Option(..., "--prompt", "-p"),
    target: str = typer.Option("youtube", "--target", "-t"),
    target_duration: str | None = typer.Option(None, help="e.g. 10m, 90s, 1:30"),
    provider: str = PROVIDER,
    model: str | None = typer.Option(None),
    asr_model: str = typer.Option("distil-large-v3"),
    language: str | None = typer.Option(None),
    encoder: str = ENCODER,
    draft: bool = typer.Option(False),
    out: Path | None = typer.Option(None, "--out", "-o"),
    home: Path | None = HOME,
) -> None:
    """watch -> plan -> cut-audit -> render, stopping if the cut audit fails."""
    from ..llm import get_provider
    from ..stages.asr import FasterWhisperTranscriber
    from ..stages.cut_audit import run_cut_audit
    from ..stages.durations import parse_duration
    from ..stages.plan import plan
    from ..stages.render import render
    from ..stages.watch import watch as run_watch

    ws = _open(inputs, home)
    console.print(f"Workspace: {ws.root}")
    run_watch(ws, transcriber=FasterWhisperTranscriber(asr_model), language=language)
    dur = parse_duration(target_duration) if target_duration else None
    plan(ws, prompt, get_provider(provider, model), target_platform=target, target_duration=dur)
    _, report = run_cut_audit(ws)
    if not report.passed:
        for v in report.unresolved:
            console.print(f"[red]{v.kind}[/red] {v.clip_id or ''} {v.message}")
        console.print("Cut audit failed; fix edl_raw.json or re-plan, then run `llmve cut-audit`.")
        raise typer.Exit(1)
    res = render(ws, out=out, encoder=encoder, quality="draft" if draft else "high")
    _print_qc(res.qc)
    typer.echo(str(res.output))


if __name__ == "__main__":
    app()
