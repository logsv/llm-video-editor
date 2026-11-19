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


if __name__ == "__main__":
    app()
