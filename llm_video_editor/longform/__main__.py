"""CLI: python -m llm_video_editor.longform {ingest,render} ..."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="llm-longform", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest", help="probe + transcribe + transcript.md + contact sheet")
    i.add_argument("source", help="the raw recording")
    i.add_argument("project", help="project folder to create/use")
    i.add_argument("--model", default="small", help="faster-whisper model: small, medium, large-v3 (default small)")
    i.add_argument("--language", default=None, help="e.g. en, hi; auto-detect if omitted")
    i.add_argument("--keywords", default=None, help="names/terms to bias ASR, e.g. 'Claude, LangGraph, Ollama'")
    i.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    i.add_argument("--words-json", default=None, help="reuse an existing words.json instead of transcribing")
    i.add_argument("--no-sheet", action="store_true", help="skip the contact sheet")

    r = sub.add_parser("render", help="render edl.json")
    r.add_argument("project")
    r.add_argument("--edl", default=None, help="EDL path (default <project>/edl.json)")
    r.add_argument("--preview", action="store_true", help="fast 540p review copy, no loudness pass")
    r.add_argument("--vertical", action="store_true", help="9:16 1080x1920 (Shorts) using per-segment cx")
    r.add_argument("--height", type=int, default=None, help="output height (default: source height)")
    r.add_argument("--encoder", default="x264", choices=["x264", "nvenc", "videotoolbox"])
    r.add_argument("--lufs", type=float, default=-14.0)
    r.add_argument("--output", "-o", default=None)
    r.add_argument("--dry-run", action="store_true", help="print the cut plan only")

    a = p.parse_args(argv)
    if a.cmd == "ingest":
        from .ingest import ingest
        ingest(a.source, a.project, a.model, a.language, a.keywords, a.words_json, a.device, not a.no_sheet)
    else:
        from .render import render
        rep = render(a.project, a.edl, a.preview, a.vertical, a.height, a.encoder, a.lufs,
                     output=a.output, dry_run=a.dry_run)
        if a.dry_run:
            print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
