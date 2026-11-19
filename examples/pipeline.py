"""Drive the checkpointed pipeline from Python.

    uv run python examples/pipeline.py ./raw_clips "10 minute YouTube edit: the key ideas" youtube

Stops after the review sheet; approve it with
``uv run llmve review approve <workspace>`` and then ``uv run llmve render <workspace>``.
"""

from __future__ import annotations

import sys

from llm_video_editor.llm import get_provider
from llm_video_editor.stages.broll import match_broll
from llm_video_editor.stages.cut_audit import run_cut_audit
from llm_video_editor.stages.geometry import run_geometry
from llm_video_editor.stages.plan import plan
from llm_video_editor.stages.review import build_review
from llm_video_editor.stages.watch import watch
from llm_video_editor.workspace import Workspace


def main(inputs: str, prompt: str, target: str = "youtube") -> None:
    ws = Workspace.open(inputs)
    print("workspace:", ws.root)

    res = watch(ws)  # faster-whisper distil-large-v3, scene/motion scores, contact sheets
    for c in res.manifest.clips:
        print(f"  {c.clip_id} {c.kind:12} {c.duration:7.1f}s  speech {c.speech_ratio:.0%}")

    asset_plan, edl = plan(ws, prompt, get_provider("claude-cli"), target_platform=target)
    print(f"plan: {len(edl.clips)} cuts, target {asset_plan.target_duration:.0f}s")

    match_broll(ws)  # session B-roll only; pass a library folder to use one
    _, report = run_cut_audit(ws)
    print("cut-audit:", "PASSED" if report.passed else "FAILED")
    if not report.passed:
        for v in report.unresolved:
            print("  ", v.kind, v.message)
        return

    run_geometry(ws)
    review = build_review(ws)
    print("review sheets:", *[ws.root / s for s in review.sheets], sep="\n  ")


if __name__ == "__main__":
    main(*sys.argv[1:])
