"""`llmve auto`: the whole edit in one command (fakes for ASR and the planner)."""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from llm_video_editor import llm as llm_pkg
from llm_video_editor.cli.app import app
from llm_video_editor.ffmpeg.runner import ffmpeg, ffprobe_json
from llm_video_editor.schemas import EDL, Review
from llm_video_editor.stages import asr as asr_mod
from llm_video_editor.workspace import Workspace
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeProvider, FakeTranscriber
from tests.fixtures.make_fixture import make_clip

pytestmark = [requires_ffmpeg, pytest.mark.integration]
TOPICS = ["Why budget mics matter.", "How I tested them.", "The surprising winner."]
# continuous speech like a real recording: 45 sentences of ~3 s, short pauses between
SCRIPT = [(i * 3.3 + 0.3, f"{TOPICS[i // 15].split()[0]} point {i} matters a lot here today.")
          for i in range(45)]


def responder(schema, prompt):
    name = schema.__name__
    if name == "ReduceResult":
        cid = re.findall(r"chunk (v[0-9a-f]{8})#", prompt)[0]
        return schema.model_validate({"selections": [
            {"clip_id": cid, "first_segment": a, "last_segment": a + 14, "purpose": p,
             "priority": 1} for a, p in ((0, "intro"), (15, "test"), (30, "winner"))]})
    if name == "_Chapters":
        return schema.model_validate({"chapters": [
            {"clip_id": "c001", "title": "Why budget mics"},
            {"clip_id": "c002", "title": "How I tested"},
            {"clip_id": "c003", "title": "The winner"}]})
    if name == "_Picks":
        return schema.model_validate({"cutaways": []})
    if name == "_Copy":
        return schema.model_validate({
            "titles": ["I Tested Budget Mics So You Don't Have To"], "hook": "Budget mics, tested.",
            "body": "Three parts.", "tags": ["budget mic"], "hashtags": ["mics"],
            "category": "Science & Technology", "pinned_comment": "Which mic?",
            "thumbnails": [{"clip_id": f"c00{i + 1}", "segment_id": i * 15 + 1,
                            "text": t, "style": s, "why": "x"}
                           for i, (t, s) in enumerate([("Cheap beats pricey?", "question"),
                                                       ("The test", "bold"),
                                                       ("Winner", "reveal")])]})
    raise AssertionError(f"unexpected schema {name}")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    make_clip(raw / "talk.mp4", duration=152, size=(640, 360), fps=30, audio="sine")
    monkeypatch.setattr(asr_mod, "FasterWhisperTranscriber",
                        lambda *a, **k: FakeTranscriber(SCRIPT))
    monkeypatch.setattr(llm_pkg, "get_provider", lambda *a, **k: FakeProvider(responder))
    bed = tmp_path / "bed.mp3"
    ffmpeg("-f", "lavfi", "-i", "sine=f=220:d=40", "-ac", "2", str(bed))
    return raw, bed


def test_one_command_edits_approves_renders_and_publishes(setup, tmp_path):
    raw, bed = setup
    out = tmp_path / "out" / "final.mp4"
    res = CliRunner().invoke(app, [
        "auto", str(raw), "-p", "3 minute review of budget mics", "-t", "youtube",
        "-o", str(out), "--music", str(bed), "--no-graphics", "--draft", "--detector", "none",
        "--home", str(tmp_path / "home")])
    assert res.exit_code == 0, res.output
    assert out.exists() and out.with_suffix(".upload.md").exists()
    assert len(list(out.parent.glob("final.thumb*.jpg"))) == 3

    report = json.loads(out.with_suffix(".auto_report.json").read_text())
    status = {s["step"]: s["status"] for s in report["steps"]}
    assert status == {"watch": "done", "plan": "done", "pacing": "done", "captions": "done",
                      "colour": "skipped",
                      "chapters": "done", "music": "done", "cutaways": "skipped",
                      "graphics": "skipped", "cut-audit": "done", "geometry": "done",
                      "review": "done", "render": "done", "publish": "done"}
    assert report["approved"] is True

    ws = Workspace.load(next((tmp_path / "home").iterdir()))
    review = Workspace.read(ws.review_dir / "review.json", Review)
    assert review.status == "approved" and "llmve auto" in review.notes
    edl = Workspace.read(ws.edl_audited, EDL)
    assert [c.chapter for c in edl.clips if c.chapter] == [
        "Why budget mics", "How I tested", "The winner"]
    assert edl.pacing.tighten and edl.pacing.punch_in == "off"  # 360p source: no zoom headroom
    assert edl.captions.enabled is False  # YouTube long-form: sidecar .srt
    assert [m.track_id for m in edl.music] == ["auto-music"] and edl.music[0].duck
    titles = [c["tags"]["title"] for c in ffprobe_json(out, "-show_chapters")["chapters"]]
    assert titles == ["Why budget mics", "How I tested", "The winner"]
    assert "Chapters\n0:00 Why budget mics" in out.with_suffix(".upload.md").read_text()
    for step in ("watch", "plan", "cut-audit", "review", "render", "publish"):
        assert re.search(rf"done\s+{step}", res.output), step


def test_review_flag_stops_before_render(setup, tmp_path):
    raw, bed = setup
    out = tmp_path / "final.mp4"
    res = CliRunner().invoke(app, [
        "auto", str(raw), "-p", "review", "-o", str(out), "--music", "off", "--review",
        "--no-graphics", "--no-cutaways", "--detector", "none", "--home", str(tmp_path / "h")])
    assert res.exit_code == 2, res.output
    assert not out.exists() and "llmve review approve" in res.output


def test_finish_reapplies_a_note_without_replanning(setup, tmp_path, monkeypatch):
    """The refine loop: auto, then a change to edl_raw.json, then `llmve finish`."""
    raw, bed = setup
    out = tmp_path / "final.mp4"
    home = tmp_path / "home"
    args = ["auto", str(raw), "-p", "3 minute review of budget mics", "-o", str(out),
            "--music", str(bed), "--no-graphics", "--no-publish", "--draft",
            "--detector", "none", "--home", str(home)]
    assert CliRunner().invoke(app, args).exit_code == 0
    ws = Workspace.load(next(home.iterdir()))
    first = ffprobe_json(out)["format"]["duration"]

    # note: "drop the music and the last section"
    edl = Workspace.read(ws.edl_raw, EDL)
    Workspace.write(ws.edl_raw, edl.model_copy(update={"music": [], "clips": edl.clips[:2]}))
    planned = []
    monkeypatch.setattr(llm_pkg, "get_provider",
                        lambda *a, **k: FakeProvider(lambda s, p: planned.append(s) or
                                                     responder(s, p)))
    res = CliRunner().invoke(app, ["finish", str(ws.root), "--no-publish", "--draft",
                                   "--detector", "none"])
    assert res.exit_code == 0, res.output
    assert planned == []  # no planning, no transcription: only the downstream stages
    assert float(ffprobe_json(out)["format"]["duration"]) < float(first) * 0.8
    assert Workspace.read(ws.edl_audited, EDL).music == []
    assert "approved automatically" in res.output
