"""Agent mode P0: a whole edit made only through tool calls, the agent choosing every
parameter (here the test plays the agent)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_video_editor.agent import tools
from llm_video_editor.agent.tools import Context, call
from llm_video_editor.ffmpeg.runner import ffmpeg, ffprobe_json
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeTranscriber
from tests.fixtures.make_fixture import make_clip

pytestmark = [requires_ffmpeg, pytest.mark.integration]
SCRIPT = [(0.3, "So today I ride from Bhopal."), (2.3, "The road is long and empty."),
          (4.4, "This is the best part.")]


@pytest.fixture
def footage(tmp_path):
    raw = tmp_path / "trip"
    make_clip(raw / "GX010042.MP4", duration=6, size=(1280, 720), fps=30)
    make_clip(raw / "GX020042.MP4", duration=6, size=(1280, 720), fps=60)
    make_clip(raw / "zz_extra.mp4", duration=4, size=(1280, 720), audio=None)
    return raw


def ctx(tmp_path, privacy="stills"):
    return Context(privacy=privacy, home=tmp_path / "home",
                   transcriber_factory=lambda model: FakeTranscriber(SCRIPT))


def ok(res):
    assert "error" not in res, res
    return res


def test_full_edit_through_tools_only(footage, tmp_path):
    c = ctx(tmp_path)
    opened = ok(call("open_project", {"inputs": [str(footage)], "first": 2}, c))
    ws = opened["workspace"]
    assert opened["clips"] == 2  # only the first N clips
    assert opened["split_recordings"] == {"gopro-X0042": ["GX010042.MP4", "GX020042.MP4"]}
    issues = {f["name"]: f["issues"] for f in opened["files"]}
    assert any("slow motion" in i for i in issues["GX020042.MP4"])
    assert any("split across 2 files" in i for i in issues["GX010042.MP4"])

    clips = ok(call("analyze", {"workspace": ws}, c))["clips"]
    assert len(clips) == 2 and all(x["kind"] == "talking_head" for x in clips)
    a, b = clips[0]["clip_id"], clips[1]["clip_id"]
    ov = ok(call("overview", {"workspace": ws}, c))
    assert ov["clips"][0]["issues"] and ov["total_seconds"] == pytest.approx(12, abs=0.2)
    segs = ok(call("transcript", {"workspace": ws, "clip_id": a}, c))["segments"]
    assert [s["id"] for s in segs] == [0, 1, 2]
    sm = ok(call("speech_map", {"workspace": ws, "clip_id": a}, c))
    assert sm["words"][0]["text"] == "So" and sm["pauses"]
    fr = ok(call("frames", {"workspace": ws, "clip_id": a, "start": 0, "end": 4}, c))
    assert fr["sheets"] and c.images_out == [s["path"] for s in fr["sheets"]]

    # the agent builds its cut: hook from clip b first, then clip a, tight pacing
    ok(call("new_edit", {"workspace": ws, "target_platform": "youtube",
                         "reason": "long-form trip video"}, c))
    r = ok(call("batch", {"workspace": ws, "reason": "first cut: hook then the ride",
                          "operations": [
                              {"op": "add_clip", "params": {"source_clip_id": b, "start": 4.3,
                                                            "end": 5.6, "chapter": "Hook"}},
                              {"op": "add_clip", "params": {"source_clip_id": a, "start": 0.2,
                                                            "end": 4.3, "chapter": "The road"}},
                              {"op": "set_pacing", "params": {"tighten": True,
                                                              "max_gap": 0.5}}]}, c))
    assert r["version"] == 2 and "added c001, c002" in r["summary"]
    st = ok(call("edit_state", {"workspace": ws}, c))
    assert [x["chapter"] for x in st["clips"]] == ["Hook", "The road"]
    assert st["clips"][1]["out_start"] == pytest.approx(1.3, abs=0.01)

    # audit reports, never fixes
    au = ok(call("audit", {"workspace": ws}, c))
    assert all(not f.get("fixed") for f in au["findings"] if f["kind"] != "out_of_bounds")
    assert ok(call("edit_state", {"workspace": ws}, c))["version"] == 2

    rv = ok(call("review", {"workspace": ws, "detector": "none"}, c))
    assert rv["sheets"] and c.images_out == rv["sheets"]
    out = tmp_path / "out" / "trip.mp4"
    rd = ok(call("render", {"workspace": ws, "out": str(out), "draft": True,
                            "reason": "cuts land on pauses, hook reads well"}, c))
    assert rd["qc"]["passed"], rd["qc"]["problems"]
    chapters = [x["tags"]["title"] for x in ffprobe_json(out, "-show_chapters")["chapters"]]
    assert chapters == ["Hook", "The road"]

    pub = ok(call("publish", {"workspace": ws, "titles": ["Bhopal to Khajuraho, Solo"],
                              "hook": "A solo ride.", "body": "The road and the temples.",
                              "tags": ["bike trip"], "hashtags": ["biketrip"],
                              "category": "Travel & Events",
                              "thumbnails": [{"clip_id": "c002", "segment_id": 1,
                                              "text": "Empty road", "style": "bold",
                                              "why": "scale"}]}, c))
    assert Path(pub["upload_md"]).exists() and len(pub["thumbnails"]) == 1

    log = [json.loads(x) for x in (Path(ws) / "agent_log.jsonl").read_text().splitlines()]
    tools_used = [e["tool"] for e in log]
    assert tools_used[:2] == ["import", "batch"] and "render" in tools_used
    assert any(e["reason"] == "first cut: hook then the ride" for e in log)
    md = (Path(ws) / "decisions.md").read_text()
    assert "cuts land on pauses" in md


def test_privacy_errors_and_conflicts(footage, tmp_path):
    c = ctx(tmp_path, privacy="text-only")
    ws = ok(call("open_project", {"inputs": [str(footage)], "first": 1}, c))["workspace"]
    a = ok(call("analyze", {"workspace": ws}, c))["clips"][0]["clip_id"]
    res = call("frames", {"workspace": ws, "clip_id": a, "start": 0, "end": 2}, c)
    assert "text-only" in res["error"] and c.images_out == []

    ok(call("new_edit", {"workspace": ws, "target_platform": "youtube", "reason": "start"}, c))
    assert "not inside source" in call("add_clip", {"workspace": ws, "source_clip_id": a,
                                                    "start": 1, "end": 99,
                                                    "reason": "too long"}, c)["error"]
    assert "bad arguments" in call("add_clip", {"workspace": ws, "reason": "x"}, c)["error"]
    assert "at least 3 characters at reason" in call(
        "remove_clip", {"workspace": ws, "clip_id": "c001", "reason": ""}, c)["error"]
    assert "unknown tool" in call("make_it_good", {}, c)["error"]
    assert "operation 1 (add_clip)" in call("batch", {
        "workspace": ws, "reason": "two clips", "operations": [
            {"op": "add_clip", "params": {"source_clip_id": a, "start": 0, "end": 2}},
            {"op": "add_clip", "params": {"source_clip_id": "nope", "start": 0, "end": 1}}]},
        c)["error"]
    assert ok(call("edit_state", {"workspace": ws}, c))["clips"] == []  # all or nothing
    v = ok(call("add_clip", {"workspace": ws, "source_clip_id": a, "start": 0, "end": 2,
                             "reason": "opening"}, c))["version"]
    stale = call("trim_clip", {"workspace": ws, "clip_id": "c001", "end": 1.5,
                               "reason": "tighter", "expected_version": v - 1}, c)
    assert "read it again" in stale["error"]
    assert "no review yet" in call("render", {"workspace": ws, "out": str(tmp_path / "x.mp4"),
                                              "reason": "looks done"}, c)["error"]
    assert ok(call("undo", {"workspace": ws, "to_version": 1, "reason": "back to empty"},
                   c))["summary"] == "restored version 1"
    assert len(tools.describe()) == len(tools.REGISTRY)


def test_first_n_by_recording_time(tmp_path):
    raw = tmp_path / "t"
    for name, when in (("b.mp4", "2026-03-14T10:00:00Z"), ("a.mp4", "2026-03-14T12:00:00Z")):
        tmp = make_clip(tmp_path / f"tmp_{name}", duration=1, size=(320, 240))
        raw.mkdir(exist_ok=True)
        ffmpeg("-i", str(tmp), "-c", "copy", "-metadata", f"creation_time={when}", str(raw / name))
    c = ctx(tmp_path)
    by_name = ok(call("open_project", {"inputs": [str(raw)], "first": 1}, c))["files"]
    by_time = ok(call("open_project", {"inputs": [str(raw)], "first": 1, "order": "time"},
                      c))["files"]
    assert by_name[0]["name"] == "a.mp4" and by_time[0]["name"] == "b.mp4"


def test_unreadable_files_are_skipped_not_fatal(footage, tmp_path):
    (footage / "0_broken.mp4").write_bytes(b"not a video")
    c = ctx(tmp_path)
    opened = ok(call("open_project", {"inputs": [str(footage)], "first": 2}, c))
    assert opened["skipped"] == [{"name": "0_broken.mp4", "reason": opened["skipped"][0]["reason"]}]
    assert opened["clips"] == 1 and ok(call("analyze", {"workspace": opened["workspace"]}, c))
