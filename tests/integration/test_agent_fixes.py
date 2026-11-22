"""Fixes from the first real agent run (Bhopal -> Khajuraho, 262 clips): media results
from earlier searches, batch params in the tools' shape, background jobs for long
tools, the `python -m` launcher, renders kept out of footage folders, and relative
workspace paths."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from llm_video_editor.agent import jobs, setup
from llm_video_editor.agent.tools import Context, call
from llm_video_editor.media import library
from llm_video_editor.schemas import MediaResult
from llm_video_editor.workspace import Workspace, check_output_path, resolve_inputs
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeTranscriber
from tests.fixtures.make_fixture import make_clip

SCRIPT = [(0.3, "So today I ride from Bhopal."), (2.3, "The road is long and empty.")]


def ok(res):
    assert "error" not in res, res
    return res


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "trip"
    make_clip(raw / "a.mp4", duration=4, size=(640, 360), fps=30)
    c = Context(home=tmp_path / "home",
                transcriber_factory=lambda model: FakeTranscriber(SCRIPT))
    ws = ok(call("open_project", {"inputs": [str(raw)]}, c))["workspace"]
    return raw, ws, c


def _hit(i: int, kind="music") -> MediaResult:
    return MediaResult(result_id=f"openverse:t{i}", provider="openverse", kind=kind,
                       title=f"Track {i}", download_url=f"https://cdn.example/t{i}.mp3",
                       license="CC BY 4.0", attribution=f"Track {i} by X")


def test_media_results_from_any_earlier_search(monkeypatch, tmp_path):
    ws = Workspace(tmp_path / "ws", [])
    batches = iter([[_hit(1), _hit(2)], [_hit(3)]])
    monkeypatch.setitem(library.PROVIDERS, "openverse", lambda *a, **k: next(batches))
    library.search("ambient india", "music", providers=("openverse",), ws=ws)
    library.search("river", "music", providers=("openverse",), ws=ws)
    assert library.find_result(ws, "openverse:t1").title == "Track 1"  # first search
    assert library.find_result(ws, "openverse:t3").title == "Track 3"
    with pytest.raises(KeyError, match="not returned by any media_search"):
        library.find_result(ws, "openverse:nope")


@requires_ffmpeg
def test_batch_accepts_the_standalone_tools_shape(project):
    raw, ws, c = project
    a = ok(call("analyze", {"workspace": ws}, c))["clips"][0]["clip_id"]
    ok(call("new_edit", {"workspace": ws, "target_platform": "youtube", "reason": "start"}, c))
    ok(call("batch", {"workspace": ws, "reason": "cut with a lower third", "operations": [
        {"op": "add_clip", "params": {"source_clip_id": a, "start": 0.2, "end": 3.5}},
        # nested `fields`, exactly as add_item / set_pacing take them standalone
        {"op": "add_item", "params": {"kind": "graphics", "fields": {
            "template": "lower-third", "variables": {"title": "Bhopal"},
            "anchor_clip_id": "c001", "offset": 0.5, "duration": 2.0}}},
        {"op": "set_pacing", "params": {"fields": {"tighten": True},
                                        "reason": "per-op reasons are ignored"}}]}, c))
    st = ok(call("edit_state", {"workspace": ws}, c))
    assert st["graphics"][0]["anchor_clip_id"] == "c001"
    clash = call("batch", {"workspace": ws, "reason": "both shapes", "operations": [
        {"op": "set_pacing", "params": {"tighten": False, "fields": {"tighten": True}}}]}, c)
    assert "both inside and outside `fields`" in clash["error"]


@requires_ffmpeg
def test_long_tools_run_as_jobs(project):
    raw, ws, c = project
    gate = threading.Event()

    def slow(model):  # analysis that runs until the test lets it finish
        gate.wait(30)
        return FakeTranscriber(SCRIPT)

    c.jobs, c.job_wait, c.transcriber_factory = "thread", 0.0, slow
    started = ok(call("analyze", {"workspace": ws}, c))
    assert started["status"] in ("queued", "running") and "job_status" in started["next"]
    busy = call("review", {"workspace": ws}, c)  # one long job per workspace
    assert "still running" in busy["error"]
    gate.set()
    done = ok(call("job_status", {"workspace": ws, "job_id": started["job_id"],
                                  "wait_seconds": 55}, c))
    assert done["clips"][0]["kind"] == "talking_head"  # the tool's own result

    # a job whose process died is reported failed, with the log tail
    path = jobs._path(Path(ws), "render-dead")
    path.write_text(json.dumps({"job_id": "render-dead", "tool": "render", "args": {},
                                "status": "running", "created_at": time.time()}))
    path.with_suffix(".pid").write_text("99999999")  # above any pid_max: never alive
    path.with_suffix(".log").write_text("Traceback ...\nSegmentation fault\n")
    res = call("job_status", {"workspace": ws, "job_id": "render-dead", "wait_seconds": 0}, c)
    assert "exited without a result" in res["error"] and "Segmentation" in res["log_tail"]


@requires_ffmpeg
def test_job_process_runs_outside_the_server(project):
    raw, ws, c = project
    c.jobs = "process"
    job = jobs.start("edit_state", {"workspace": ws}, c, Path(ws), wait=55)
    assert job["status"] == "done", job
    assert job["result"]["version"] == 0 and job["pid"] != os.getpid()


@requires_ffmpeg
def test_renders_stay_out_of_footage_folders(project, tmp_path):
    raw, ws, c = project
    sources = Workspace.load(Path(ws)).sources
    with pytest.raises(ValueError, match="inside the footage folder"):
        check_output_path(raw / "edit" / "trip.mp4", sources)
    check_output_path(tmp_path / "renders" / "trip.mp4", sources)  # beside it is fine
    res = call("render", {"workspace": ws, "out": str(raw / "edit" / "x.mp4"),
                          "reason": "ready"}, c)
    assert "inside the footage folder" in res["error"]
    assert not (raw / "edit").exists()  # refused before anything was created

    # earlier renders and workspaces found in a footage folder are never ingested
    make_clip(raw / "edit" / "old_render.mp4", duration=1, size=(320, 240))
    (raw / "edit" / "old_render.llmve.json").write_text("{}")
    make_clip(raw / ".llmve" / "abc" / "render" / "youtube.mp4", duration=1, size=(320, 240))
    assert [p.name for p in resolve_inputs(raw)] == ["a.mp4"]


def test_relative_workspace_paths_become_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws = Workspace(Path(".llmve") / "abc", [])
    assert ws.root.is_absolute() and ws.root == tmp_path / ".llmve" / "abc"
    assert Workspace.load(Path(".llmve/abc")).root.is_absolute()


def test_python_m_launcher_and_mcp_config(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, "-m", "llm_video_editor", "--help"], cwd=repo,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "mcp" in out.stdout
    args = setup.mcp_config(repo)["mcpServers"]["llmve"]["args"]
    assert args[:2] == ["--directory", str(repo)]
    assert args[-4:] == ["python", "-m", "llm_video_editor", "mcp"]
    assert json.loads((repo / ".mcp.json").read_text())["mcpServers"]["llmve"]["args"][-3:] \
        == ["-m", "llm_video_editor", "mcp"]


@requires_ffmpeg
def test_scanning_a_folder_inside_a_workspace_still_works(tmp_path):
    lib = tmp_path / ".llmve" / "abc" / "media" / "broll"  # the B-roll library
    make_clip(lib / "goat.mp4", duration=1, size=(320, 240))
    assert [p.name for p in resolve_inputs(lib)] == ["goat.mp4"]
    assert [p.name for p in resolve_inputs(str(lib / "*.mp4"))] == ["goat.mp4"]
    with pytest.raises(FileNotFoundError):  # a glob from above skips the workspace
        resolve_inputs(str(tmp_path / "**" / "*.mp4"))
