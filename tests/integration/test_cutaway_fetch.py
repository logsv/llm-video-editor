"""suggest -> fetch -> cut-audit -> render, with a fake planner and a fake library search."""

import pytest

from llm_video_editor.schemas import EDL, EDLClip, MediaResult, MediaSearch
from llm_video_editor.stages import cutaways
from llm_video_editor.stages.cut_audit import run_cut_audit
from llm_video_editor.stages.render import render
from llm_video_editor.stages.watch import watch
from llm_video_editor.workspace import Workspace
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeProvider, FakeTranscriber
from tests.fixtures.make_fixture import make_clip

pytestmark = [requires_ffmpeg, pytest.mark.integration]
SCRIPT = [(0.2, "Welcome to the show."), (8.0, "Then my cat knocked everything over."),
          (14.0, "That was the end.")]


def test_suggested_cutaway_is_fetched_credited_and_rendered(tmp_path):
    raw = tmp_path / "raw"
    make_clip(raw / "talk.mp4", duration=18, size=(640, 360), fps=30)
    ws = Workspace.open(raw, home=tmp_path / "home")
    info = watch(ws, transcriber=FakeTranscriber(SCRIPT)).manifest.clips[0]
    Workspace.write(ws.edl_raw, EDL(
        target_platform="youtube", width=640, height=360, captions={"enabled": False},
        clips=[EDLClip(clip_id="c1", source_clip_id=info.clip_id, source_file=info.source_file,
                       start=0.0, end=17.0, chapter="Story")]))

    llm = FakeProvider(lambda schema, prompt: schema.model_validate({"cutaways": [
        {"clip_id": "c1", "segment_id": 1, "kind": "humor", "query": "cat knocks cup off table",
         "seconds": 2.5, "reason": "punchline"},
        {"clip_id": "c1", "segment_id": 0, "kind": "illustrate", "query": "studio",
         "seconds": 3, "reason": "intro"}]}))
    plan = cutaways.suggest(ws, llm, budget=cutaways.Budget(every=10, humor_every=10))
    assert [s.kind for s in plan.slots] == ["humor"]  # the intro one covers the hook
    assert "chunk" not in llm.prompts[0][1] and "[1] Then my cat" in llm.prompts[0][1]

    clip = make_clip(tmp_path / "lib_src" / "cat.mp4", duration=5, size=(1280, 720), audio=None)
    hit = MediaResult(result_id="pexels:7", provider="pexels", kind="video", title="Cat",
                      creator="Kim", license="pexels", attribution="Video by Kim from Pexels",
                      download_url="https://cdn/cat.mp4", duration=5, width=1280, height=720)
    searched = []

    def search(query, kind, **kw):
        searched.append((query, kind, kw["providers"]))
        return MediaSearch(query=query, kind=kind, results=[hit])

    def download(ws_, result, library=None, description=None):
        dest = cutaways.lib.media_dir(ws_) / "broll" / "pexels-7-cat.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(clip.read_bytes())
        cutaways.lib._save_credit(ws_, cutaways.lib.credit_of(result, dest))
        return dest

    plan = cutaways.fetch(ws, search=search, download=download)
    assert searched == [("cat knocks cup off table", "video", ("pixabay", "pexels", "wikimedia"))]
    edl = Workspace.read(ws.edl_raw, EDL)
    (ov,) = edl.overlays
    assert ov.overlay_id == plan.slots[0].overlay_id == "x001"
    assert ov.anchor_clip_id == "c1" and ov.duration == 2.5
    assert edl.clips[0].chapter == "Story"  # hand edits in edl_raw survive
    assert cutaways.lib.load_credits(ws).items[0].attribution == "Video by Kim from Pexels"

    again = cutaways.fetch(ws, search=search, download=download)
    assert len(searched) == 1 and again.slots[0].overlay_id == "x001"  # idempotent

    audited, report = run_cut_audit(ws)
    assert report.passed and len(audited.overlays) == 1
    res = render(ws, audited, quality="draft", burn_captions=False)
    assert res.qc.passed, res.qc.problems
