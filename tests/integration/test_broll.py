import json
import os

import pytest

from llm_video_editor.schemas import EDL, AssetPlan, BrollSelection, ClipsManifest
from llm_video_editor.stages.broll import (
    _Choices,
    describe,
    index_library,
    load_index,
    match_broll,
    thumbnail_sheets,
)
from llm_video_editor.stages.cut_audit import run_cut_audit
from llm_video_editor.stages.plan import ReduceResult, plan
from llm_video_editor.stages.render import render
from llm_video_editor.stages.watch import watch
from llm_video_editor.workspace import Workspace
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeProvider, FakeTranscriber
from tests.fixtures.make_fixture import make_clip

pytestmark = [requires_ffmpeg, pytest.mark.integration]

SCRIPT = [(0.1, "Fresh coffee every morning."),  # 0.10-1.25 s
          (1.6, "Then we write code.")]  # 1.60-2.75 s


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    lib = tmp_path_factory.mktemp("broll_lib")
    make_clip(lib / "aerials" / "DJI_0042_city_skyline.mp4", duration=3, size=(1280, 720),
              audio=None)
    make_clip(lib / "food" / "pouring_coffee_closeup.mp4", duration=3, size=(720, 1280),
              audio=None)
    make_clip(lib / "C0042.MP4", duration=3, size=(1280, 720), audio=None)
    (lib / "C0042.json").write_text(json.dumps({"description": "hands typing code on a laptop",
                                                "tags": ["laptop", "keyboard", "code"]}))
    return lib


def test_index_is_incremental_and_keeps_manual_descriptions(library, tmp_path):
    index, scanned = index_library(library, rescan=True)
    assert scanned == 3 and len(index.items) == 3
    by_file = {i.rel_path: i for i in index.items}
    assert by_file["C0042.MP4"].described_by == "sidecar"
    assert by_file["C0042.MP4"].tags == ["laptop", "keyboard", "code"]
    assert by_file["aerials/DJI_0042_city_skyline.mp4"].tags == ["aerials", "city", "skyline"]
    assert by_file["food/pouring_coffee_closeup.mp4"].orientation == "portrait"
    assert all((library / i.thumbnail).exists() for i in index.items)
    assert (library / "broll_index.md").read_text().count("| v") == 3

    assert index_library(library)[1] == 0  # nothing changed
    item = by_file["aerials/DJI_0042_city_skyline.mp4"]
    describe(library, item.item_id, "aerial city skyline at dusk", ["city", "aerial", "dusk"])
    f = library / item.rel_path
    os.utime(f, ns=(f.stat().st_mtime_ns + 10**9,) * 2)
    index, scanned = index_library(library)
    assert scanned == 1
    again = next(i for i in index.items if i.rel_path == item.rel_path)
    assert again.described_by == "manual" and again.description == "aerial city skyline at dusk"
    sheets = thumbnail_sheets(library)
    assert len(sheets) == 1 and sheets[0].exists()


@pytest.fixture
def planned(media_dir, tmp_path):
    ws = Workspace.open(media_dir, home=tmp_path / "home")
    watch(ws, transcriber=FakeTranscriber(script=SCRIPT))
    m = Workspace.read(ws.clips_manifest, ClipsManifest)
    talk = [c.clip_id for c in m.clips if c.kind == "talking_head"]

    def responder(schema, prompt):
        return ReduceResult.model_validate({
            "selections": [{"clip_id": talk[0], "first_segment": 0, "last_segment": 0,
                            "purpose": "coffee", "priority": 1},
                           {"clip_id": talk[1], "first_segment": 1, "last_segment": 1,
                            "purpose": "code", "priority": 1}],
            "broll": [{"over_selection": 0, "clip_id": None, "query": "pouring fresh coffee",
                       "seconds": 1.0},
                      {"over_selection": 1, "clip_id": None, "query": "typing code laptop",
                       "seconds": 1.0},
                      {"over_selection": 1, "clip_id": None, "query": "underwater shark",
                       "seconds": 1.0}]})

    plan(ws, "5s reel", FakeProvider(responder), target_platform="reels")
    return ws


def test_match_fills_slots_from_library_and_renders(planned, library):
    index_library(library)
    lib = load_index(library)
    ids = {i.rel_path: i.item_id for i in lib.items}
    sel = match_broll(planned, library)
    chosen = [s.chosen for s in sel.slots]
    assert chosen[0] == ids["food/pouring_coffee_closeup.mp4"]
    assert chosen[1] == ids["C0042.MP4"]
    assert chosen[2] is None  # nothing about sharks: left empty, not forced
    assert Workspace.read(planned.broll_select, BrollSelection) == sel

    edl = Workspace.read(planned.edl_raw, EDL)
    assert len(edl.overlays) == 2
    assert edl.overlays[0].source_file.endswith("pouring_coffee_closeup.mp4")
    manifest = Workspace.read(planned.clips_manifest, ClipsManifest)
    assert manifest.get(chosen[0]).kind == "broll"  # registered for cut-audit/render

    audited, report = run_cut_audit(planned)
    assert report.passed and len(audited.clips) == 2 and len(audited.overlays) == 2
    res = render(planned, quality="draft", burn_captions=False)
    assert res.qc.passed, res.qc.problems


def test_llm_rerank_and_generated_fallback(planned, library, tmp_path):
    index_library(library)
    prompts = []

    def responder(schema, prompt):
        assert schema is _Choices
        prompts.append(prompt)
        return _Choices.model_validate({"choices": [
            {"slot": 0, "item_id": None, "reason": "none fit"},
            {"slot": 1, "item_id": "vnot-a-candidate", "reason": "bogus"}]})

    made = []

    def generator(query, seconds, orientation):
        path = tmp_path / f"gen_{len(made)}.mp4"
        make_clip(path, duration=2, size=(720, 1280), audio=None)
        made.append(query)
        return path

    sel = match_broll(planned, library, provider=FakeProvider(responder), generator=generator)
    assert ".mp4" not in prompts[0] and ".MP4" not in prompts[0]  # zero-upload boundary
    s0, s1, s2 = sel.slots
    assert s0.generated and s0.reason == "generated"  # LLM said none -> generated
    assert s1.reason == "bm25" and not s1.generated  # bogus id ignored, BM25 pick kept
    assert s2.generated
    assert made == ["pouring fresh coffee", "underwater shark"]
    plan_ = Workspace.read(planned.asset_plan, AssetPlan)
    assert all(b.clip_id for b in plan_.broll)
