import pytest

from llm_video_editor.schemas import EDL, AssetPlan, ClipsManifest
from llm_video_editor.stages import plan as plan_mod
from llm_video_editor.stages.plan import MapResult, ReduceResult, chunk_transcript, plan
from llm_video_editor.stages.watch import watch
from llm_video_editor.workspace import Workspace
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeProvider, FakeTranscriber


@pytest.fixture
def watched(media_dir, tmp_path):
    ws = Workspace.open(media_dir, home=tmp_path)
    watch(ws, transcriber=FakeTranscriber())
    return ws


def _clip_ids(ws):
    m = Workspace.read(ws.clips_manifest, ClipsManifest)
    talk = [c.clip_id for c in m.clips if c.kind == "talking_head"]
    broll = [c.clip_id for c in m.clips if c.kind == "broll"]
    return talk, broll


@requires_ffmpeg
def test_plan_resolves_segment_ids_to_times(watched):
    talk, broll = _clip_ids(watched)

    def responder(schema, prompt):
        assert schema is ReduceResult
        return ReduceResult.model_validate({
            "selections": [
                {"clip_id": talk[1], "first_segment": 1, "last_segment": 1, "purpose": "hook",
                 "priority": 1},
                {"clip_id": talk[0], "first_segment": 0, "last_segment": 1, "purpose": "body",
                 "priority": 2},
                {"clip_id": "vbogus", "first_segment": 0, "last_segment": 0, "purpose": "x",
                 "priority": 3},
            ],
            "broll": [{"over_selection": 1, "clip_id": broll[0], "query": "city", "seconds": 2}],
        })

    provider = FakeProvider(responder)
    asset_plan, edl = plan(watched, "5s reel: best bits", provider, target_platform="reels")
    assert asset_plan.target_duration == 5
    assert [s.clip_id for s in asset_plan.segments] == [talk[1], talk[0]]
    s0 = asset_plan.segments[0]
    assert (s0.start, s0.end) == (2.2, 3.65)  # "Today we cover three ideas." from the fake ASR
    assert "vbogus" in asset_plan.notes
    assert (edl.width, edl.height) == (1080, 1920)
    assert edl.clips[0].source_file.endswith("b_talk_720p60.mov")
    assert len(edl.overlays) == 1 and edl.overlays[0].anchor_clip_id == "c002"
    Workspace.read(watched.asset_plan, AssetPlan)
    Workspace.read(watched.edl_raw, EDL)
    # Only clip ids and text crossed the boundary; never file names.
    _, prompt, _ = provider.prompts[0]
    assert ".mp4" not in prompt and ".mov" not in prompt and "a_talk" not in prompt
    assert talk[0] in prompt and broll[0] in prompt


@requires_ffmpeg
def test_long_transcripts_use_map_then_reduce(watched, monkeypatch):
    talk, _ = _clip_ids(watched)
    monkeypatch.setattr(plan_mod, "DIRECT_LIMIT_CHARS", 10)

    def responder(schema, prompt):
        if schema is MapResult:
            ids = [ln.split()[2] for ln in prompt.splitlines() if ln.startswith("## chunk")]
            return MapResult.model_validate({"chunks": [
                {"chunk_id": cid, "score": 7, "summary": "intro", "best_segments": [1]}
                for cid in ids]})
        assert "score 7/10" in prompt and "[0]" not in prompt  # only best segments forwarded
        return ReduceResult.model_validate({"selections": [
            {"clip_id": talk[0], "first_segment": 1, "last_segment": 1, "purpose": "p",
             "priority": 1}]})

    provider = FakeProvider(responder)
    asset_plan, _ = plan(watched, "short cut", provider, target_platform="youtube",
                         target_duration=5)
    assert [p[2] for p in provider.prompts].count(MapResult) == 1
    assert asset_plan.segments[0].first_segment_id == 1


@requires_ffmpeg
@requires_ffmpeg
def test_target_is_capped_by_available_speech(watched):
    talk, _ = _clip_ids(watched)
    provider = FakeProvider(lambda s, p: ReduceResult.model_validate({"selections": [
        {"clip_id": talk[0], "first_segment": 0, "last_segment": 1, "purpose": "p",
         "priority": 1}]}))
    asset_plan, _ = plan(watched, "10m edit", provider, target_platform="youtube")
    assert asset_plan.target_duration == pytest.approx(6.4)  # 2 clips x (1.75s + 1.45s) of speech


def test_all_broll_folder_builds_montage(media_dir, tmp_path):
    ws = Workspace.open(media_dir, home=tmp_path)
    watch(ws, transcriber=FakeTranscriber(silent=True))
    provider = FakeProvider(lambda s, p: pytest.fail("no LLM needed for a montage"))
    asset_plan, edl = plan(ws, "8s montage", provider, target_platform="reels")
    assert asset_plan.target_duration == 8
    assert sum(s.duration for s in asset_plan.segments) == pytest.approx(8, abs=0.01)


def test_chunking_respects_window():
    from llm_video_editor.schemas import ClipTranscript
    from tests.fakes import make_segments

    ct = ClipTranscript(clip_id="v1", source_file="/x", duration=200, segments=make_segments(
        [(t, "Some words here.") for t in range(0, 180, 20)]))
    chunks = chunk_transcript(ct, seconds=60)
    assert len(chunks) == 3
    assert all(c.end - c.start <= 60 for c in chunks)
