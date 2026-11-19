import pytest

from llm_video_editor.schemas import EDL, EDLClip, GeometryMap, Overlay, Review
from llm_video_editor.stages.geometry import run_geometry
from llm_video_editor.stages.render import render
from llm_video_editor.stages.review import ReviewRequired, build_review, decide
from llm_video_editor.stages.watch import watch
from llm_video_editor.workspace import Workspace
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeDetector, FakeTranscriber

pytestmark = [requires_ffmpeg, pytest.mark.integration]


@pytest.fixture
def cut(media_dir, tmp_path):
    ws = Workspace.open(media_dir, home=tmp_path / "home")
    res = watch(ws, transcriber=FakeTranscriber())
    m = {c.source_file.rsplit("/", 1)[-1]: c for c in res.manifest.clips}
    a, b, c = m["a_talk_1080p30.mp4"], m["b_talk_720p60.mov"], m["c_broll_720p24.mp4"]
    edl = EDL(target_platform="reels", width=1080, height=1920, clips=[
        EDLClip(clip_id="c1", source_clip_id=a.clip_id, source_file=a.source_file, start=0.2,
                end=1.95),
        EDLClip(clip_id="c2", source_clip_id=b.clip_id, source_file=b.source_file, start=2.13,
                end=3.7)],
        overlays=[Overlay(overlay_id="o1", source_clip_id=c.clip_id, source_file=c.source_file,
                          anchor_clip_id="c2", offset=0.2, duration=1.0, zone="pip")])
    Workspace.write(ws.edl_audited, edl)
    return ws, edl


def test_geometry_then_review_gate_then_render(cut):
    ws, edl = cut
    det = FakeDetector(box=(0.05, 0.55, 0.30, 0.95))  # speaker low on the left
    gmap = run_geometry(ws, detector=det)
    assert det.frames == 6  # 3 keyframes x 1 scene x 2 sources, no full decode
    assert Workspace.read(ws.geometry_map, GeometryMap) == gmap
    c1, c2 = gmap.placements
    for p in (c1, c2):
        assert p.crop_center[0] < 0.5  # crop follows the speaker left
        assert p.subject_box_out is not None and p.caption_overlap == 0
    assert c1.caption_position == "top"  # face fills the lower half
    # c2 also carries a top-corner PiP (the face blocks both bottom corners), so its
    # captions move to the centre band, clear of both the face and the inset
    assert c2.pip_corner in ("tl", "tr") and c2.caption_position == "center"

    with pytest.raises(ReviewRequired, match="No review"):
        render(ws, quality="draft", require_review=True)

    review = build_review(ws)
    assert review.status == "pending" and len(review.tiles) == 3  # 2 cuts + 1 overlay
    assert all((ws.root / s).exists() for s in review.sheets)
    assert review.tiles[0].caption.startswith("Hello")
    ass = (ws.review_dir / "captions.ass").read_text()
    assert "\\an8" in ass and "\\an5" in ass
    with pytest.raises(ReviewRequired, match="pending"):
        render(ws, quality="draft", require_review=True)

    decide(ws, approve=True, notes="looks good")
    assert Workspace.read(ws.review_dir / "review.json", Review).status == "approved"
    res = render(ws, quality="draft", require_review=True)
    assert res.qc.passed, res.qc.problems
    assert "\\an8" in (ws.render_dir / "captions.ass").read_text()

    changed = edl.model_copy(update={"clips": edl.clips[:1], "overlays": []})
    Workspace.write(ws.edl_audited, changed)
    with pytest.raises(ReviewRequired, match="changed"):
        render(ws, quality="draft", require_review=True)


def test_mediapipe_detector_smoke():
    pytest.importorskip("mediapipe")
    import numpy as np

    from llm_video_editor.stages.detectors import MediaPipeFaceDetector

    try:
        det = MediaPipeFaceDetector()
    except Exception as e:  # model download blocked or EGL missing
        pytest.skip(f"MediaPipe unavailable here: {e}")
    try:
        assert det.detect(np.zeros((360, 640, 3), np.uint8)) == []
    finally:
        det.close()
