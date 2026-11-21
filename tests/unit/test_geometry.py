import pytest

from llm_video_editor.schemas import (
    EDL,
    ClipGeometry,
    ClipInfo,
    ClipsManifest,
    Detection,
    EDLClip,
    GeometryMap,
    Overlay,
    SceneGeometry,
)
from llm_video_editor.stages.geometry import (
    apply_geometry,
    box_to_output,
    caption_bands,
    choose_caption,
    choose_pip,
    edl_hash,
    keep_box_inside,
    place_clips,
    summarise_scene,
)


def test_summarise_prefers_faces_and_unions_samples():
    dets = [Detection(t=1, label="face", bbox=(0.4, 0.2, 0.5, 0.4), score=0.9),
            Detection(t=1, label="face", bbox=(0.0, 0.0, 0.02, 0.02), score=0.6),
            Detection(t=2, label="face", bbox=(0.45, 0.25, 0.55, 0.45), score=0.9),
            Detection(t=2, label="person", bbox=(0.1, 0.1, 0.9, 0.9), score=0.99)]
    sc = summarise_scene(0, 5, dets)
    assert sc.subject_box == (0.4, 0.2, 0.55, 0.45)
    assert sc.focus == pytest.approx((0.475, 0.325))
    person_only = summarise_scene(0, 5, [Detection(t=1, label="person",
                                                   bbox=(0.2, 0.0, 0.6, 1.0), score=0.8)])
    assert person_only.focus == pytest.approx((0.4, 0.2))  # aims at the head
    assert summarise_scene(0, 5, []).focus is None


def test_crop_keeps_subject_inside_window():
    # 16:9 source to 9:16 output: crop window is 0.316 of the source width
    box = (0.05, 0.3, 0.25, 0.7)  # speaker on the far left
    c = keep_box_inside((0.5, 0.5), box, 1920, 1080, 1080, 1920)
    out = box_to_output(box, 1920, 1080, 1080, 1920, c)
    assert out is not None and out[0] >= 0 and out[2] <= 1
    assert out[2] - out[0] == pytest.approx((0.25 - 0.05) / 0.3164, rel=0.01)  # not clipped
    # a subject wider than the window cannot fit; centre stays on its middle
    wide = keep_box_inside((0.5, 0.5), (0.1, 0.2, 0.9, 0.8), 1920, 1080, 1080, 1920)
    assert 0 <= wide[0] <= 1


def test_caption_moves_when_face_is_low():
    bands = caption_bands(1080, 1920)
    assert bands["bottom"][1] == pytest.approx(0.86)
    assert choose_caption((0.3, 0.2, 0.7, 0.5), 1080, 1920) == ("bottom", 0.0)
    pos, overlap = choose_caption((0.3, 0.6, 0.7, 0.9), 1080, 1920)
    assert pos == "top" and overlap == 0
    assert choose_caption(None, 1920, 1080) == ("bottom", 0.0)


def test_pip_corner_avoids_subject():
    assert choose_pip(None) == "br"
    assert choose_pip((0.6, 0.6, 0.95, 0.95)) == "bl"
    assert choose_pip((0.0, 0.5, 1.0, 1.0)) == "tr"
    # subject bottom-right and captions moved to the top band: both top corners collide
    # with captions, bottom-left is clear
    assert choose_pip((0.5, 0.55, 0.9, 0.95), avoid=(0.08, 0.14, 0.92, 0.22)) == "bl"


def _edl():
    return EDL(target_platform="reels", width=1080, height=1920, clips=[
        EDLClip(clip_id="c1", source_clip_id="va", source_file="/a", start=0, end=4)],
        overlays=[Overlay(overlay_id="o1", source_clip_id="vb", source_file="/b",
                          anchor_clip_id="c1", duration=1, zone="pip")])


def test_place_and_apply_geometry():
    edl = _edl()
    manifest = ClipsManifest(batch_hash="x", clips=[
        ClipInfo(clip_id="va", source_file="/a", duration=10, width=1920, height=1080, fps=30,
                 has_audio=True)])
    geo = ClipGeometry(clip_id="va", source_file="/a", width=1920, height=1080, scenes=[
        SceneGeometry(start=0, end=10, focus=(0.75, 0.6), subject_box=(0.68, 0.55, 0.82, 0.95))])
    placements = place_clips(edl, manifest, {"va": geo})
    p = placements[0]
    assert p.crop_center[0] > 0.6
    assert p.caption_position == "center"  # face low + top-corner PiP -> centre band
    gmap = GeometryMap(detector="t", edl_hash=edl_hash(edl), placements=placements)
    applied = apply_geometry(edl, gmap)
    assert applied.clips[0].crop_center == p.crop_center
    assert applied.clips[0].caption_position == "center"
    assert applied.overlays[0].corner == p.pip_corner
    # geometry fields do not change the cut's fingerprint ...
    assert edl_hash(applied) == edl_hash(edl)
    # ... but a different cut ignores a stale map
    moved = edl.model_copy(update={"clips": [edl.clips[0].model_copy(update={"end": 3})]})
    assert apply_geometry(moved, gmap) == moved


def test_agent_chosen_crop_and_caption_are_kept():
    from llm_video_editor.schemas import EDL, ClipPlacement, EDLClip, GeometryMap
    from llm_video_editor.stages.geometry import apply_geometry, edl_hash

    edl = EDL(target_platform="reels", width=1080, height=1920, clips=[
        EDLClip(clip_id="c1", source_clip_id="v", source_file="/v.mp4", start=0, end=2,
                crop_center=(0.2, 0.5)),
        EDLClip(clip_id="c2", source_clip_id="v", source_file="/v.mp4", start=2, end=4)])
    place = [ClipPlacement(clip_id=c, crop_center=(0.7, 0.4), caption_position="top")
             for c in ("c1", "c2")]
    gmap = GeometryMap(edl_hash=edl_hash(edl), detector="test", placements=place)
    out = apply_geometry(edl, gmap)
    assert out.clips[0].crop_center == (0.2, 0.5) and out.clips[1].crop_center == (0.7, 0.4)
    assert out.clips[0].caption_position == "top"
