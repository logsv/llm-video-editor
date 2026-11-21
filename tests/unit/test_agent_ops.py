import pytest

from llm_video_editor.agent import ops
from llm_video_editor.agent.edits import EditError
from llm_video_editor.schemas import EDL, ClipInfo, ClipsManifest

M = ClipsManifest(batch_hash="x", clips=[
    ClipInfo(clip_id="va", source_file="/a.mp4", duration=60, width=1920, height=1080, fps=30,
             has_audio=True),
    ClipInfo(clip_id="vb", source_file="/b.mp4", duration=8, width=1920, height=1080, fps=30,
             has_audio=False, kind="broll")])


def edl():
    e = ops.new_edit(None, M, target_platform="youtube", target_duration=600)
    e = ops.add_clip(e, M, source_clip_id="va", start=0, end=10, chapter="Start")
    return ops.add_clip(e, M, source_clip_id="va", start=20, end=30)


def test_new_edit_and_add_clips():
    e = edl()
    assert (e.width, e.height, e.target_duration) == (1920, 1080, 600)
    assert [(c.clip_id, c.start, c.end, c.chapter) for c in e.clips] == [
        ("c001", 0, 10, "Start"), ("c002", 20, 30, None)]
    e = ops.add_clip(e, M, source_clip_id="vb", start=1, end=4, after="c001", role="broll")
    assert [c.clip_id for c in e.clips] == ["c001", "c003", "c002"]
    with pytest.raises(EditError, match="not inside source va"):
        ops.add_clip(e, M, source_clip_id="va", start=50, end=70)
    with pytest.raises(EditError, match="no source clip 'vz'"):
        ops.add_clip(e, M, source_clip_id="vz", start=0, end=1)
    with pytest.raises(EditError, match="unknown platform|Unknown platform"):
        ops.new_edit(None, M, target_platform="myspace")


def test_trim_split_move_set_remove():
    e = ops.trim_clip(edl(), M, clip_id="c001", start=1.5)
    assert e.clips[0].start == 1.5 and e.clips[0].end == 10
    e = ops.split_clip(e, M, clip_id="c001", at=5)
    assert [(c.clip_id, c.start, c.end, c.chapter) for c in e.clips[:2]] == [
        ("c001", 1.5, 5, "Start"), ("c003", 5, 10, None)]
    e = ops.move_clip(e, M, clip_id="c002", position=0)
    assert [c.clip_id for c in e.clips] == ["c002", "c001", "c003"]
    e = ops.move_clip(e, M, clip_id="c002", after="c003")
    assert [c.clip_id for c in e.clips] == ["c001", "c003", "c002"]
    e = ops.set_clip(e, M, clip_id="c001", zoom=1.2, chapter="")
    assert e.clips[0].zoom == 1.2 and e.clips[0].chapter is None
    with pytest.raises(EditError, match="invalid value"):
        ops.set_clip(e, M, clip_id="c001", zoom=9)
    with pytest.raises(EditError, match="use trim_clip"):
        ops.set_clip(e, M, clip_id="c001", start=0)
    with pytest.raises(EditError, match="unknown field"):
        ops.set_clip(e, M, clip_id="c001", sparkle=True)
    with pytest.raises(EditError, match="must be inside c001"):
        ops.split_clip(e, M, clip_id="c001", at=30)
    e = ops.remove_clip(e, M, clip_id="c003")
    assert [c.clip_id for c in e.clips] == ["c001", "c002"]
    with pytest.raises(EditError, match="no clip 'c009'; clips are c001, c002"):
        ops.remove_clip(e, M, clip_id="c009")


def test_settings_and_anchored_items():
    e = ops.set_pacing(edl(), M, tighten=True, max_gap=0.5)
    e = ops.set_captions(e, M, enabled=False)
    e = ops.set_mix(e, M, duck_depth_db=15)
    e = ops.set_output(e, M, clear_target=True, fit_mode="contain")
    assert e.pacing.max_gap == 0.5 and not e.captions.enabled and e.mix.duck_depth_db == 15
    assert e.target_duration is None and e.fit_mode == "contain"
    e = ops.add_item(e, M, kind="overlays", source_clip_id="vb", anchor_clip_id="c002",
                     offset=1, duration=2)
    e = ops.add_item(e, M, kind="music", file="/bed.mp3", anchor_clip_id="c001")
    e = ops.add_item(e, M, kind="graphics", template="lower-third", anchor_clip_id="c001",
                     duration=3, variables={"title": "Vikas"})
    assert [o.overlay_id for o in e.overlays] == ["x001"] and e.overlays[0].source_file == "/b.mp4"
    assert e.music[0].track_id == "a001" and e.graphics[0].graphic_id == "g001"
    with pytest.raises(EditError, match="c002 anchors x001"):
        ops.remove_clip(e, M, clip_id="c002")
    e = ops.update_item(e, M, kind="music", item_id="a001", volume_db=-24)
    assert e.music[0].volume_db == -24
    with pytest.raises(EditError, match="no music track 'a009'"):
        ops.update_item(e, M, kind="music", item_id="a009", volume_db=0)
    with pytest.raises(EditError, match="invalid graphic"):
        ops.add_item(e, M, kind="graphics", anchor_clip_id="c001", duration=2)  # no template
    e = ops.remove_item(e, M, kind="overlays", item_id="x001")
    assert ops.remove_clip(e, M, clip_id="c002").clips[-1].clip_id == "c001"
    assert isinstance(e, EDL)
