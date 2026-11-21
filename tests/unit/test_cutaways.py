from llm_video_editor.schemas import EDL, ClipTranscript, EDLClip, Transcript
from llm_video_editor.stages.cutaways import Budget, _Pick, enforce_budget
from tests.fakes import make_segments

SCRIPT = [(t, f"Line number {i} ends here.") for i, t in enumerate(range(0, 200, 10))]


def _world():
    tr = Transcript(clips=[ClipTranscript(clip_id="va", source_file="/a.mp4", duration=200,
                                          segments=make_segments(SCRIPT))])
    edl = EDL(target_platform="youtube", width=1920, height=1080, clips=[
        EDLClip(clip_id="c1", source_clip_id="va", source_file="/a.mp4", start=0, end=100),
        EDLClip(clip_id="c2", source_clip_id="va", source_file="/a.mp4", start=150, end=200)])
    return edl, tr


def P(clip, seg, kind="illustrate", seconds=4.0):
    return _Pick(clip_id=clip, segment_id=seg, kind=kind, query="city at night",
                 seconds=seconds, reason="test")


def test_budget_hook_spacing_humor_and_total():
    edl, tr = _world()  # 150 s cut -> 3 cutaways, 1 humor (every 45 s / 120 s)
    picks = [P("c1", 0), P("c1", 2), P("c1", 3), P("c1", 4, "humor"), P("c1", 6, "humor"),
             P("c2", 16), P("c2", 18)]
    slots, dropped = enforce_budget(picks, edl, tr, Budget())
    assert [(s.anchor_clip_id, round(s.offset)) for s in slots] == [("c1", 20), ("c1", 40),
                                                                    ("c2", 10)]
    assert [s.kind for s in slots] == ["illustrate", "humor", "illustrate"]
    reasons = " | ".join(dropped)
    assert "opening hook" in reasons and "closer than 20s" in reasons
    assert "humor budget used" in reasons and "cutaway budget used" in reasons


def test_humor_length_is_capped_and_unknown_moments_dropped():
    edl, tr = _world()
    slots, dropped = enforce_budget([P("c1", 3, "humor", seconds=9), P("c1", 12), P("c9", 1)],
                                    edl, tr, Budget())
    assert slots[0].duration == 3.0  # humor: 1.5-3 s
    assert len(dropped) == 2 and all("not in the cut" in d for d in dropped)


def test_no_humor_mode_and_empty_answer():
    edl, tr = _world()
    assert enforce_budget([], edl, tr, Budget()) == ([], [])
    slots, _ = enforce_budget([P("c1", 3, "humor")], edl, tr, Budget(humor=False))
    assert slots == []
