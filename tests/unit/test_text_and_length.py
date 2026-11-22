"""Readable, safe, shot-matched on-screen text; and long YouTube cuts (2-50 min) from
mostly silent travel footage."""

from __future__ import annotations

import numpy as np
import pytest

from llm_video_editor.schemas import EDL, AssetPlan, ClipInfo, ClipsManifest, PlannedSegment
from llm_video_editor.schemas.manifest import SceneCut
from llm_video_editor.stages import text_safety as ts
from llm_video_editor.stages.plan import FILL_PRIORITY, build_edl, choose_target, fill_with_footage

H, W = 270, 480


def _edl(width=1920, height=1080, captions=True) -> EDL:
    return EDL(target_platform="youtube", width=width, height=height,
               captions={"enabled": captions})


def _bg(rgb) -> np.ndarray:
    return np.full((H, W, 3), rgb, np.uint8)


def _text(box, rgb=(255, 255, 255), panel=None) -> np.ndarray:
    """White 'text' strokes inside ``box`` (x1, y1, x2, y2 in 0-1), optionally on a card."""
    g = np.zeros((H, W, 4), np.uint8)
    x1, y1, x2, y2 = (round(box[0] * W), round(box[1] * H), round(box[2] * W), round(box[3] * H))
    if panel is not None:
        g[y1:y2, x1:x2] = (*panel, 230)
    for x in range(x1 + 2, x2 - 2, 6):  # thin vertical strokes like letters
        g[y1 + 3:y2 - 3, x:x + 2] = (*rgb, 255)
    return g


def test_white_text_on_bright_sky_is_flagged_and_dark_shot_passes():
    sky = ts.measure(_edl(), _bg((235, 240, 250)), _text((0.06, 0.07, 0.4, 0.17)),
                     graphic_id="g1", t_out=3.0)
    assert sky.backdrop == "video" and sky.contrast < 3 and "low contrast" in sky.issues[0]
    assert sky.suggested["text"] == "#111111"
    night = ts.measure(_edl(), _bg((20, 25, 30)), _text((0.06, 0.07, 0.4, 0.17)),
                       graphic_id="g1", t_out=3.0)
    assert night.contrast > 10 and night.issues == [] and night.suggested["text"] == "#FFFFFF"


def test_a_card_carries_its_own_contrast():
    chk = ts.measure(_edl(), _bg((235, 240, 250)),
                     _text((0.06, 0.6, 0.45, 0.72), panel=(12, 14, 20)), graphic_id="g2",
                     t_out=1.0)
    assert chk.backdrop == "panel" and chk.contrast > 10 and chk.issues == []


def test_busy_background_safe_area_faces_and_captions():
    rng = np.random.default_rng(1)
    foliage = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    busy = ts.measure(_edl(), foliage, _text((0.06, 0.07, 0.4, 0.17)), graphic_id="g", t_out=0)
    assert any("busy background" in i for i in busy.issues)

    edge = ts.measure(_edl(), _bg((20, 20, 20)), _text((0.0, 0.0, 0.3, 0.08)),
                      graphic_id="g", t_out=0)
    assert any("outside the safe area (left, top)" in i for i in edge.issues)
    face = ts.measure(_edl(), _bg((20, 20, 20)), _text((0.4, 0.3, 0.6, 0.45)),
                      graphic_id="g", t_out=0, subject_box=(0.38, 0.2, 0.62, 0.6))
    assert any("subject/face" in i for i in face.issues)
    caps = ts.measure(_edl(), _bg((20, 20, 20)), _text((0.3, 0.8, 0.7, 0.9)),
                      graphic_id="g", t_out=0, caption_position="bottom")
    assert any("caption band" in i for i in caps.issues)
    no_caps = ts.measure(_edl(captions=False), _bg((20, 20, 20)), _text((0.3, 0.8, 0.7, 0.9)),
                         graphic_id="g", t_out=0, caption_position="bottom")
    assert not any("caption band" in i for i in no_caps.issues)
    # 9:16: the platform UI takes the right edge and the bottom fifth
    tall = ts.measure(_edl(1080, 1920), _bg((20, 20, 20)), _text((0.3, 0.75, 0.9, 0.85)),
                      graphic_id="g", t_out=0)
    assert any("outside the safe area (right, bottom)" in i for i in tall.issues)


def test_palette_and_colours_follow_the_shot():
    sky = _bg((90, 150, 230))  # blue sky
    s = ts.suggest(sky)
    r, g, b = (int(s["contrast"][i:i + 2], 16) for i in (1, 3, 5))
    assert r > b  # the complement of blue is warm
    r, g, b = (int(s["match"][i:i + 2], 16) for i in (1, 3, 5))
    assert b > r  # the match stays blue
    assert ts.palette(sky)[0] == pytest.approx((90, 150, 230))
    assert ts.auto_accent(sky).startswith("#")


def _clip(cid, dur, kind, scenes=()):
    return ClipInfo(clip_id=cid, source_file=f"/v/{cid}.mp4", duration=dur, width=1920,
                    height=1080, fps=30, has_audio=True, kind=kind,
                    scenes=[SceneCut(start=a, end=b) for a, b in scenes])


def test_silent_footage_fills_a_long_youtube_cut_in_recording_order():
    m = ClipsManifest(batch_hash="x", clips=[
        _clip("ride1", 300, "broll", [(0, 120), (120, 300)]), _clip("talk1", 60, "talking_head"),
        _clip("ride2", 600, "broll"), _clip("talk2", 40, "talking_head"),
        _clip("ride3", 900, "broll")])
    speech = [PlannedSegment(clip_id="talk1", source_file="/v/talk1.mp4", start=1, end=50),
              PlannedSegment(clip_id="talk2", source_file="/v/talk2.mp4", start=0, end=30)]
    plan = AssetPlan(prompt="p", target_platform="youtube", target_duration=1500,
                     segments=speech)
    # speech alone (~90 s) could never make 25 minutes; the rides can
    assert choose_target("25 minute trip video", "youtube", None, 79, footage=1798.5) == 1500
    assert choose_target("50 minute trip video", "youtube", None, 79, footage=1798.5) == \
        pytest.approx(79 * 1.15 + 1798.5)
    out = fill_with_footage(plan, m, 1500)
    order = [s.clip_id for s in out.segments]
    assert order.index("talk1") > order.index("ride1")  # filmed before talk1
    assert order.index("ride2") > order.index("talk1") and order.index("ride3") > order.index("talk2")
    total = sum(s.duration for s in out.segments)
    assert 1450 <= total <= 1510
    fills = [s for s in out.segments if s.purpose == "footage"]
    assert all(s.priority == FILL_PRIORITY for s in fills)
    assert all(0.25 <= s.start and s.end <= 900 - 0.25 for s in fills)
    edl = build_edl(out, m)
    assert {c.role for c in edl.clips if c.notes == "footage"} == {"broll"}
    assert "filled" in out.notes

    # 2-minute cut: little or no fill
    short = fill_with_footage(plan.model_copy(update={"target_duration": 120}), m, 120)
    assert sum(s.duration for s in short.segments) <= 125
    # enough speech already: unchanged
    assert fill_with_footage(plan, m, 70).segments == plan.segments


def test_hook_stays_first_when_the_planner_opens_with_a_later_moment():
    m = ClipsManifest(batch_hash="x", clips=[
        _clip("ride1", 100, "broll"), _clip("talk1", 60, "talking_head"),
        _clip("talk2", 40, "talking_head")])
    hook = PlannedSegment(clip_id="talk2", source_file="/v/talk2.mp4", start=0, end=5,
                          purpose="hook")
    rest = PlannedSegment(clip_id="talk1", source_file="/v/talk1.mp4", start=0, end=50)
    plan = AssetPlan(prompt="p", target_platform="youtube", target_duration=120,
                     segments=[hook, rest])
    out = fill_with_footage(plan, m, 120)
    assert out.segments[0].purpose == "hook"
    assert [s.clip_id for s in out.segments][1] == "ride1"
