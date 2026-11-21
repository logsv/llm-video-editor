"""Tests for the long-form EDL logic (pure Python, no ffmpeg needed)."""
from llm_video_editor.longform.edl import (
    build_plan, captions_srt, chapters_text, map_time, runs, tighten_range, validate_chapters,
)


def words_at(*spans, text="w"):
    return [{"w": f"{text}{i}", "s": s, "e": e} for i, (s, e) in enumerate(spans)]


def test_tighten_removes_long_pauses_only():
    w = words_at((1.0, 1.4), (1.6, 2.0), (4.0, 4.5))  # 0.2s gap kept, 2.0s gap removed
    assert tighten_range(0, 10, w, max_gap=0.6, pad_before=0.1, pad_after=0.1) == [(0.9, 2.1), (3.9, 4.6)]


def test_tighten_keeps_range_without_words():
    assert tighten_range(5, 9, words_at((1, 2))) == [(5, 9)]


def test_tighten_drops_fillers():
    w = [{"w": "hello", "s": 1.0, "e": 1.4}, {"w": "um,", "s": 1.9, "e": 2.3}, {"w": "world", "s": 2.8, "e": 3.2}]
    assert len(tighten_range(0, 5, w, max_gap=0.6, drop_fillers=True)) == 2
    assert len(tighten_range(0, 5, w, max_gap=0.6, drop_fillers=False)) == 1


def test_plan_reorder_chapters_and_snapping():
    edl = {"segments": [
        {"in": 100.0, "out": 112.0, "chapter": "Cold open"},
        {"in": 0.01, "out": 30.0, "chapter": "Intro"},
        {"in": 30.0, "out": 60.0, "chapter": "Main"},
    ]}
    plan = build_plan(edl, [], fps=30.0, source_duration=200.0)
    assert [r.src_in for r in plan.ranges] == [100.0, 0.0, 30.0]  # 0.01 snapped to frame 0
    assert [c[1] for c in plan.chapters] == ["Cold open", "Intro", "Main"]
    assert plan.chapters[1][0] == 12.0
    assert len(runs(plan.ranges)) == 2  # cold open jumps backwards -> separate pass
    assert chapters_text(plan.chapters).splitlines()[0] == "0:00 Cold open"


def test_plan_clips_to_source_and_warns():
    plan = build_plan({"segments": [{"in": 0, "out": 50}, {"in": 80, "out": 90}]}, [], 25.0, 40.0)
    assert len(plan.ranges) == 1 and plan.ranges[0].src_out == 40.0
    assert any("clipped" in w for w in plan.warnings) and any("skipped" in w for w in plan.warnings)


def test_alternate_punch_in():
    w = words_at((1, 2), (4, 5), (7, 8))
    edl = {"tighten": {"enabled": True}, "punch_in": {"mode": "alternate", "zoom": 1.2},
           "segments": [{"in": 0, "out": 10}]}
    assert [r.zoom for r in build_plan(edl, w, 30.0).ranges] == [1.0, 1.2, 1.0]


def test_chapter_rules():
    assert validate_chapters([(0, "a"), (20, "b"), (40, "c")], 60) == []
    warns = validate_chapters([(0, "a"), (5, "b")], 60)
    assert any("at least 3" in w for w in warns) and any("'a'" in w for w in warns)


def test_captions_follow_edited_timeline():
    edl = {"segments": [{"in": 10, "out": 12}, {"in": 0, "out": 2}]}
    plan = build_plan(edl, [], 30.0)
    assert map_time(10.5, plan.ranges) == 0.5 and map_time(1.0, plan.ranges) == 3.0
    assert map_time(5.0, plan.ranges) is None
    w = [{"w": "later.", "s": 10.2, "e": 10.6}, {"w": "first", "s": 0.5, "e": 0.9},
         {"w": "cut", "s": 5.0, "e": 5.3}, {"w": "cloud", "s": 1.0, "e": 1.3}]
    srt = captions_srt(w, plan.ranges, replacements={"cloud": "Claude"})
    assert "later." in srt and "first Claude" in srt and "cut" not in srt
    assert "00:00:02,500 -->" in srt
