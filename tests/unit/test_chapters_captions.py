from llm_video_editor.ffmpeg.graph import fit_filter
from llm_video_editor.schemas import (
    EDL,
    CaptionSettings,
    ClipTranscript,
    EDLClip,
    Segment,
    Transcript,
    Word,
)
from llm_video_editor.stages.captions import build_srt, group_cues, output_words
from llm_video_editor.stages.chapters import (
    chapter_marks,
    chapter_warnings,
    description_block,
    ffmetadata,
)
from llm_video_editor.stages.geometry import crop_window
from llm_video_editor.stages.plan import format_transcript, retake_ids
from tests.fakes import make_segments


def _clip(cid, s, e, chapter=None):
    return EDLClip(clip_id=cid, source_clip_id="va", source_file="/a.mp4", start=s, end=e,
                   chapter=chapter)


def test_chapters_move_first_to_zero_and_follow_output_time():
    edl = EDL(target_platform="youtube", width=1920, height=1080, clips=[
        _clip("c1", 50, 55), _clip("c2", 0, 30, "Intro"), _clip("c3", 30, 60, "Setup"),
        _clip("c4", 60, 75, "Demo")])
    marks = chapter_marks(edl, edl.output_starts())
    assert marks == [(0.0, "Intro"), (35.0, "Setup"), (65.0, "Demo")]
    assert chapter_warnings(marks, 80.0) == []
    assert description_block(marks).splitlines() == ["0:00 Intro", "0:35 Setup", "1:05 Demo"]
    assert "START=35000" in ffmetadata(marks, 80.0)


def test_chapter_rules_are_warnings():
    warns = chapter_warnings([(0.0, "A"), (5.0, "B")], 60.0)
    assert any("at least 3" in w for w in warns) and any("'A'" in w for w in warns)


def test_srt_skips_fillers_and_applies_replacements():
    words = [Word(start=0.0, end=0.3, text="I"), Word(start=0.35, end=0.6, text="um"),
             Word(start=0.7, end=1.0, text="love"), Word(start=1.05, end=1.4, text="cloud.")]
    tr = Transcript(clips=[ClipTranscript(clip_id="va", source_file="/a.mp4", duration=5, segments=[
        Segment(id=0, start=0, end=1.4, text="I um love cloud.", words=words)])])
    edl = EDL(target_platform="youtube", width=1920, height=1080, clips=[_clip("c1", 0, 2)],
              captions=CaptionSettings(replacements={"cloud": "Claude"}))
    srt = build_srt(group_cues(output_words(edl, tr, [0.0], [2.0]), 10))
    assert srt.startswith("1\n00:00:00,000 --> ")
    assert "I love Claude." in srt and "um" not in srt


def test_zoom_punches_in_and_geometry_agrees():
    assert fit_filter(1920, 1080) == fit_filter(1920, 1080, zoom=1.0)
    assert "scale=2150:1210" in fit_filter(1920, 1080, zoom=1.12)
    s1, *_ = crop_window(1920, 1080, 1920, 1080, (0.5, 0.5))
    s2, x0, y0, _ = crop_window(1920, 1080, 1920, 1080, (0.5, 0.5), zoom=1.12)
    assert s2 == 1.12 * s1 and x0 > 0 and y0 > 0


def test_retakes_are_marked():
    segs = make_segments([(0.0, "So the first thing you"), (2.0, "So the first thing you need."),
                          (5.0, "Then install it.")])
    assert retake_ids(segs) == {0}
    text = format_transcript(ClipTranscript(clip_id="va", source_file="/a.mp4", duration=9,
                                            segments=segs))
    assert "[0] 0.0-1.4 So the first thing you (retake?)" in text
    assert "-- pause" not in text.splitlines()[1]
