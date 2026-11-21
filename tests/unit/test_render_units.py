import pytest

from llm_video_editor.ffmpeg.encoders import detect_encoder, video_args
from llm_video_editor.ffmpeg.graph import (
    Graph,
    OverlaySpec,
    Segment,
    add_overlays,
    add_segments,
    fit_filter,
    quantize,
    snap_to_frame,
)
from llm_video_editor.ffmpeg.loudnorm import apply_filter, is_silent, parse_loudnorm_json
from llm_video_editor.schemas import EDL, ClipTranscript, EDLClip, Transcript
from llm_video_editor.stages.captions import build_ass, group_cues, output_words
from tests.conftest import requires_ffmpeg
from tests.fakes import make_segments

LOUDNORM_STDERR = """
[Parsed_loudnorm_0 @ 0x55]
{
	"input_i" : "-27.61",
	"input_tp" : "-9.85",
	"input_lra" : "3.20",
	"input_thresh" : "-37.90",
	"output_i" : "-14.10",
	"output_tp" : "-1.50",
	"output_lra" : "2.90",
	"output_thresh" : "-24.30",
	"normalization_type" : "dynamic",
	"target_offset" : "0.10"
}
"""


def test_quantize_and_snap():
    assert quantize(1.0, 30) == 1.0
    assert quantize(1.01, 30) == 30 / 30
    assert quantize(0.001, 30) == 1 / 30
    assert snap_to_frame(2.13, 60) == pytest.approx(128 / 60)


def test_fit_never_stretches():
    cover = fit_filter(1080, 1920)
    contain = fit_filter(1080, 1920, "contain")
    for f in (cover, contain):
        assert "force_original_aspect_ratio" in f
    assert "crop=1080:1920" in cover and "pad=1080:1920" in contain
    assert "0.2500*iw" in fit_filter(1080, 1920, center=(0.25, 0.5))


def test_segment_graph_normalises_every_input():
    g = Graph()
    segs = [Segment("/a.mp4", 1.0, 2.0, True), Segment("/b.mov", 0.0, 1.5, False)]
    v, a = add_segments(g, segs, width=1080, height=1920, fps=30)
    fc = g.filter_complex
    assert g.input_args[:6] == ["-ss", "1.000000", "-t", "2.500000", "-i", "/a.mp4"]
    assert fc.count("fps=30") == 2 and fc.count("setsar=1") == 2
    assert fc.count("channel_layouts=stereo") == 2 and "anullsrc=r=48000:cl=stereo" in fc
    assert "apad=whole_dur=2.000000,atrim=duration=2.000000" in fc
    assert "trim=duration=1.500000" in fc
    assert fc.endswith(f"concat=n=2:v=1:a=1[{v}][{a}]")
    base = add_overlays(g, v, [OverlaySpec("/c.mp4", 0.5, 1.0, 0.8)], width=1080, height=1920,
                        fps=30)
    assert base == "vb0" and "setpts=PTS-STARTPTS+1.000000/TB" in g.filter_complex
    assert g.n_inputs == 3


def test_loudnorm_parsing_and_second_pass():
    m = parse_loudnorm_json("noise\n" + LOUDNORM_STDERR)
    assert m["input_i"] == -27.61 and m["target_offset"] == 0.1
    f = apply_filter(m)
    assert f.startswith("loudnorm=I=-14.0:TP=-1.5:LRA=11.0:measured_I=-27.61")
    assert "linear=true" in f
    silent = parse_loudnorm_json(LOUDNORM_STDERR.replace('"-27.61"', '"-inf"'))
    assert is_silent(silent) and apply_filter(silent) == "anull"
    with pytest.raises(ValueError):
        parse_loudnorm_json("no json here")


def test_video_args_force_cfr_and_gop():
    args = video_args("libx264", 30)
    assert args[args.index("-r") + 1] == "30" and args[args.index("-g") + 1] == "60"
    assert "h264_nvenc" in video_args("h264_nvenc", 30)


@requires_ffmpeg
def test_detect_encoder_falls_back_to_working_one():
    assert detect_encoder() in ("h264_nvenc", "h264_videotoolbox", "libx264")
    assert detect_encoder("libx264") == "libx264"


def test_captions_follow_output_timeline():
    tr = Transcript(clips=[ClipTranscript(clip_id="va", source_file="/a", duration=10,
                                          segments=make_segments([(1.0, "One two three four."),
                                                                  (3.0, "Five six.")]))])
    edl = EDL(target_platform="reels", width=1080, height=1920, clips=[
        EDLClip(clip_id="c1", source_clip_id="va", source_file="/a", start=3.0, end=3.6),
        EDLClip(clip_id="c2", source_clip_id="va", source_file="/a", start=0.9, end=2.3),
    ])
    words = output_words(edl, tr, [0.0, 0.6], [0.6, 1.4])
    assert [w[2] for w in words] == ["Five", "six.", "One", "two", "three", "four."]
    assert words[0][0] == 0.0 and words[2][0] == pytest.approx(0.7)
    cues = group_cues(words, max_words=3)
    assert [c.text for c in cues] == ["Five six.", "One two three", "four."]
    ass = build_ass(cues, 1080, 1920, style="word_highlight")
    assert "PlayResY: 1920" in ass and ass.count("Dialogue:") == 3 and "\\k" in ass
