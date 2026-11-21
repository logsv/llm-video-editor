
from llm_video_editor.ffmpeg.probe import probe
from llm_video_editor.ffmpeg.scenes import parse_scene_scores, scenes_from_scores
from llm_video_editor.ffmpeg.sheets import SheetSpec, estimate_tokens, frame_times, make_sheets
from tests.conftest import requires_ffmpeg


def test_token_estimate_and_tile_size():
    spec = SheetSpec()
    assert spec.tile_size(1920, 1080) == (384, 216)
    assert spec.tile_size(1080, 1920) == (216, 384)
    w, h = spec.sheet_size(1920, 1080)
    assert w <= 1568
    assert estimate_tokens(w, h) < 2000
    assert estimate_tokens(3136, 3136) == estimate_tokens(1568, 1568)


def test_frame_times():
    assert frame_times(0, 3, 0.5) == [0.0, 2.0]
    assert frame_times(10, 11, 2) == [10.0, 10.5]


def test_scene_parsing_and_cuts():
    log = "\n".join(
        f"[Parsed_metadata_3 @ 0x1] frame:{i} pts:{i} pts_time:{i * 0.25}\n"
        f"[Parsed_metadata_3 @ 0x1] lavfi.scene_score={0.9 if i == 12 else 0.01}"
        for i in range(40)
    )
    scores = parse_scene_scores(log)
    assert len(scores) == 40
    sa = scenes_from_scores(scores, duration=10.0, threshold=0.3, min_len=1.0)
    assert sa.cuts == [3.0]
    assert sa.scenes == [(0.0, 3.0), (3.0, 10.0)]
    assert 0 < sa.motion_score < 0.1


@requires_ffmpeg
def test_make_sheets_overview_and_window(media_dir, tmp_path):
    src = media_dir / "a_talk_1080p30.mp4"
    res = make_sheets(src, tmp_path, "ov", src_size=(1920, 1080), start=0, end=4,
                      spec=SheetSpec(fps=2, cols=2, rows=2))
    assert len(res) == 2  # 8 frames / 4 per sheet
    assert res[0].tile_times == [0.0, 0.5, 1.0, 1.5]
    p = probe(res[0].path)
    assert (p.width, p.height) == res[0].sheet_size
    dense = make_sheets(src, tmp_path, "win", src_size=(1920, 1080), start=1.0, end=2.0,
                        spec=SheetSpec(fps=2, cols=4, rows=4))
    assert len(dense) == 1 and dense[0].tile_times == [1.0, 1.5]


