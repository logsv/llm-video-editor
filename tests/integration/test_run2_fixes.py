"""Fixes from the second Bhopal -> Khajuraho run: big image replies closed the MCP
connection; templates had no text colour; music levels depended on each file's
mastering; mid-word findings on words ASR invented over wind."""

from __future__ import annotations

from llm_video_editor.agent import mcp_server
from llm_video_editor.ffmpeg.runner import ffmpeg
from llm_video_editor.hyperframes.compose import templates
from llm_video_editor.schemas import MusicTrack, Word
from llm_video_editor.stages.cut_audit import doubtful
from tests.conftest import requires_ffmpeg


@requires_ffmpeg
def test_images_are_shrunk_and_capped(tmp_path):
    paths = []
    for i in range(9):
        p = tmp_path / f"sheet_{i:03d}.png"
        ffmpeg("-f", "lavfi", "-i", f"testsrc2=s=3000x1200:d=1,noise=alls={20 + i}:allf=t",
               "-frames:v", "1", str(p))
        paths.append(str(p))
    images, left_out = mcp_server._images(paths, "stills")
    assert 1 <= len(images) <= mcp_server.MAX_IMAGES and left_out
    assert all(im.mime_type == "image/jpeg" for im in images)
    total = sum(len(im.data) * 3 // 4 for im in images)
    assert total <= mcp_server.MAX_TOTAL_BYTES
    small = mcp_server._shrunk(tmp_path / "sheet_000.png")
    w = ffmpeg("-i", str(small), "-f", "null", "-").stderr
    assert "1568x627" in w or "1568x628" in w
    assert mcp_server._images(paths, "text-only") == ([], [])


def test_every_template_takes_a_text_colour_and_location_a_card():
    t = templates()
    for name in ("location", "label", "lower-third", "callout", "title-card"):
        assert "text_color" in {v["id"] for v in t[name].variables}, name
    assert "style" in {v["id"] for v in t["location"].variables}


def test_music_volume_defaults_are_relative_to_speech():
    assert MusicTrack(track_id="a", file="x", anchor_clip_id="c").volume_db == -10
    assert MusicTrack(track_id="a", file="x", anchor_clip_id="c", role="sfx").volume_db == -4
    assert MusicTrack(track_id="a", file="x", anchor_clip_id="c",
                      role="voiceover").volume_db == 0


def test_doubtful_words():
    assert doubtful(Word(start=0, end=1, text="鱼", prob=0.9), "hi")
    assert not doubtful(Word(start=0, end=1, text="鱼", prob=0.9), "zh")
    assert doubtful(Word(start=0, end=1, text="I'm", prob=0.2), "en")
    assert not doubtful(Word(start=0, end=1, text="road", prob=0.9), "en")


def test_nearby_lookup_gives_up_after_repeated_failures(tmp_path):
    import requests

    from llm_video_editor.agent import places

    calls = []

    def fetch(*a):
        calls.append(a)
        raise requests.Timeout("busy")

    stops = [{"stop": i, "lat": 24.0 + i, "lon": 79.0, "key": f"k{i}"} for i in range(5)]
    _, errs = places.nearby_stops(stops, tmp_path / "n.json", fetch=fetch,
                                  sleep=lambda s: None)
    assert len(calls) == places.GIVE_UP_AFTER and any("skipped the rest" in e for e in errs)


def test_yolo_weights_go_to_the_cache(tmp_path, monkeypatch):
    from llm_video_editor.stages.detectors import yolo_weights

    monkeypatch.setenv("LLMVE_CACHE", str(tmp_path))
    assert yolo_weights("yolo11n.pt") == str(tmp_path / "models" / "yolo11n.pt")
    assert yolo_weights("/abs/custom.pt") == "/abs/custom.pt"
