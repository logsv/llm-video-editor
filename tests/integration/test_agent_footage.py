import pytest

from llm_video_editor.agent.footage import group_recordings, inspect
from llm_video_editor.ffmpeg.runner import ffmpeg
from tests.conftest import requires_ffmpeg

pytestmark = [requires_ffmpeg, pytest.mark.integration]


def clip(path, *, vf="", pix="yuv420p", trc=None, ac=2, audio=True, fps=30):
    args = ["-f", "lavfi", "-i", f"testsrc2=s=320x240:r={fps}:d=1"]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=d=1", "-ac", str(ac), "-c:a", "aac"]
    args += ["-c:v", "libx264", "-pix_fmt", pix]
    if trc:
        args += ["-color_primaries", "bt2020", "-color_trc", trc, "-colorspace", "bt2020nc"]
    if audio:
        args += ["-shortest"]
    ffmpeg(*args, str(path), timeout=60)
    return path


def test_hdr_bit_depth_audio_and_fps_are_reported(tmp_path):
    hlg = inspect(clip(tmp_path / "hlg.mp4", pix="yuv420p10le", trc="arib-std-b67"))
    assert hlg.hdr == "HLG" and hlg.bit_depth == 10
    assert any("tone-map" in i for i in hlg.issues)
    flat = inspect(clip(tmp_path / "flat.mp4", pix="yuv420p10le"))
    assert flat.hdr is None and any("log/flat" in i for i in flat.issues)
    mono = inspect(clip(tmp_path / "mono.mp4", ac=1, fps=60))
    assert "mono audio" in mono.issues and any("slow motion" in i for i in mono.issues)
    silent = inspect(clip(tmp_path / "silent.mp4", audio=False))
    assert "no audio track" in silent.issues
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video")
    info = inspect(bad)
    assert not info.ok and info.issues[0].startswith("unreadable")


def test_split_recordings_grouped_in_chapter_order(tmp_path):
    infos = [inspect(clip(tmp_path / n)) for n in ("GX020007.MP4", "GX010007.MP4", "GX010008.MP4")]
    groups = group_recordings(infos)
    assert groups == {"gopro-X0007": ["GX010007.MP4", "GX020007.MP4"]}
    lone = next(i for i in infos if i.name == "GX010008.MP4")
    assert not any("split" in i for i in lone.issues)
