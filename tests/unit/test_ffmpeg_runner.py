import pytest

from llm_video_editor.ffmpeg.probe import probe
from llm_video_editor.ffmpeg.runner import FFmpegError, ffmpeg, run
from tests.conftest import requires_ffmpeg


def test_run_rejects_shell_strings():
    with pytest.raises(TypeError):
        run("ffmpeg -version")


def test_missing_executable_is_reported():
    with pytest.raises(FFmpegError, match="not found"):
        run(["definitely-not-a-real-binary-xyz"])


@requires_ffmpeg
def test_ffmpeg_error_includes_stderr(tmp_path):
    with pytest.raises(FFmpegError) as e:
        ffmpeg("-i", str(tmp_path / "missing.mp4"), "-f", "null", "-")
    assert "missing.mp4" in str(e.value)


@requires_ffmpeg
def test_probe_mixed_clips(media_dir):
    a = probe(media_dir / "a_talk_1080p30.mp4")
    assert (a.width, a.height, a.fps, a.has_audio) == (1920, 1080, 30.0, True)
    assert a.duration == pytest.approx(4.0, abs=0.1)
    b = probe(media_dir / "b_talk_720p60.mov")
    assert (b.width, b.height, b.fps) == (1280, 720, 60.0)
    c = probe(media_dir / "c_broll_720p24.mp4")
    assert c.has_audio is False and c.fps == 24.0
