from __future__ import annotations

import shutil

import pytest

from tests.fixtures.make_fixture import make_clip

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory):
    """A multi-camera folder: mixed fps, resolutions and one silent B-roll clip."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("raw_clips")
    make_clip(d / "a_talk_1080p30.mp4", duration=4, size=(1920, 1080), fps=30, audio="sine")
    make_clip(d / "b_talk_720p60.mov", duration=4, size=(1280, 720), fps=60, audio="sine",
              sample_rate=48000)
    make_clip(d / "c_broll_720p24.mp4", duration=3, size=(1280, 720), fps=24, audio=None)
    (d / "notes.txt").write_text("not media")
    return d


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMVE_HOME", str(tmp_path / ".llmve"))
