"""
Basic tests for LLM Video Editor components.
"""

import pytest

from llm_video_editor.presets import get_available_platforms, get_platform_specs
from llm_video_editor.utils import (
    create_output_filename,
    get_safe_filename,
    is_audio_file,
    is_video_file,
)


class TestPlatformPresets:
    """Test platform preset functionality."""

    def test_get_available_platforms(self):
        """Test getting available platforms."""
        platforms = get_available_platforms()
        assert isinstance(platforms, list)
        assert len(platforms) > 0
        assert "youtube" in platforms
        assert "reels" in platforms

    def test_youtube_specs(self):
        """Test YouTube platform specifications."""
        specs = get_platform_specs("youtube")
        assert specs["aspect_ratio"] == "16:9"
        assert specs["resolution"] == "1920x1080"
        assert specs["max_duration"] > 0

    def test_reels_specs(self):
        """Test Instagram Reels specifications."""
        specs = get_platform_specs("reels")
        assert specs["aspect_ratio"] == "9:16"
        assert specs["resolution"] == "1080x1920"
        assert specs["max_duration"] == 90

    def test_invalid_platform(self):
        """Test handling of invalid platform."""
        with pytest.raises(ValueError):
            get_platform_specs("invalid_platform")


class TestFileUtils:
    """Test file utility functions."""

    def test_is_video_file(self):
        """Test video file detection."""
        assert is_video_file("test.mp4") is True
        assert is_video_file("test.avi") is True
        assert is_video_file("test.mov") is True
        assert is_video_file("test.txt") is False
        assert is_video_file("test.jpg") is False

    def test_is_audio_file(self):
        """Test audio file detection."""
        assert is_audio_file("test.wav") is True
        assert is_audio_file("test.mp3") is True
        assert is_audio_file("test.aac") is True
        assert is_audio_file("test.mp4") is False
        assert is_audio_file("test.txt") is False

    def test_create_output_filename(self):
        """Test output filename creation."""
        output = create_output_filename(
            "input/video.mp4",
            suffix="edited",
            extension=".mp4"
        )
        assert "video_edited.mp4" in output

        output = create_output_filename(
            "input/video.mov",
            suffix="reels",
            extension=".mp4",
            output_dir="output"
        )
        assert output.endswith("video_reels.mp4")
        assert "output" in output

    def test_get_safe_filename(self):
        """Test safe filename generation."""
        assert get_safe_filename("file<name>.mp4") == "file_name_.mp4"
        assert get_safe_filename("file:name.mp4") == "file_name.mp4"
        assert get_safe_filename("file/name.mp4") == "file_name.mp4"
        assert get_safe_filename("normal_name.mp4") == "normal_name.mp4"


class TestPipelineModules:
    """The stage pipeline and CLI import without optional extras."""

    def test_stage_modules_import(self):
        from llm_video_editor.cli.app import app
        from llm_video_editor.stages import broll, cut_audit, geometry, plan, render, review, watch

        assert all(m is not None for m in (broll, cut_audit, geometry, plan, render, review, watch))
        names = {c.name or c.callback.__name__ for c in app.registered_commands}
        assert {"watch", "plan", "cut-audit", "geometry", "render", "run"} <= names



if __name__ == "__main__":
    pytest.main([__file__])
