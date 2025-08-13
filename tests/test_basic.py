"""
Basic tests for LLM Video Editor components.
"""
import pytest
from pathlib import Path
import tempfile
import json

from llm_video_editor.presets import get_platform_specs, get_available_platforms
from llm_video_editor.utils import (
    is_video_file, is_audio_file, validate_input_path,
    create_output_filename, get_safe_filename
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


class TestWorkflowComponents:
    """Test workflow component initialization."""
    
    def test_import_core_modules(self):
        """Test that core modules can be imported."""
        from llm_video_editor.core.media_probe import MediaProbe
        from llm_video_editor.core.asr import ASRProcessor
        from llm_video_editor.core.scene_detection import SceneDetector
        from llm_video_editor.core.planner import VideoPlanner
        from llm_video_editor.core.workflow import create_workflow
        
        # Test that classes can be instantiated
        asr = ASRProcessor(model_size="tiny")
        assert asr.model_size == "tiny"
        
        detector = SceneDetector(threshold=30.0)
        assert detector.threshold == 30.0
        
        # Test workflow creation (skip if no OpenAI API key)
        try:
            workflow = create_workflow(asr_model="tiny")
            assert workflow is not None
        except Exception as e:
            # Expected when no API key is available
            if "api_key" in str(e).lower():
                print("Skipping workflow test - no OpenAI API key available")
            else:
                raise e


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config_structure(self):
        """Test that example config has valid structure."""
        config_path = Path(__file__).parent.parent / "examples" / "config.json"
        
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            # Check required sections
            assert "asr_model" in config
            assert "scene_threshold" in config
            assert "platforms" in config
            
            # Check platform configs
            assert "youtube" in config["platforms"]
            assert "reels" in config["platforms"]


if __name__ == "__main__":
    pytest.main([__file__])