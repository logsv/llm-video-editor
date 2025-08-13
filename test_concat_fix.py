#!/usr/bin/env python3
"""
Quick test to verify video concatenation fix.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_video_editor.core.renderer import VideoRenderer
from llm_video_editor.core.planner import EDLClip, EditDecisionList, EditOperation

def test_concatenation_fix():
    """Test the concatenation fix with existing rendered clips."""
    
    output_dir = "test_media/rendered_output"
    clip_files = [
        "test_media/rendered_output/clip_000.mp4",
        "test_media/rendered_output/clip_001.mp4"
    ]
    
    # Check if clip files exist
    for clip_file in clip_files:
        if not Path(clip_file).exists():
            print(f"❌ Missing clip file: {clip_file}")
            return False
    
    # Create renderer and test concatenation directly
    renderer = VideoRenderer(use_gpu=True)
    
    try:
        output_file = os.path.join(output_dir, "concatenation_test.mp4")
        success = renderer._concatenate_clips(clip_files, output_file)
        
        if success and Path(output_file).exists():
            file_size = Path(output_file).stat().st_size
            print(f"✅ Concatenation successful!")
            print(f"   Output: {output_file}")
            print(f"   Size: {file_size:,} bytes")
            return True
        else:
            print("❌ Concatenation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during concatenation: {e}")
        return False
    finally:
        renderer.cleanup()

if __name__ == "__main__":
    print("🔧 Testing concatenation fix...")
    success = test_concatenation_fix()
    sys.exit(0 if success else 1)