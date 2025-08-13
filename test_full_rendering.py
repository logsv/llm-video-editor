#!/usr/bin/env python3
"""
Test full rendering workflow with the concatenation fix.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_video_editor.core.renderer import VideoRenderer
from llm_video_editor.core.planner import EDLClip, EditDecisionList, EditOperation

def create_test_edl():
    """Create test EDL for rendering."""
    
    # Create clips with reframing operations
    clips = [
        EDLClip(
            clip_id="clip_001",
            source="test_media/sample_720p_scenes.mp4",
            start_time=5.0,
            end_time=15.0,
            duration=10.0,
            operations=[
                EditOperation(
                    type="reframe",
                    params={"target_aspect": "9:16"}
                ),
                EditOperation(
                    type="subtitle",
                    params={"text": "First clip", "start": 0, "end": 5}
                )
            ]
        ),
        EDLClip(
            clip_id="clip_002",
            source="test_media/sample_720p_scenes.mp4", 
            start_time=20.0,
            end_time=25.0,
            duration=5.0,
            operations=[
                EditOperation(
                    type="reframe",
                    params={"target_aspect": "9:16"}
                ),
                EditOperation(
                    type="subtitle",
                    params={"text": "Second clip", "start": 0, "end": 3}
                )
            ]
        )
    ]
    
    return EditDecisionList(
        clips=clips,
        target_platform="reels",
        target_duration=15.0,
        global_operations=[],
        metadata={"created_by": "test"}
    )

def test_full_rendering():
    """Test complete rendering workflow."""
    
    print("🎬 Testing complete video rendering workflow...")
    
    # Input file
    source_file = "test_media/sample_720p_scenes.mp4"
    output_dir = "test_media/full_render_test"
    
    if not Path(source_file).exists():
        print(f"❌ Source file not found: {source_file}")
        return False
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create test EDL
    edl = create_test_edl()
    
    # Create renderer
    renderer = VideoRenderer(use_gpu=True)
    
    try:
        print(f"   Source: {source_file}")
        print(f"   Output: {output_dir}")
        print(f"   Platform: {edl.target_platform}")
        print(f"   Clips: {len(edl.clips)}")
        
        # Render EDL
        results = renderer.render_edl(edl, source_file, output_dir)
        
        # Check results
        success = True
        
        if results.get("clips"):
            print(f"✅ Individual clips created: {len(results['clips'])}")
            for i, clip_path in enumerate(results["clips"]):
                if Path(clip_path).exists():
                    size = Path(clip_path).stat().st_size
                    print(f"   Clip {i+1}: {size:,} bytes")
                else:
                    print(f"❌ Missing clip {i+1}")
                    success = False
        else:
            print("❌ No clips created")
            success = False
        
        if results.get("final_video"):
            final_path = results["final_video"]
            if Path(final_path).exists():
                size = Path(final_path).stat().st_size
                print(f"✅ Final video: {size:,} bytes")
            else:
                print("❌ Final video missing")
                success = False
        else:
            print("❌ Final video not created")
            success = False
        
        if results.get("subtitle_file"):
            sub_path = results["subtitle_file"] 
            if Path(sub_path).exists():
                with open(sub_path, 'r') as f:
                    content = f.read()
                print(f"✅ Subtitle file: {len(content)} characters")
            else:
                print("❌ Subtitle file missing")
        
        return success
        
    except Exception as e:
        print(f"❌ Error during rendering: {e}")
        return False
    finally:
        renderer.cleanup()

if __name__ == "__main__":
    success = test_full_rendering()
    sys.exit(0 if success else 1)