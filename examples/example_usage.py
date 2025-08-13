#!/usr/bin/env python3
"""
Example usage of LLM Video Editor programmatically.
"""
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_video_editor.core.workflow import create_workflow
from llm_video_editor.presets import get_platform_specs


def main():
    # Example video file (you'll need to provide an actual video file)
    video_file = "example_video.mp4"
    
    if not Path(video_file).exists():
        print("Please provide an actual video file named 'example_video.mp4'")
        return
    
    # Create workflow
    workflow = create_workflow(
        asr_model="large-v3",
        scene_threshold=27.0,
        planner_model="gpt-4"
    )
    
    # Example 1: Create YouTube video
    print("🎬 Creating YouTube video...")
    youtube_inputs = {
        "path": video_file,
        "prompt": "Create a 5-minute highlights video with the most engaging moments, add captions",
        "target": "youtube",
        "language": "en",
        "output_dir": "output/youtube_example"
    }
    
    try:
        result = workflow.run(youtube_inputs)
        print(f"✅ YouTube processing: {result['status']}")
        if result['status'] == 'completed':
            print(f"📊 Generated {result['edl_summary']['clips_count']} clips")
            print(f"🎯 Target duration: {result['edl_summary']['target_duration']:.1f}s")
        
    except Exception as e:
        print(f"❌ YouTube processing failed: {e}")
    
    # Example 2: Create Instagram Reel
    print("\\n📱 Creating Instagram Reel...")
    reels_inputs = {
        "path": video_file,
        "prompt": "Make a 30-second vertical reel with the top 3 key points, upbeat energy, center captions",
        "target": "reels",
        "language": "en",
        "output_dir": "output/reels_example"
    }
    
    try:
        result = workflow.run(reels_inputs)
        print(f"✅ Reels processing: {result['status']}")
        if result['status'] == 'completed':
            print(f"📊 Generated {result['edl_summary']['clips_count']} clips")
            print(f"🎯 Target duration: {result['edl_summary']['target_duration']:.1f}s")
        
    except Exception as e:
        print(f"❌ Reels processing failed: {e}")
    
    # Show platform specifications
    print("\\n📋 Platform Specifications:")
    for platform in ["youtube", "reels", "tiktok"]:
        specs = get_platform_specs(platform)
        print(f"{platform.upper()}: {specs['resolution']} @ {specs['aspect_ratio']}, max {specs['max_duration']}s")


if __name__ == "__main__":
    main()