#!/usr/bin/env python3
"""
Test end-to-end workflow with Ollama LLM.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_video_editor.core.ollama_planner import OllamaVideoPlanner
from llm_video_editor.core.media_probe import MediaProbe
from llm_video_editor.core.asr import ASRProcessor
from llm_video_editor.core.scene_detection import SceneDetector
from llm_video_editor.core.renderer import VideoRenderer

def test_ollama_workflow():
    """Test complete workflow with Ollama LLM."""
    
    print("🚀 Testing end-to-end workflow with Ollama...")
    
    # Input video file
    video_file = "test_media/sample_720p_scenes.mp4"
    output_dir = "test_media/ollama_workflow_test"
    
    if not Path(video_file).exists():
        print(f"❌ Video file not found: {video_file}")
        return False
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Step 1: Probe media file
        print("📊 Step 1: Probing media file...")
        probe = MediaProbe()
        media_info = probe.probe_file(video_file)
        print(f"   Duration: {media_info.duration:.1f}s")
        print(f"   Resolution: {media_info.width}x{media_info.height}")
        print(f"   FPS: {media_info.fps:.1f}")
        
        # Step 2: Scene detection (quick)
        print("🎬 Step 2: Detecting scenes...")
        scene_detector = SceneDetector(threshold=30.0)
        scenes = scene_detector.detect_scenes(video_file)
        print(f"   Found {len(scenes)} scenes")
        for i, scene in enumerate(scenes[:3]):  # Show first 3
            print(f"   Scene {i+1}: {scene.start_time:.1f}s - {scene.end_time:.1f}s")
        
        # Step 3: ASR transcription (using tiny model for speed)
        print("🎤 Step 3: Transcribing audio...")
        asr = ASRProcessor(model_size="tiny")  # Use tiny model for speed
        segments = asr.transcribe_file(video_file, language="en")
        print(f"   Found {len(segments)} speech segments")
        for segment in segments[:2]:  # Show first 2
            print(f"   {segment.start:.1f}s: {segment.text.strip()}")
        
        # Step 4: Generate EDL with Ollama
        print("🤖 Step 4: Generating EDL with Ollama...")
        planner = OllamaVideoPlanner(model_name="llama3.1")
        
        # Create simple prompt
        prompt = (
            f"Create a short Instagram Reels video (9:16) from this {media_info.duration:.0f}s video. "
            f"Select 2 interesting segments, each 5-7 seconds long. "
            f"The video has {len(scenes)} scenes and contains speech. "
            f"Focus on the most engaging parts."
        )
        
        print(f"   Prompt: {prompt}")
        
        # Generate EDL
        edl = planner.generate_edl(
            prompt=prompt,
            media_info=media_info,
            scenes=scenes,
            transcript_segments=segments,
            target_platform="reels"
        )
        
        print(f"   Generated EDL with {len(edl.clips)} clips")
        for i, clip in enumerate(edl.clips):
            print(f"   Clip {i+1}: {clip.start_time:.1f}s - {clip.end_time:.1f}s ({len(clip.operations)} operations)")
        
        # Step 5: Render video
        print("🎥 Step 5: Rendering video...")
        renderer = VideoRenderer(use_gpu=True)
        
        results = renderer.render_edl(edl, video_file, output_dir)
        
        # Check results
        success = True
        
        if results.get("clips"):
            print(f"✅ Created {len(results['clips'])} clips")
        else:
            print("❌ No clips created")
            success = False
        
        if results.get("final_video"):
            final_path = results["final_video"]
            if Path(final_path).exists():
                size = Path(final_path).stat().st_size
                print(f"✅ Final video: {size:,} bytes")
                print(f"   Output: {final_path}")
            else:
                print("❌ Final video missing")
                success = False
        
        return success
        
    except Exception as e:
        print(f"❌ Error in workflow: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ollama_workflow()
    if success:
        print("\n🎉 End-to-end workflow with Ollama completed successfully!")
    else:
        print("\n💥 Workflow failed")
    sys.exit(0 if success else 1)