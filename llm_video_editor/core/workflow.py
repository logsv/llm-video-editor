"""
LangGraph workflow implementation for the video editing pipeline.
"""
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from pathlib import Path
import tempfile
import os

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from .media_probe import MediaProbe, MediaInfo
from .asr import ASRProcessor, TranscriptSegment
from .scene_detection import SceneDetector, Scene
from .planner import VideoPlanner, EditDecisionList
from .ollama_planner import OllamaVideoPlanner, create_ollama_planner
from .renderer import VideoRenderer


class VideoEditingState(TypedDict):
    """State for video editing workflow."""
    # Input
    inputs: Dict[str, Any]
    
    # Media analysis
    media_info: Optional[MediaInfo]
    transcript_segments: List[TranscriptSegment]
    scenes: List[Scene]
    
    # Planning
    edl: Optional[EditDecisionList]
    
    # Processing artifacts
    artifacts: Dict[str, str]  # Paths to generated files
    
    # Status and metadata
    status: str
    metadata: Dict[str, Any]  # Metadata and completion flags


class VideoEditingWorkflow:
    """LangGraph-based video editing workflow."""
    
    def __init__(
        self,
        asr_model: str = "large-v3",
        scene_threshold: float = 27.0,
        planner_model: str = "llama3.2:latest",
        use_ollama: bool = True,
        ollama_base_url: str = "http://localhost:11434",
        enable_rendering: bool = True
    ):
        """
        Initialize workflow.
        
        Args:
            asr_model: Whisper model size for ASR
            scene_threshold: Threshold for scene detection
            planner_model: LLM model for planning (OpenAI) or Ollama model name
            use_ollama: Whether to use local Ollama instead of OpenAI
            ollama_base_url: Ollama server URL
            enable_rendering: Whether to render the final video
        """
        self.asr_processor = ASRProcessor(model_size=asr_model)
        self.scene_detector = SceneDetector(threshold=scene_threshold)
        self.enable_rendering = enable_rendering
        
        if use_ollama:
            self.planner = OllamaVideoPlanner(model_name=planner_model, base_url=ollama_base_url)
            print(f"Using Ollama model: {planner_model}")
        else:
            self.planner = VideoPlanner(model_name=planner_model)
            print(f"Using OpenAI model: {planner_model}")
        
        # Initialize renderer if enabled
        if self.enable_rendering:
            self.renderer = VideoRenderer(
                use_gpu=True,
                enable_smart_reframing=False,  # Disable for now to avoid YOLO dependency
                enable_music_ducking=False,    # Disable for now to avoid Demucs dependency  
                enable_qc=True
            )
        
        # Create workflow graph
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(VideoEditingState)
        
        # Add nodes
        workflow.add_node("probe", self.probe_node)
        workflow.add_node("asr", self.asr_node)
        workflow.add_node("scene_detection", self.scenes_node)
        workflow.add_node("plan", self.planner_node)
        workflow.add_node("validate", self.validate_node)
        
        # Add rendering node if enabled
        if self.enable_rendering:
            workflow.add_node("render", self.render_node)
        
        # Add edges - Sequential processing to avoid state conflicts
        workflow.set_entry_point("probe")
        workflow.add_edge("probe", "asr")
        workflow.add_edge("asr", "scene_detection")  # Sequential processing
        workflow.add_edge("scene_detection", "plan")
        workflow.add_edge("plan", "validate")
        
        if self.enable_rendering:
            workflow.add_edge("validate", "render")
            workflow.add_edge("render", END)
        else:
            workflow.add_edge("validate", END)
        
        return workflow
    
    def probe_node(self, state: VideoEditingState) -> VideoEditingState:
        """Probe media file and extract metadata."""
        try:
            video_path = state["inputs"]["path"]
            print(f"Probing media file: {video_path}")
            
            media_info = MediaProbe.probe_file(video_path)
            
            state["media_info"] = media_info
            state["status"] = "media_probed"
            state["metadata"]["probe_complete"] = True
            
            print(f"Media info: {media_info.duration:.1f}s, {media_info.width}x{media_info.height}, {media_info.fps}fps")
            
        except Exception as e:
            state["status"] = "error"
            state["metadata"]["error"] = f"Probe failed: {str(e)}"
            raise
        
        return state
    
    def asr_node(self, state: VideoEditingState) -> VideoEditingState:
        """Perform automatic speech recognition."""
        try:
            video_path = state["inputs"]["path"]
            print("Starting ASR processing...")
            
            # Extract audio if needed
            temp_audio = None
            if state["media_info"] and state["media_info"].has_audio:
                if not video_path.lower().endswith(('.wav', '.mp3', '.flac')):
                    temp_audio = self.asr_processor.extract_audio_for_transcription(video_path)
                    audio_path = temp_audio
                else:
                    audio_path = video_path
            else:
                print("No audio detected, skipping ASR")
                state["transcript_segments"] = []
                state["metadata"]["asr_complete"] = True
                return state
            
            # Transcribe
            segments = self.asr_processor.transcribe_file(
                audio_path,
                language=state["inputs"].get("language"),
                vad_filter=True,
                word_timestamps=True
            )
            
            # Export SRT file
            srt_path = os.path.join(
                state.get("artifacts", {}).get("output_dir", "output"),
                "captions.srt"
            )
            Path(srt_path).parent.mkdir(parents=True, exist_ok=True)
            self.asr_processor.export_srt(segments, srt_path)
            
            state["transcript_segments"] = segments
            state["artifacts"]["srt_file"] = srt_path
            state["metadata"]["asr_complete"] = True
            
            print(f"ASR complete: {len(segments)} segments, exported to {srt_path}")
            
            # Cleanup temporary audio
            if temp_audio:
                self.asr_processor.cleanup_temp_audio(temp_audio)
                
        except Exception as e:
            state["metadata"]["asr_error"] = f"ASR failed: {str(e)}"
            print(f"ASR error: {str(e)}")
            # Don't set status to error - let the workflow continue
            state["metadata"]["asr_complete"] = True  # Mark as complete even if failed
        
        return state
    
    def scenes_node(self, state: VideoEditingState) -> VideoEditingState:
        """Detect scene boundaries."""
        try:
            video_path = state["inputs"]["path"]
            print("Starting scene detection...")
            
            scenes = self.scene_detector.detect_scenes(video_path, method="content")
            
            # Export scene list
            scene_list_path = os.path.join(
                state.get("artifacts", {}).get("output_dir", "output"),
                "scenes.json"
            )
            Path(scene_list_path).parent.mkdir(parents=True, exist_ok=True)
            self.scene_detector.export_scene_list(scenes, scene_list_path, format="json")
            
            state["scenes"] = scenes
            state["artifacts"]["scene_list"] = scene_list_path
            state["metadata"]["scenes_complete"] = True
            
            print(f"Scene detection complete: {len(scenes)} scenes, exported to {scene_list_path}")
            
        except Exception as e:
            state["metadata"]["scenes_error"] = f"Scene detection failed: {str(e)}"
            print(f"Scene detection error: {str(e)}")
            # Don't set status to error - let the workflow continue
            state["metadata"]["scenes_complete"] = True  # Mark as complete even if failed
        
        return state
    
    def planner_node(self, state: VideoEditingState) -> VideoEditingState:
        """Generate Edit Decision List using LLM planner."""
        try:
            # Wait for both ASR and scene detection to complete
            if not state["metadata"].get("asr_complete") or not state["metadata"].get("scenes_complete"):
                print("Waiting for ASR and scene detection to complete...")
                return state
            
            # Check if there were errors in preprocessing
            if state["metadata"].get("asr_error") and state["metadata"].get("scenes_error"):
                print("Both ASR and scene detection failed, cannot continue planning")
                state["status"] = "error"
                state["metadata"]["error"] = "Both ASR and scene detection failed"
                return state
            
            print("Starting LLM planning...")
            
            user_prompt = state["inputs"]["prompt"]
            target_platform = state["inputs"].get("target", "youtube")
            
            # Prepare data for planner - handle missing data gracefully
            transcript_data = []
            if not state["metadata"].get("asr_error"):
                transcript_data = [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "words": seg.words
                    }
                    for seg in state["transcript_segments"]
                ]
            else:
                print("Warning: No transcript available, planning without speech data")
            
            scene_data = []
            if not state["metadata"].get("scenes_error"):
                scene_data = [
                    {
                        "start_time": scene.start_time,
                        "end_time": scene.end_time,
                        "duration": scene.duration
                    }
                    for scene in state["scenes"]
                ]
            else:
                print("Warning: No scene data available, planning without scene boundaries")
            
            media_data = {
                "filepath": state["media_info"].filepath,
                "duration": state["media_info"].duration,
                "width": state["media_info"].width,
                "height": state["media_info"].height,
                "fps": state["media_info"].fps,
                "aspect_ratio": state["media_info"].aspect_ratio
            }
            
            # Generate EDL
            edl = self.planner.generate_edl(
                prompt=user_prompt,
                transcript_segments=transcript_data,
                scenes=scene_data,
                media_info=media_data,
                target_platform=target_platform
            )
            
            # Export EDL
            edl_path = os.path.join(
                state.get("artifacts", {}).get("output_dir", "output"),
                "edit_decision_list.json"
            )
            Path(edl_path).parent.mkdir(parents=True, exist_ok=True)
            self.planner.export_edl(edl, edl_path, format="json")
            
            state["edl"] = edl
            state["artifacts"]["edl_file"] = edl_path
            state["metadata"]["planning_complete"] = True
            
            print(f"Planning complete: {len(edl.clips)} clips, target duration: {edl.target_duration:.1f}s")
            print(f"EDL exported to: {edl_path}")
            
        except Exception as e:
            state["status"] = "error"
            state["metadata"]["error"] = f"Planning failed: {str(e)}"
            raise
        
        return state
    
    def validate_node(self, state: VideoEditingState) -> VideoEditingState:
        """Validate the generated EDL and perform quality checks."""
        try:
            print("Validating EDL...")
            
            edl = state["edl"]
            media_info = state["media_info"]
            
            # Basic validation checks
            validation_results = {
                "total_clips": len(edl.clips),
                "target_duration": edl.target_duration,
                "source_duration": media_info.duration,
                "compression_ratio": edl.target_duration / media_info.duration,
                "checks": {}
            }
            
            # Check for overlapping clips
            clips_sorted = sorted(edl.clips, key=lambda x: x.start_time)
            overlaps = []
            for i in range(len(clips_sorted) - 1):
                if clips_sorted[i].end_time > clips_sorted[i+1].start_time:
                    overlaps.append((clips_sorted[i].clip_id, clips_sorted[i+1].clip_id))
            validation_results["checks"]["overlapping_clips"] = overlaps
            
            # Check for clips outside source duration
            out_of_bounds = []
            for clip in edl.clips:
                if clip.start_time < 0 or clip.end_time > media_info.duration:
                    out_of_bounds.append(clip.clip_id)
            validation_results["checks"]["out_of_bounds_clips"] = out_of_bounds
            
            # Check duration constraints
            platform_specs = self.planner._get_platform_specs(edl.target_platform)
            duration_valid = edl.target_duration <= platform_specs["max_duration"]
            validation_results["checks"]["duration_within_limits"] = duration_valid
            
            # Export validation report
            validation_path = os.path.join(
                state.get("artifacts", {}).get("output_dir", "output"),
                "validation_report.json"
            )
            Path(validation_path).parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(validation_path, 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            state["artifacts"]["validation_report"] = validation_path
            state["metadata"]["validation_results"] = validation_results
            state["status"] = "completed"
            
            print("Validation complete:")
            print(f"  - {validation_results['total_clips']} clips")
            print(f"  - Target duration: {validation_results['target_duration']:.1f}s")
            print(f"  - Compression ratio: {validation_results['compression_ratio']:.2%}")
            print(f"  - Validation report: {validation_path}")
            
        except Exception as e:
            state["status"] = "error"
            state["metadata"]["error"] = f"Validation failed: {str(e)}"
            raise
        
        return state
    
    def render_node(self, state: VideoEditingState) -> VideoEditingState:
        """Render the final video using the EDL."""
        try:
            print("Starting video rendering...")
            
            edl = state["edl"]
            media_info = state["media_info"]
            output_dir = state["artifacts"]["output_dir"]
            target_platform = state["inputs"].get("target", "youtube")
            
            # Generate output filename
            input_path = Path(state["inputs"]["path"])
            output_filename = f"{input_path.stem}_{target_platform}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"Rendering video to: {output_path}")
            print(f"Using EDL with {len(edl.clips)} clips")
            
            # Render the video
            render_results = self.renderer.render_edl(
                edl=edl,
                source_file=state["inputs"]["path"],
                output_dir=output_dir
            )
            
            # Update state with rendered video path
            final_video = render_results.get("final_video")
            if final_video:
                state["artifacts"]["rendered_video"] = final_video
                state["artifacts"]["clip_files"] = render_results.get("clips", [])
                if render_results.get("subtitle_file"):
                    state["artifacts"]["subtitle_file"] = render_results["subtitle_file"]
                state["metadata"]["rendering_complete"] = True
                print(f"✅ Video rendering completed: {final_video}")
            else:
                raise RuntimeError("No final video was generated by renderer")
            
        except Exception as e:
            print(f"❌ Rendering failed: {str(e)}")
            state["status"] = "error"
            state["metadata"]["error"] = f"Rendering failed: {str(e)}"
            # Don't raise - let the workflow continue for debugging
        
        return state
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete video editing workflow.
        
        Args:
            inputs: Input parameters with keys:
                - path: Path to input video file
                - prompt: User editing prompt
                - target: Target platform ("youtube", "reels", "tiktok")
                - language: Optional language code for ASR
                - output_dir: Optional output directory
        
        Returns:
            Dictionary with workflow results and artifacts
        """
        # Initialize state
        output_dir = inputs.get("output_dir", "output")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        initial_state: VideoEditingState = {
            "inputs": inputs,
            "media_info": None,
            "transcript_segments": [],
            "scenes": [],
            "edl": None,
            "artifacts": {"output_dir": output_dir},
            "status": "starting",
            "metadata": {
                "probe_complete": False,
                "asr_complete": False,
                "scenes_complete": False,
                "planning_complete": False
            }
        }
        
        # Run workflow
        print("Starting video editing workflow...")
        final_state = self.app.invoke(initial_state)
        
        # Prepare results
        results = {
            "status": final_state["status"],
            "artifacts": final_state["artifacts"],
            "metadata": final_state["metadata"]
        }
        
        if final_state.get("edl"):
            results["edl_summary"] = {
                "clips_count": len(final_state["edl"].clips),
                "target_duration": final_state["edl"].target_duration,
                "target_platform": final_state["edl"].target_platform
            }
        
        return results


def create_workflow(
    asr_model: str = "large-v3",
    scene_threshold: float = 27.0,
    planner_model: str = "llama3.2:latest",
    use_ollama: bool = True,
    ollama_base_url: str = "http://localhost:11434",
    enable_rendering: bool = True
) -> VideoEditingWorkflow:
    """
    Factory function to create a video editing workflow.
    
    Args:
        asr_model: Whisper model size
        scene_threshold: Scene detection threshold
        planner_model: LLM model for planning
        use_ollama: Whether to use local Ollama instead of OpenAI
        ollama_base_url: Ollama server URL
        enable_rendering: Whether to render the final video
        
    Returns:
        VideoEditingWorkflow instance
    """
    return VideoEditingWorkflow(
        asr_model=asr_model,
        scene_threshold=scene_threshold,
        planner_model=planner_model,
        use_ollama=use_ollama,
        ollama_base_url=ollama_base_url,
        enable_rendering=enable_rendering
    )