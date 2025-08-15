"""
Video rendering and cutting functionality for applying EDL operations.
"""
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tempfile

from .planner import EditDecisionList, EDLClip
from .smart_reframing import SmartReframer, fallback_to_autoflip_style
from .music_ducking import MusicDucker, DuckingProfile
from .quality_control import QualityController
from ..utils.ffmpeg_utils import FFmpegProcessor
from ..presets.platform_presets import get_ffmpeg_preset, get_subtitle_style


@dataclass
class RenderJob:
    """Container for a rendering job."""
    input_file: str
    output_file: str
    start_time: float
    end_time: float
    operations: List[Dict[str, Any]]
    platform_preset: Dict[str, Any]


class VideoRenderer:
    """Video renderer that applies EDL operations and creates output videos."""
    
    def __init__(self, use_gpu: bool = True, enable_smart_reframing: bool = True, enable_music_ducking: bool = True, enable_qc: bool = True):
        """
        Initialize video renderer.
        
        Args:
            use_gpu: Whether to use GPU acceleration for rendering
            enable_smart_reframing: Whether to use YOLO-based smart reframing
            enable_music_ducking: Whether to enable music ducking with Demucs
            enable_qc: Whether to perform quality control checks
        """
        self.use_gpu = use_gpu and FFmpegProcessor.check_gpu_acceleration()
        self.temp_files = []  # Track temporary files for cleanup
        
        # Initialize pro features
        self.smart_reframer = SmartReframer() if enable_smart_reframing else None
        self.music_ducker = MusicDucker() if enable_music_ducking else None
        self.quality_controller = QualityController() if enable_qc else None
        
        self.enable_smart_reframing = enable_smart_reframing
        self.enable_music_ducking = enable_music_ducking
        self.enable_qc = enable_qc
    
    def render_edl(
        self,
        edl: EditDecisionList,
        source_file: str,
        output_dir: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, str]:
        """
        Render complete EDL to output videos.
        
        Args:
            edl: Edit Decision List to render
            source_file: Path to source video file
            output_dir: Directory for output files
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with paths to generated files
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get platform preset for encoding
        platform_preset = get_ffmpeg_preset(edl.target_platform, self.use_gpu)
        
        results = {
            "clips": [],
            "final_video": None,
            "subtitle_file": None
        }
        
        print(f"🎬 Rendering EDL for {edl.target_platform}")
        print(f"   Target duration: {edl.target_duration:.1f}s")
        print(f"   Clips to render: {len(edl.clips)}")
        print(f"   Using GPU: {self.use_gpu}")
        
        # Step 1: Render individual clips
        clip_files = []
        total_clips = len(edl.clips)
        
        for i, clip in enumerate(edl.clips):
            if progress_callback:
                progress_callback(i / (total_clips + 1))  # +1 for final concatenation
            
            clip_output = os.path.join(output_dir, f"clip_{i:03d}.mp4")
            
            success = self._render_clip(
                source_file, clip_output, clip, platform_preset
            )
            
            if success:
                clip_files.append(clip_output)
                results["clips"].append(clip_output)
                print(f"   ✅ Rendered clip {i+1}/{total_clips}: {clip.start_time:.1f}-{clip.end_time:.1f}s")
            else:
                print(f"   ❌ Failed to render clip {i+1}/{total_clips}")
        
        # Step 2: Concatenate clips into final video
        if clip_files:
            final_output = os.path.join(output_dir, f"{edl.target_platform}_final.mp4")
            
            if progress_callback:
                progress_callback(total_clips / (total_clips + 1))
            
            success = self._concatenate_clips(clip_files, final_output)
            
            if success:
                # Apply music ducking if enabled
                if self.enable_music_ducking and self.music_ducker:
                    try:
                        ducked_output = os.path.join(output_dir, f"{edl.target_platform}_ducked.mp4")
                        ducking_results = self.music_ducker.create_ducked_mix(
                            final_output, ducked_output
                        )
                        if ducking_results.get("output_video"):
                            results["final_video"] = ducked_output
                            results["ducked_audio"] = ducking_results.get("ducked_music")
                            print(f"   ✅ Music ducking applied: {ducked_output}")
                        else:
                            results["final_video"] = final_output
                            print(f"   ⚠️ Music ducking failed, using original: {final_output}")
                    except Exception as e:
                        print(f"   ⚠️ Music ducking error: {e}")
                        results["final_video"] = final_output
                else:
                    results["final_video"] = final_output
                    
                print(f"   ✅ Final video created: {results['final_video']}")
            else:
                print(f"   ❌ Failed to create final video")
        
        # Step 3: Generate subtitle file if needed
        subtitle_file = self._generate_subtitle_file(edl, output_dir)
        if subtitle_file:
            results["subtitle_file"] = subtitle_file
        
        # Step 4: Perform quality control checks if enabled
        if self.enable_qc and self.quality_controller and results.get("final_video"):
            try:
                print("   🔍 Running quality control checks...")
                qc_report = self.quality_controller.analyze_video_file(
                    results["final_video"],
                    expected_duration=edl.target_duration,
                    check_black_frames=True,
                    check_audio=True
                )
                
                # Export QC report
                qc_report_path = os.path.join(output_dir, "qc_report.json")
                self.quality_controller.export_report(qc_report, qc_report_path, format="json")
                results["qc_report"] = qc_report_path
                
                # Export HTML report for easier viewing
                qc_html_path = os.path.join(output_dir, "qc_report.html")
                self.quality_controller.export_report(qc_report, qc_html_path, format="html")
                results["qc_report_html"] = qc_html_path
                
                if qc_report.passed:
                    print(f"   ✅ Quality control: PASSED")
                else:
                    print(f"   ⚠️ Quality control: FAILED ({len(qc_report.errors)} errors)")
                    for error in qc_report.errors[:3]:  # Show first 3 errors
                        print(f"      - {error}")
                
                if qc_report.warnings:
                    print(f"   ⚠️ QC Warnings: {len(qc_report.warnings)}")
                    for warning in qc_report.warnings[:2]:  # Show first 2 warnings
                        print(f"      - {warning}")
                        
            except Exception as e:
                print(f"   ⚠️ Quality control error: {e}")
        
        if progress_callback:
            progress_callback(1.0)  # Complete
        
        return results
    
    def _render_clip(
        self,
        source_file: str,
        output_file: str,
        clip: EDLClip,
        platform_preset: Dict[str, Any]
    ) -> bool:
        """Render a single clip with operations."""
        duration = clip.end_time - clip.start_time
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-y']  # -y to overwrite output files
        
        # Input with seek
        cmd.extend(['-ss', str(clip.start_time), '-i', source_file])
        
        # Duration
        cmd.extend(['-t', str(duration)])
        
        # Video codec settings
        if self.use_gpu and 'h264_videotoolbox' in platform_preset.get('video_codec', ''):
            cmd.extend(['-c:v', 'h264_videotoolbox'])
            if platform_preset.get('bitrate'):
                cmd.extend(['-b:v', platform_preset['bitrate']])
        else:
            cmd.extend(['-c:v', 'libx264'])
            if platform_preset.get('crf'):
                cmd.extend(['-crf', str(platform_preset['crf'])])
        
        # Audio settings
        cmd.extend(['-c:a', platform_preset.get('audio_codec', 'aac')])
        cmd.extend(['-ar', str(platform_preset.get('audio_sample_rate', 48000))])
        cmd.extend(['-b:a', platform_preset.get('audio_bitrate', '128k')])
        
        # Video filters for operations
        filters = []
        
        # Apply clip operations
        for operation in clip.operations:
            if operation.type == "reframe":
                target_aspect = operation.params.get("target_aspect")
                
                # Use smart reframing if enabled and available
                if self.enable_smart_reframing and self.smart_reframer and target_aspect == "9:16":
                    try:
                        # Apply smart reframing instead of static crop
                        crop_regions = self.smart_reframer.analyze_video_for_reframing(
                            source_file, target_aspect=9/16, sample_interval=60
                        )
                        
                        if crop_regions:
                            # Use first crop region for this clip (could be enhanced for temporal alignment)
                            region = crop_regions[0]
                            filters.append(f"crop={region.width}:{region.height}:{region.x}:{region.y}")
                            filters.append(f"scale={platform_preset['width']}:{platform_preset['height']}")
                        else:
                            # Fallback to center crop
                            filters.append("crop=ih*9/16:ih:(iw-ih*9/16)/2:0")
                            filters.append(f"scale={platform_preset['width']}:{platform_preset['height']}")
                    except Exception as e:
                        print(f"Smart reframing failed, using fallback: {e}")
                        # Fallback to static center crop
                        filters.append("crop=ih*9/16:ih:(iw-ih*9/16)/2:0")
                        filters.append(f"scale={platform_preset['width']}:{platform_preset['height']}")
                elif target_aspect == "9:16":
                    # Static center crop fallback
                    filters.append("crop=ih*9/16:ih:(iw-ih*9/16)/2:0")
                    filters.append(f"scale={platform_preset['width']}:{platform_preset['height']}")
                elif target_aspect == "16:9":
                    # Scale to 16:9
                    filters.append(f"scale={platform_preset['width']}:{platform_preset['height']}")
            
            elif operation.type == "color_correction":
                brightness = operation.params.get("brightness")
                contrast = operation.params.get("contrast")
                saturation = operation.params.get("saturation")
                if brightness or contrast or saturation:
                    eq_filter = "eq="
                    eq_params = []
                    if brightness:
                        eq_params.append(f"brightness={brightness}")
                    if contrast:
                        eq_params.append(f"contrast={contrast}")
                    if saturation:
                        eq_params.append(f"saturation={saturation}")
                    eq_filter += ":".join(eq_params)
                    filters.append(eq_filter)
        
        # Default scaling if no reframe operation
        if not any(op.type == "reframe" for op in clip.operations):
            filters.append(f"scale={platform_preset['width']}:{platform_preset['height']}")
        
        # Apply video filters
        if filters:
            cmd.extend(['-vf', ','.join(filters)])
        
        # Pixel format
        cmd.extend(['-pix_fmt', platform_preset.get('pixel_format', 'yuv420p')])
        
        # Additional flags
        if platform_preset.get('movflags'):
            cmd.extend(['-movflags', platform_preset['movflags']])
        
        # Output file
        cmd.append(output_file)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0 and Path(output_file).exists()
        except subprocess.TimeoutExpired:
            print(f"Timeout rendering clip: {clip.clip_id}")
            return False
        except Exception as e:
            print(f"Error rendering clip {clip.clip_id}: {e}")
            return False
    
    def _concatenate_clips(self, clip_files: List[str], output_file: str) -> bool:
        """Concatenate multiple video clips into final output."""
        if not clip_files:
            return False
        
        # Create temporary file list for FFmpeg concat
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for clip_file in clip_files:
                f.write(f"file '{os.path.abspath(clip_file)}'\n")
            f.flush()  # Ensure file is written
            filelist_path = f.name
            self.temp_files.append(filelist_path)
        
        # Concatenate using FFmpeg
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', filelist_path,
            '-c', 'copy',  # Copy streams without re-encoding
            output_file
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            success = result.returncode == 0 and Path(output_file).exists()
            
            if not success:
                print(f"FFmpeg concat error: {result.stderr}")
            
            return success
            
        except Exception as e:
            print(f"Error concatenating clips: {e}")
            return False
    
    def _generate_subtitle_file(self, edl: EditDecisionList, output_dir: str) -> Optional[str]:
        """Generate subtitle file from EDL subtitle operations."""
        subtitle_entries = []
        current_time = 0.0
        
        for clip in edl.clips:
            for operation in clip.operations:
                if operation.type == "subtitle":
                    start_time = current_time + operation.params.get("start", 0)
                    end_time = current_time + operation.params.get("end", clip.duration)
                    text = operation.params.get("text", "")
                    
                    if text.strip():
                        subtitle_entries.append({
                            "start": start_time,
                            "end": end_time,
                            "text": text
                        })
            
            current_time += clip.duration
        
        if not subtitle_entries:
            return None
        
        # Generate SRT file
        srt_path = os.path.join(output_dir, "subtitles.srt")
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(subtitle_entries, 1):
                start_time = self._seconds_to_srt_time(entry["start"])
                end_time = self._seconds_to_srt_time(entry["end"])
                
                f.write(f"{i}\\n")
                f.write(f"{start_time} --> {end_time}\\n")
                f.write(f"{entry['text']}\\n\\n")
        
        return srt_path
    
    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """Convert seconds to SRT time format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass
        self.temp_files.clear()
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()


def render_video_from_edl(
    edl: EditDecisionList,
    source_file: str,
    output_dir: str,
    use_gpu: bool = True,
    enable_smart_reframing: bool = True,
    enable_music_ducking: bool = True,
    enable_qc: bool = True,
    progress_callback: Optional[callable] = None
) -> Dict[str, str]:
    """
    Convenience function to render video from EDL with pro polish features.
    
    Args:
        edl: Edit Decision List
        source_file: Source video file path
        output_dir: Output directory
        use_gpu: Whether to use GPU acceleration
        enable_smart_reframing: Whether to use YOLO-based smart reframing
        enable_music_ducking: Whether to enable music ducking with Demucs
        enable_qc: Whether to perform quality control checks
        progress_callback: Optional progress callback
        
    Returns:
        Dictionary with output file paths
    """
    renderer = VideoRenderer(
        use_gpu=use_gpu,
        enable_smart_reframing=enable_smart_reframing,
        enable_music_ducking=enable_music_ducking,
        enable_qc=enable_qc
    )
    try:
        return renderer.render_edl(edl, source_file, output_dir, progress_callback)
    finally:
        renderer.cleanup()