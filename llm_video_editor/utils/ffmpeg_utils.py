"""
FFmpeg utilities for video processing and rendering.
"""
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import tempfile

from ..presets.platform_presets import get_ffmpeg_preset, get_subtitle_style


class FFmpegProcessor:
    """Wrapper for FFmpeg operations."""
    
    @staticmethod
    def check_ffmpeg() -> bool:
        """Check if FFmpeg is available."""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    @staticmethod
    def check_gpu_acceleration() -> bool:
        """Check if GPU acceleration is available (NVENC or VideoToolbox)."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-encoders'],
                capture_output=True, text=True, check=True
            )
            # Check for NVENC (NVIDIA) or VideoToolbox (Apple) hardware encoding
            return ('h264_nvenc' in result.stdout or 
                    'h264_videotoolbox' in result.stdout or 
                    'hevc_videotoolbox' in result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    @staticmethod
    def get_video_info(filepath: str) -> Dict[str, Any]:
        """Get detailed video information using ffprobe."""
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            filepath
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to get video info: {e}")
    
    @staticmethod
    def create_video_filter_complex(
        operations: List[Dict[str, Any]],
        input_width: int,
        input_height: int,
        target_width: int,
        target_height: int
    ) -> str:
        """
        Create FFmpeg filter_complex string for video processing.
        
        Args:
            operations: List of video operations to apply
            input_width: Source video width
            input_height: Source video height
            target_width: Target output width
            target_height: Target output height
            
        Returns:
            FFmpeg filter_complex string
        """
        filters = []
        
        # Smart reframing for aspect ratio conversion
        input_aspect = input_width / input_height
        target_aspect = target_width / target_height
        
        if abs(input_aspect - target_aspect) > 0.1:  # Significant aspect ratio change
            if target_aspect < input_aspect:  # Going from wide to tall (e.g., 16:9 to 9:16)
                # Crop to target aspect, then scale
                crop_width = int(input_height * target_aspect)
                crop_x = (input_width - crop_width) // 2
                filters.append(f"crop={crop_width}:{input_height}:{crop_x}:0")
            else:  # Going from tall to wide
                crop_height = int(input_width / target_aspect)
                crop_y = (input_height - crop_height) // 2
                filters.append(f"crop={input_width}:{crop_height}:0:{crop_y}")
        
        # Scale to target resolution
        filters.append(f"scale={target_width}:{target_height}")
        
        # Apply additional operations
        for op in operations:
            if op.get("type") == "reframe":
                # Custom reframing already handled above
                continue
            elif op.get("type") == "color_correction":
                params = op.get("params", {})
                if params.get("brightness"):
                    filters.append(f"eq=brightness={params['brightness']}")
                if params.get("contrast"):
                    filters.append(f"eq=contrast={params['contrast']}")
                if params.get("saturation"):
                    filters.append(f"eq=saturation={params['saturation']}")
            elif op.get("type") == "fade":
                params = op.get("params", {})
                fade_in = params.get("fade_in", 0)
                fade_out = params.get("fade_out", 0)
                if fade_in > 0:
                    filters.append(f"fade=t=in:st=0:d={fade_in}")
                if fade_out > 0:
                    # fade_out timing would need video duration
                    pass
        
        return ",".join(filters) if filters else "scale={target_width}:{target_height}"
    
    @staticmethod
    def create_subtitle_filter(
        srt_file: str,
        style: Dict[str, Any],
        target_width: int,
        target_height: int
    ) -> str:
        """
        Create FFmpeg subtitle filter string.
        
        Args:
            srt_file: Path to SRT subtitle file
            style: Subtitle style parameters
            target_width: Video width
            target_height: Video height
            
        Returns:
            FFmpeg subtitle filter string
        """
        # Escape the file path for FFmpeg
        srt_path_escaped = srt_file.replace(':', '\\\\:').replace(',', '\\\\,')
        
        # Build subtitle filter
        subtitle_filter = f"subtitles='{srt_path_escaped}'"
        
        # Add styling
        font_size = style.get("font_size", 32)
        font_color = style.get("font_color", "white")
        font_family = style.get("font_family", "Arial")
        
        subtitle_filter += f":force_style='FontSize={font_size},PrimaryColour=&H{font_color.replace('#', '')},FontName={font_family}'"
        
        return subtitle_filter
    
    @staticmethod
    def render_video_segment(
        input_file: str,
        output_file: str,
        start_time: float,
        duration: float,
        platform_preset: Dict[str, Any],
        operations: List[Dict[str, Any]] = None,
        srt_file: str = None,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Render a video segment with specified operations.
        
        Args:
            input_file: Source video file
            output_file: Output video file
            start_time: Start time in seconds
            duration: Duration in seconds
            platform_preset: Platform-specific encoding preset
            operations: List of video operations to apply
            srt_file: Optional SRT subtitle file
            progress_callback: Optional progress callback function
            
        Returns:
            True if successful, False otherwise
        """
        if operations is None:
            operations = []
        
        # Create output directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Get input video info
        video_info = FFmpegProcessor.get_video_info(input_file)
        video_stream = next(s for s in video_info['streams'] if s['codec_type'] == 'video')
        input_width = video_stream['width']
        input_height = video_stream['height']
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-y']  # -y to overwrite output files
        
        # Input file with seek
        cmd.extend(['-ss', str(start_time), '-i', input_file])
        
        # Duration
        cmd.extend(['-t', str(duration)])
        
        # Video codec and settings
        if platform_preset.get('video_codec') == 'h264_nvenc':
            cmd.extend(['-c:v', 'h264_nvenc'])
            cmd.extend(['-preset', platform_preset.get('preset', 'medium')])
            if platform_preset.get('bitrate'):
                cmd.extend(['-b:v', platform_preset['bitrate']])
        else:
            cmd.extend(['-c:v', 'libx264'])
            cmd.extend(['-preset', platform_preset.get('preset', 'medium')])
            if platform_preset.get('crf'):
                cmd.extend(['-crf', str(platform_preset['crf'])])
        
        # Audio codec and settings
        cmd.extend(['-c:a', platform_preset.get('audio_codec', 'aac')])
        cmd.extend(['-ar', str(platform_preset.get('audio_sample_rate', 48000))])
        cmd.extend(['-b:a', platform_preset.get('audio_bitrate', '128k')])
        
        # Pixel format
        cmd.extend(['-pix_fmt', platform_preset.get('pixel_format', 'yuv420p')])
        
        # Video filters
        filters = []
        
        # Create video processing filter
        video_filter = FFmpegProcessor.create_video_filter_complex(
            operations,
            input_width, input_height,
            platform_preset['width'], platform_preset['height']
        )
        filters.append(video_filter)
        
        # Add subtitles if provided
        if srt_file and Path(srt_file).exists():
            # Note: subtitle filter must be applied after video filters
            subtitle_filter = FFmpegProcessor.create_subtitle_filter(
                srt_file,
                {"font_size": 32, "font_color": "white"},  # Default style
                platform_preset['width'], platform_preset['height']
            )
            filters.append(subtitle_filter)
        
        if filters:
            cmd.extend(['-vf', ','.join(filters)])
        
        # Frame rate
        if platform_preset.get('fps'):
            cmd.extend(['-r', str(platform_preset['fps'])])
        
        # Additional flags
        if platform_preset.get('movflags'):
            cmd.extend(['-movflags', platform_preset['movflags']])
        
        # Output file
        cmd.append(output_file)
        
        try:
            # Run FFmpeg command
            if progress_callback:
                # Use progress callback if provided
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    universal_newlines=True
                )
                
                # Monitor progress (basic implementation)
                while True:
                    line = process.stderr.readline()
                    if not line:
                        break
                    if 'time=' in line:
                        # Parse time from FFmpeg output
                        time_part = line.split('time=')[1].split()[0]
                        try:
                            # Convert time to seconds and call callback
                            h, m, s = time_part.split(':')
                            current_time = int(h) * 3600 + int(m) * 60 + float(s)
                            progress = min(current_time / duration, 1.0)
                            progress_callback(progress)
                        except:
                            pass
                
                process.wait()
                success = process.returncode == 0
            else:
                # Run without progress monitoring
                result = subprocess.run(cmd, capture_output=True, text=True)
                success = result.returncode == 0
                
                if not success:
                    print(f"FFmpeg error: {result.stderr}")
            
            return success and Path(output_file).exists()
            
        except Exception as e:
            print(f"Error running FFmpeg: {e}")
            return False
    
    @staticmethod
    def apply_audio_normalization(
        input_file: str,
        output_file: str,
        target_lufs: float = -16.0,
        loudrange: float = 11.0,
        true_peak: float = -1.5
    ) -> bool:
        """
        Apply audio normalization using FFmpeg's loudnorm filter.
        
        Args:
            input_file: Input audio/video file
            output_file: Output file
            target_lufs: Target integrated loudness (LUFS)
            loudrange: Target loudness range (LU)
            true_peak: Target true peak (dBTP)
            
        Returns:
            True if successful
        """
        try:
            # Two-pass loudnorm
            # First pass: measure
            cmd1 = [
                'ffmpeg', '-i', input_file,
                '-af', f'loudnorm=I={target_lufs}:LRA={loudrange}:TP={true_peak}:print_format=json',
                '-f', 'null', '-'
            ]
            
            result1 = subprocess.run(cmd1, capture_output=True, text=True)
            if result1.returncode != 0:
                return False
            
            # Extract measurements from output
            output_lines = result1.stderr.split('\\n')
            json_start = False
            json_lines = []
            
            for line in output_lines:
                if '{' in line:
                    json_start = True
                if json_start:
                    json_lines.append(line)
                if '}' in line and json_start:
                    break
            
            if not json_lines:
                # Fallback to single-pass
                cmd = [
                    'ffmpeg', '-y', '-i', input_file,
                    '-af', f'loudnorm=I={target_lufs}:LRA={loudrange}:TP={true_peak}',
                    output_file
                ]
                result = subprocess.run(cmd, capture_output=True)
                return result.returncode == 0
            
            # Parse measurements
            try:
                measurements = json.loads('\\n'.join(json_lines))
                measured_i = measurements.get('input_i', str(target_lufs))
                measured_lra = measurements.get('input_lra', str(loudrange))
                measured_tp = measurements.get('input_tp', str(true_peak))
                measured_thresh = measurements.get('input_thresh', '-70.0')
                target_offset = measurements.get('target_offset', '0.0')
            except:
                # Fallback to single-pass
                cmd = [
                    'ffmpeg', '-y', '-i', input_file,
                    '-af', f'loudnorm=I={target_lufs}:LRA={loudrange}:TP={true_peak}',
                    output_file
                ]
                result = subprocess.run(cmd, capture_output=True)
                return result.returncode == 0
            
            # Second pass: apply normalization with measurements
            cmd2 = [
                'ffmpeg', '-y', '-i', input_file,
                '-af', (f'loudnorm=I={target_lufs}:LRA={loudrange}:TP={true_peak}:'
                       f'measured_I={measured_i}:measured_LRA={measured_lra}:'
                       f'measured_TP={measured_tp}:measured_thresh={measured_thresh}:'
                       f'offset={target_offset}'),
                output_file
            ]
            
            result2 = subprocess.run(cmd2, capture_output=True)
            return result2.returncode == 0
            
        except Exception as e:
            print(f"Audio normalization error: {e}")
            return False
    
    @staticmethod
    def concatenate_video_segments(
        segment_files: List[str],
        output_file: str
    ) -> bool:
        """
        Concatenate multiple video segments into a single file.
        
        Args:
            segment_files: List of video segment file paths
            output_file: Output concatenated video file
            
        Returns:
            True if successful
        """
        if not segment_files:
            return False
        
        try:
            # Create temporary file list for FFmpeg concat
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for segment_file in segment_files:
                    f.write(f"file '{os.path.abspath(segment_file)}'\\n")
                filelist_path = f.name
            
            # Concatenate using FFmpeg
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', filelist_path,
                '-c', 'copy',  # Copy streams without re-encoding
                output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            success = result.returncode == 0
            
            # Clean up temp file
            os.unlink(filelist_path)
            
            return success
            
        except Exception as e:
            print(f"Concatenation error: {e}")
            return False