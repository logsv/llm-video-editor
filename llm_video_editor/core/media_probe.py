"""
Media probe utilities for extracting video/audio metadata and basic analysis.
"""
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MediaInfo:
    """Container for media file information."""
    filepath: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    audio_sample_rate: Optional[int]
    video_codec: str
    audio_codec: Optional[str]
    bitrate: int
    aspect_ratio: str


class MediaProbe:
    """Utility class for probing media files using FFprobe."""
    
    @staticmethod
    def probe_file(filepath: str) -> MediaInfo:
        """
        Probe a media file and extract metadata.
        
        Args:
            filepath: Path to the media file
            
        Returns:
            MediaInfo object with extracted metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            subprocess.CalledProcessError: If ffprobe fails
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Media file not found: {filepath}")
        
        # Run ffprobe command
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            filepath
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise subprocess.CalledProcessError(
                e.returncode, 
                e.cmd, 
                f"FFprobe failed: {e.stderr}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse ffprobe output: {e}")
        
        return MediaProbe._parse_probe_data(filepath, data)
    
    @staticmethod
    def _parse_probe_data(filepath: str, data: Dict[str, Any]) -> MediaInfo:
        """Parse ffprobe JSON output into MediaInfo object."""
        format_info = data.get('format', {})
        streams = data.get('streams', [])
        
        # Find video and audio streams
        video_stream = None
        audio_stream = None
        
        for stream in streams:
            if stream.get('codec_type') == 'video':
                video_stream = stream
            elif stream.get('codec_type') == 'audio':
                audio_stream = stream
        
        if not video_stream:
            raise ValueError("No video stream found in media file")
        
        # Extract video information
        duration = float(format_info.get('duration', 0))
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        
        # Calculate FPS
        fps_str = video_stream.get('r_frame_rate', '0/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 0
        else:
            fps = float(fps_str)
        
        # Audio information
        has_audio = audio_stream is not None
        audio_sample_rate = int(audio_stream.get('sample_rate', 0)) if has_audio else None
        audio_codec = audio_stream.get('codec_name') if has_audio else None
        
        # Other metadata
        video_codec = video_stream.get('codec_name', 'unknown')
        bitrate = int(format_info.get('bit_rate', 0))
        
        # Calculate aspect ratio
        if width > 0 and height > 0:
            from math import gcd
            ratio_gcd = gcd(width, height)
            aspect_w = width // ratio_gcd
            aspect_h = height // ratio_gcd
            aspect_ratio = f"{aspect_w}:{aspect_h}"
        else:
            aspect_ratio = "unknown"
        
        return MediaInfo(
            filepath=filepath,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            has_audio=has_audio,
            audio_sample_rate=audio_sample_rate,
            video_codec=video_codec,
            audio_codec=audio_codec,
            bitrate=bitrate,
            aspect_ratio=aspect_ratio
        )
    
    @staticmethod
    def get_video_frames(filepath: str, timestamps: list[float], output_dir: str = "temp_frames") -> list[str]:
        """
        Extract frames at specific timestamps.
        
        Args:
            filepath: Path to video file
            timestamps: List of timestamps (in seconds) to extract frames
            output_dir: Directory to save extracted frames
            
        Returns:
            List of paths to extracted frame images
        """
        Path(output_dir).mkdir(exist_ok=True)
        frame_paths = []
        
        for i, timestamp in enumerate(timestamps):
            output_path = f"{output_dir}/frame_{i:04d}_{timestamp:.2f}s.jpg"
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output files
                '-i', filepath,
                '-ss', str(timestamp),
                '-vframes', '1',
                '-q:v', '2',  # High quality
                output_path
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                frame_paths.append(output_path)
            except subprocess.CalledProcessError:
                # Skip frames that can't be extracted
                continue
        
        return frame_paths