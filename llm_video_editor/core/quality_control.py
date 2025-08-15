"""
Quality Control (QC) module for comprehensive video and audio validation.
Performs automated checks for black frames, duration validation, and LUFS audio reports.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import subprocess
import json
import tempfile
from datetime import datetime

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


@dataclass
class BlackFrameDetection:
    """Container for black frame detection results."""
    frame_number: int
    timestamp: float
    black_percentage: float
    threshold_used: float


@dataclass
class AudioAnalysis:
    """Container for audio analysis results."""
    integrated_lufs: float
    loudness_range_lu: float
    true_peak_dbfs: float
    sample_rate: int
    duration: float
    channels: int
    clipping_detected: bool
    silence_segments: List[Tuple[float, float]]  # (start, end) pairs


@dataclass
class DurationValidation:
    """Container for duration validation results."""
    expected_duration: float
    actual_duration: float
    difference_seconds: float
    difference_percentage: float
    within_tolerance: bool
    tolerance_used: float


@dataclass
class VideoMetrics:
    """Container for video quality metrics."""
    resolution: Tuple[int, int]
    frame_rate: float
    total_frames: int
    duration: float
    codec: str
    bitrate: Optional[int]
    aspect_ratio: str
    corrupt_frames: List[int]
    scene_changes: List[float]


@dataclass
class QCReport:
    """Comprehensive QC report."""
    file_path: str
    timestamp: str
    passed: bool
    video_metrics: Optional[VideoMetrics]
    audio_analysis: Optional[AudioAnalysis]
    black_frames: List[BlackFrameDetection]
    duration_validation: Optional[DurationValidation]
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]


class QualityController:
    """Comprehensive quality control system for video and audio validation."""
    
    def __init__(self):
        """Initialize quality controller."""
        self.black_frame_threshold = 0.98  # 98% black pixels
        self.lufs_target = -16.0  # EBU R128 standard
        self.duration_tolerance = 0.5  # 0.5 second tolerance
        
    def analyze_video_file(
        self,
        video_path: str,
        expected_duration: Optional[float] = None,
        check_black_frames: bool = True,
        check_audio: bool = True,
        sample_interval: int = 30  # Check every 30th frame for performance
    ) -> QCReport:
        """
        Perform comprehensive QC analysis on video file.
        
        Args:
            video_path: Path to video file
            expected_duration: Expected duration for validation
            check_black_frames: Whether to check for black frames
            check_audio: Whether to analyze audio
            sample_interval: Frame sampling interval for analysis
            
        Returns:
            QCReport with comprehensive analysis results
        """
        report = QCReport(
            file_path=video_path,
            timestamp=datetime.now().isoformat(),
            passed=True,
            video_metrics=None,
            audio_analysis=None,
            black_frames=[],
            duration_validation=None,
            errors=[],
            warnings=[],
            recommendations=[]
        )
        
        if not Path(video_path).exists():
            report.errors.append(f"File not found: {video_path}")
            report.passed = False
            return report
        
        try:
            # Get video metrics
            report.video_metrics = self._analyze_video_metrics(video_path)
            
            # Check black frames
            if check_black_frames:
                report.black_frames = self._detect_black_frames(
                    video_path, sample_interval
                )
                
                if report.black_frames:
                    report.warnings.append(f"Found {len(report.black_frames)} black frames")
                    if len(report.black_frames) > 10:
                        report.errors.append("Excessive black frames detected")
                        report.passed = False
            
            # Validate duration
            if expected_duration is not None:
                report.duration_validation = self._validate_duration(
                    report.video_metrics.duration, expected_duration
                )
                
                if not report.duration_validation.within_tolerance:
                    report.errors.append(
                        f"Duration mismatch: expected {expected_duration:.1f}s, "
                        f"got {report.video_metrics.duration:.1f}s"
                    )
                    report.passed = False
            
            # Analyze audio
            if check_audio:
                report.audio_analysis = self._analyze_audio(video_path)
                
                if report.audio_analysis:
                    # Check LUFS compliance
                    lufs_diff = abs(report.audio_analysis.integrated_lufs - self.lufs_target)
                    if lufs_diff > 3.0:
                        report.warnings.append(
                            f"Audio loudness deviation: {report.audio_analysis.integrated_lufs:.1f} LUFS "
                            f"(target: {self.lufs_target:.1f})"
                        )
                    
                    if lufs_diff > 6.0:
                        report.errors.append("Audio loudness significantly out of range")
                        report.passed = False
                    
                    # Check for clipping
                    if report.audio_analysis.clipping_detected:
                        report.errors.append("Audio clipping detected")
                        report.passed = False
                    
                    # Check for excessive silence
                    total_silence = sum(end - start for start, end in report.audio_analysis.silence_segments)
                    silence_percentage = (total_silence / report.audio_analysis.duration) * 100
                    
                    if silence_percentage > 20:
                        report.warnings.append(f"High silence percentage: {silence_percentage:.1f}%")
            
            # Generate recommendations
            self._generate_recommendations(report)
            
        except Exception as e:
            report.errors.append(f"Analysis failed: {str(e)}")
            report.passed = False
        
        return report
    
    def _analyze_video_metrics(self, video_path: str) -> VideoMetrics:
        """Analyze video file and extract metrics."""
        # Use ffprobe to get detailed video information
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # Find video stream
        video_stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
        format_info = data['format']
        
        # Extract metrics
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        frame_rate = eval(video_stream.get('r_frame_rate', '30/1'))  # Parse fraction
        duration = float(format_info.get('duration', 0))
        total_frames = int(video_stream.get('nb_frames', frame_rate * duration))
        codec = video_stream.get('codec_name', 'unknown')
        bitrate = int(format_info.get('bit_rate', 0)) if format_info.get('bit_rate') else None
        
        # Calculate aspect ratio
        if height > 0:
            aspect_ratio = f"{width}:{height}"
            ratio_val = width / height
            if abs(ratio_val - 16/9) < 0.1:
                aspect_ratio = "16:9"
            elif abs(ratio_val - 9/16) < 0.1:
                aspect_ratio = "9:16"
            elif abs(ratio_val - 4/3) < 0.1:
                aspect_ratio = "4:3"
        else:
            aspect_ratio = "unknown"
        
        # Check for corrupt frames (basic check)
        corrupt_frames = self._detect_corrupt_frames(video_path)
        
        # Detect scene changes
        scene_changes = self._detect_scene_changes(video_path)
        
        return VideoMetrics(
            resolution=(width, height),
            frame_rate=frame_rate,
            total_frames=total_frames,
            duration=duration,
            codec=codec,
            bitrate=bitrate,
            aspect_ratio=aspect_ratio,
            corrupt_frames=corrupt_frames,
            scene_changes=scene_changes
        )
    
    def _detect_black_frames(
        self,
        video_path: str,
        sample_interval: int = 30
    ) -> List[BlackFrameDetection]:
        """Detect black frames in video."""
        black_frames = []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return black_frames
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames for performance
            if frame_number % sample_interval == 0:
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Count black pixels (threshold at 10 for near-black)
                black_pixels = np.sum(gray < 10)
                total_pixels = gray.shape[0] * gray.shape[1]
                black_percentage = black_pixels / total_pixels
                
                if black_percentage >= self.black_frame_threshold:
                    timestamp = frame_number / fps
                    black_frames.append(BlackFrameDetection(
                        frame_number=frame_number,
                        timestamp=timestamp,
                        black_percentage=black_percentage,
                        threshold_used=self.black_frame_threshold
                    ))
            
            frame_number += 1
        
        cap.release()
        return black_frames
    
    def _detect_corrupt_frames(self, video_path: str) -> List[int]:
        """Detect potentially corrupt frames using FFmpeg."""
        corrupt_frames = []
        
        try:
            cmd = [
                'ffmpeg', '-v', 'error',
                '-i', video_path,
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse FFmpeg error output for frame corruption indicators
            if result.stderr:
                for line in result.stderr.split('\n'):
                    if 'corrupt' in line.lower() or 'error' in line.lower():
                        # Extract frame number if possible
                        if 'frame=' in line:
                            try:
                                frame_num = int(line.split('frame=')[1].split()[0])
                                corrupt_frames.append(frame_num)
                            except:
                                pass
        except Exception:
            pass
        
        return corrupt_frames
    
    def _detect_scene_changes(self, video_path: str) -> List[float]:
        """Detect scene changes using FFmpeg scene filter."""
        scene_changes = []
        
        try:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vf', 'select=gt(scene\\,0.3),showinfo',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse scene change timestamps from output
            for line in result.stderr.split('\n'):
                if 'pts_time:' in line:
                    try:
                        timestamp = float(line.split('pts_time:')[1].split()[0])
                        scene_changes.append(timestamp)
                    except:
                        pass
        except Exception:
            pass
        
        return scene_changes
    
    def _analyze_audio(self, video_path: str) -> Optional[AudioAnalysis]:
        """Analyze audio track for quality metrics."""
        try:
            # Extract audio using FFmpeg
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                audio_path = tmp_file.name
            
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '48000',
                audio_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Analyze with FFmpeg loudnorm filter
            lufs_analysis = self._analyze_lufs(audio_path)
            
            # Additional analysis with librosa if available
            if LIBROSA_AVAILABLE:
                audio_data, sr = librosa.load(audio_path, sr=None)
                
                # Detect clipping
                clipping_detected = np.any(np.abs(audio_data) >= 0.99)
                
                # Detect silence segments
                silence_segments = self._detect_silence_segments(audio_data, sr)
                
                # Get audio properties
                duration = len(audio_data) / sr
                channels = 1 if audio_data.ndim == 1 else audio_data.shape[0]
            else:
                # Fallback analysis
                clipping_detected = False
                silence_segments = []
                duration = lufs_analysis.get('duration', 0)
                channels = 1
                sr = 48000
            
            # Clean up temporary file
            Path(audio_path).unlink()
            
            return AudioAnalysis(
                integrated_lufs=lufs_analysis.get('integrated_lufs', -23.0),
                loudness_range_lu=lufs_analysis.get('loudness_range', 0.0),
                true_peak_dbfs=lufs_analysis.get('true_peak', 0.0),
                sample_rate=sr,
                duration=duration,
                channels=channels,
                clipping_detected=clipping_detected,
                silence_segments=silence_segments
            )
            
        except Exception as e:
            print(f"Audio analysis failed: {e}")
            return None
    
    def _analyze_lufs(self, audio_path: str) -> Dict[str, float]:
        """Analyze audio loudness using FFmpeg loudnorm filter."""
        try:
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-af', 'loudnorm=print_format=json',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Extract JSON from stderr
            output_lines = result.stderr.split('\n')
            json_start = False
            json_lines = []
            
            for line in output_lines:
                if '{' in line:
                    json_start = True
                if json_start:
                    json_lines.append(line)
                if '}' in line and json_start:
                    break
            
            if json_lines:
                try:
                    analysis = json.loads('\n'.join(json_lines))
                    return {
                        'integrated_lufs': float(analysis.get('input_i', -23.0)),
                        'loudness_range': float(analysis.get('input_lra', 0.0)),
                        'true_peak': float(analysis.get('input_tp', 0.0)),
                        'duration': float(analysis.get('input_duration', 0.0))
                    }
                except:
                    pass
            
        except Exception:
            pass
        
        return {
            'integrated_lufs': -23.0,
            'loudness_range': 0.0,
            'true_peak': 0.0,
            'duration': 0.0
        }
    
    def _detect_silence_segments(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        silence_threshold: float = 0.01,
        min_duration: float = 1.0
    ) -> List[Tuple[float, float]]:
        """Detect silence segments in audio."""
        if not LIBROSA_AVAILABLE:
            return []
        
        # Calculate RMS energy
        hop_length = int(0.1 * sample_rate)  # 100ms windows
        rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
        
        # Find silence frames
        silence_frames = rms < silence_threshold
        
        # Convert to time segments
        silence_segments = []
        in_silence = False
        start_time = 0.0
        
        for i, is_silent in enumerate(silence_frames):
            time_pos = i * hop_length / sample_rate
            
            if is_silent and not in_silence:
                # Start of silence
                start_time = time_pos
                in_silence = True
            elif not is_silent and in_silence:
                # End of silence
                duration = time_pos - start_time
                if duration >= min_duration:
                    silence_segments.append((start_time, time_pos))
                in_silence = False
        
        # Handle case where file ends in silence
        if in_silence:
            end_time = len(audio_data) / sample_rate
            duration = end_time - start_time
            if duration >= min_duration:
                silence_segments.append((start_time, end_time))
        
        return silence_segments
    
    def _validate_duration(
        self,
        actual_duration: float,
        expected_duration: float
    ) -> DurationValidation:
        """Validate video duration against expected value."""
        difference = abs(actual_duration - expected_duration)
        difference_percentage = (difference / expected_duration) * 100 if expected_duration > 0 else 0
        within_tolerance = difference <= self.duration_tolerance
        
        return DurationValidation(
            expected_duration=expected_duration,
            actual_duration=actual_duration,
            difference_seconds=difference,
            difference_percentage=difference_percentage,
            within_tolerance=within_tolerance,
            tolerance_used=self.duration_tolerance
        )
    
    def _generate_recommendations(self, report: QCReport) -> None:
        """Generate improvement recommendations based on QC results."""
        if report.audio_analysis:
            # Audio recommendations
            lufs = report.audio_analysis.integrated_lufs
            if lufs > -14:
                report.recommendations.append("Audio is too loud - apply compression or limiting")
            elif lufs < -20:
                report.recommendations.append("Audio is too quiet - apply gain or normalization")
            
            if report.audio_analysis.clipping_detected:
                report.recommendations.append("Remove audio clipping - check gain staging")
            
            if report.audio_analysis.silence_segments:
                total_silence = sum(end - start for start, end in report.audio_analysis.silence_segments)
                if total_silence > 5.0:
                    report.recommendations.append("Consider removing or reducing long silence segments")
        
        if report.video_metrics:
            # Video recommendations
            if report.video_metrics.bitrate and report.video_metrics.bitrate < 1000000:  # < 1Mbps
                report.recommendations.append("Video bitrate is low - consider higher quality encoding")
            
            if report.video_metrics.frame_rate < 24:
                report.recommendations.append("Frame rate is below standard - consider 24fps minimum")
        
        if report.black_frames:
            report.recommendations.append("Remove or replace black frames for better viewer experience")
    
    def export_report(
        self,
        report: QCReport,
        output_path: str,
        format: str = "json"
    ) -> str:
        """
        Export QC report to file.
        
        Args:
            report: QC report to export
            output_path: Output file path
            format: Export format ("json", "html", "txt")
            
        Returns:
            Path to exported report
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
        
        elif format == "html":
            self._export_html_report(report, output_path)
        
        elif format == "txt":
            self._export_text_report(report, output_path)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return output_path
    
    def _export_html_report(self, report: QCReport, output_path: str) -> None:
        """Export QC report as HTML."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>QC Report - {Path(report.file_path).name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {'#d4edda' if report.passed else '#f8d7da'}; padding: 10px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .error {{ color: #dc3545; }}
                .warning {{ color: #ffc107; }}
                .recommendation {{ color: #17a2b8; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Quality Control Report</h1>
                <p><strong>File:</strong> {report.file_path}</p>
                <p><strong>Timestamp:</strong> {report.timestamp}</p>
                <p><strong>Status:</strong> {'PASSED' if report.passed else 'FAILED'}</p>
            </div>
        """
        
        # Add sections based on available data
        if report.video_metrics:
            html_content += f"""
            <div class="section">
                <h2>Video Metrics</h2>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
                    <tr><td>Resolution</td><td>{report.video_metrics.resolution[0]}x{report.video_metrics.resolution[1]}</td></tr>
                    <tr><td>Frame Rate</td><td>{report.video_metrics.frame_rate:.2f} fps</td></tr>
                    <tr><td>Duration</td><td>{report.video_metrics.duration:.2f} seconds</td></tr>
                    <tr><td>Codec</td><td>{report.video_metrics.codec}</td></tr>
                    <tr><td>Aspect Ratio</td><td>{report.video_metrics.aspect_ratio}</td></tr>
                </table>
            </div>
            """
        
        if report.audio_analysis:
            html_content += f"""
            <div class="section">
                <h2>Audio Analysis</h2>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
                    <tr><td>Integrated LUFS</td><td>{report.audio_analysis.integrated_lufs:.1f} LUFS</td></tr>
                    <tr><td>Loudness Range</td><td>{report.audio_analysis.loudness_range_lu:.1f} LU</td></tr>
                    <tr><td>True Peak</td><td>{report.audio_analysis.true_peak_dbfs:.1f} dBFS</td></tr>
                    <tr><td>Sample Rate</td><td>{report.audio_analysis.sample_rate} Hz</td></tr>
                    <tr><td>Clipping Detected</td><td>{'Yes' if report.audio_analysis.clipping_detected else 'No'}</td></tr>
                </table>
            </div>
            """
        
        # Add errors, warnings, recommendations
        for section_name, items, css_class in [
            ("Errors", report.errors, "error"),
            ("Warnings", report.warnings, "warning"),
            ("Recommendations", report.recommendations, "recommendation")
        ]:
            if items:
                html_content += f"""
                <div class="section">
                    <h2>{section_name}</h2>
                    <ul class="{css_class}">
                """
                for item in items:
                    html_content += f"<li>{item}</li>"
                html_content += "</ul></div>"
        
        html_content += "</body></html>"
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _export_text_report(self, report: QCReport, output_path: str) -> None:
        """Export QC report as plain text."""
        lines = [
            "QUALITY CONTROL REPORT",
            "=" * 50,
            f"File: {report.file_path}",
            f"Timestamp: {report.timestamp}",
            f"Status: {'PASSED' if report.passed else 'FAILED'}",
            ""
        ]
        
        if report.video_metrics:
            lines.extend([
                "VIDEO METRICS:",
                f"  Resolution: {report.video_metrics.resolution[0]}x{report.video_metrics.resolution[1]}",
                f"  Frame Rate: {report.video_metrics.frame_rate:.2f} fps",
                f"  Duration: {report.video_metrics.duration:.2f} seconds",
                f"  Codec: {report.video_metrics.codec}",
                f"  Aspect Ratio: {report.video_metrics.aspect_ratio}",
                ""
            ])
        
        if report.audio_analysis:
            lines.extend([
                "AUDIO ANALYSIS:",
                f"  Integrated LUFS: {report.audio_analysis.integrated_lufs:.1f}",
                f"  Loudness Range: {report.audio_analysis.loudness_range_lu:.1f} LU",
                f"  True Peak: {report.audio_analysis.true_peak_dbfs:.1f} dBFS",
                f"  Sample Rate: {report.audio_analysis.sample_rate} Hz",
                f"  Clipping Detected: {'Yes' if report.audio_analysis.clipping_detected else 'No'}",
                ""
            ])
        
        for section_name, items in [
            ("ERRORS", report.errors),
            ("WARNINGS", report.warnings),
            ("RECOMMENDATIONS", report.recommendations)
        ]:
            if items:
                lines.append(f"{section_name}:")
                for item in items:
                    lines.append(f"  - {item}")
                lines.append("")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))


def create_quality_controller() -> QualityController:
    """Create a quality control instance."""
    return QualityController()