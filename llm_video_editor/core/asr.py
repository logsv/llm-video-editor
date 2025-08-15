"""
Automatic Speech Recognition (ASR) module using faster-whisper.
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import tempfile

from faster_whisper import WhisperModel
from tqdm import tqdm


@dataclass
class TranscriptSegment:
    """Container for a transcript segment with timing information."""
    start: float
    end: float
    text: str
    words: List[Dict[str, Any]]


class ASRProcessor:
    """Automatic Speech Recognition processor using faster-whisper."""
    
    def __init__(self, model_size: str = "large-v3", compute_type: str = "float32"):
        """
        Initialize ASR processor.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            compute_type: Compute precision (float32, float16, int8, int8_float16)
        """
        self.model_size = model_size
        # Use compatible compute type - try int8_float16, fallback to float32
        self.compute_type = compute_type
        self._model = None
    
    @property
    def model(self) -> WhisperModel:
        """Lazy load the Whisper model."""
        if self._model is None:
            print(f"Loading Whisper model: {self.model_size}")
            try:
                self._model = WhisperModel(self.model_size, compute_type=self.compute_type)
            except ValueError as e:
                if "compute type" in str(e).lower():
                    print(f"Compute type {self.compute_type} not supported, falling back to float32")
                    self._model = WhisperModel(self.model_size, compute_type="float32")
                else:
                    raise e
        return self._model
    
    def transcribe_file(
        self,
        filepath: str,
        language: Optional[str] = None,
        vad_filter: bool = True,
        word_timestamps: bool = True
    ) -> List[TranscriptSegment]:
        """
        Transcribe audio/video file and return segments with timestamps.
        
        Args:
            filepath: Path to audio/video file
            language: Language code (e.g., 'en', 'es'). Auto-detect if None
            vad_filter: Apply voice activity detection filtering
            word_timestamps: Include word-level timestamps
            
        Returns:
            List of transcript segments with timing information
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        print(f"Transcribing: {filepath}")
        
        try:
            segments, info = self.model.transcribe(
                filepath,
                language=language,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps
            )
        except ValueError as e:
            if "empty sequence" in str(e):
                print("No speech detected in audio, defaulting to English")
                # Fallback to English when no language can be detected
                segments, info = self.model.transcribe(
                    filepath,
                    language="en",
                    vad_filter=vad_filter,
                    word_timestamps=word_timestamps
                )
            else:
                raise
        
        transcript_segments = []
        for segment in tqdm(segments, desc="Processing segments"):
            words_list = []
            if hasattr(segment, 'words') and segment.words:
                words_list = [
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": getattr(word, 'probability', 1.0)
                    }
                    for word in segment.words
                ]
            
            transcript_segments.append(TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                words=words_list
            ))
        
        print(f"Transcription complete. Language: {info.language}, Duration: {info.duration:.2f}s")
        return transcript_segments
    
    def export_srt(self, segments: List[TranscriptSegment], output_path: str) -> str:
        """
        Export transcript segments to SRT subtitle format.
        
        Args:
            segments: List of transcript segments
            output_path: Output SRT file path
            
        Returns:
            Path to created SRT file
        """
        srt_content = []
        
        for i, segment in enumerate(segments, 1):
            start_time = self._seconds_to_srt_time(segment.start)
            end_time = self._seconds_to_srt_time(segment.end)
            
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(segment.text)
            srt_content.append("")  # Blank line
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(srt_content))
        
        return output_path
    
    def export_vtt(self, segments: List[TranscriptSegment], output_path: str) -> str:
        """
        Export transcript segments to WebVTT format.
        
        Args:
            segments: List of transcript segments
            output_path: Output VTT file path
            
        Returns:
            Path to created VTT file
        """
        vtt_content = ["WEBVTT", ""]
        
        for segment in segments:
            start_time = self._seconds_to_vtt_time(segment.start)
            end_time = self._seconds_to_vtt_time(segment.end)
            
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(segment.text)
            vtt_content.append("")  # Blank line
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(vtt_content))
        
        return output_path
    
    def get_speech_segments(self, segments: List[TranscriptSegment]) -> List[Tuple[float, float]]:
        """
        Extract speech time ranges from transcript segments.
        
        Args:
            segments: List of transcript segments
            
        Returns:
            List of (start_time, end_time) tuples for speech segments
        """
        return [(seg.start, seg.end) for seg in segments if seg.text.strip()]
    
    def get_word_timestamps(self, segments: List[TranscriptSegment]) -> List[Dict[str, Any]]:
        """
        Extract all word-level timestamps from segments.
        
        Args:
            segments: List of transcript segments
            
        Returns:
            List of word dictionaries with timing information
        """
        all_words = []
        for segment in segments:
            all_words.extend(segment.words)
        return all_words
    
    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    @staticmethod
    def _seconds_to_vtt_time(seconds: float) -> str:
        """Convert seconds to WebVTT time format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def extract_audio_for_transcription(self, video_path: str) -> str:
        """
        Extract audio from video file for transcription.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file
        """
        import subprocess
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            audio_path = tmp_file.name
        
        # Extract audio using FFmpeg
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            audio_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return audio_path
        except subprocess.CalledProcessError as e:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")
    
    def cleanup_temp_audio(self, audio_path: str) -> None:
        """Clean up temporary audio file."""
        if os.path.exists(audio_path):
            os.unlink(audio_path)