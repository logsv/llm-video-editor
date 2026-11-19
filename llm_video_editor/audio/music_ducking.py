"""
Music ducking module using Demucs for audio source separation and intelligent mixing.
Automatically reduces music volume when speech is detected.
"""
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import tempfile
import subprocess
import json

try:
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


@dataclass
class AudioSegment:
    """Container for audio segment with timing information."""
    start_time: float
    end_time: float
    audio_type: str  # 'speech', 'music', 'mixed', 'silence'
    volume_level: float  # RMS volume level
    ducking_factor: float = 1.0  # Multiplier for music volume (0.0-1.0)


@dataclass
class DuckingProfile:
    """Container for ducking parameters."""
    speech_threshold: float = 0.01  # Minimum speech volume to trigger ducking
    music_reduction: float = 0.3  # Factor to reduce music volume (0.0-1.0)
    fade_duration: float = 0.1  # Fade in/out duration in seconds
    lookahead: float = 0.05  # Look ahead time for smoother transitions
    min_duck_duration: float = 0.5  # Minimum ducking duration


class MusicDucker:
    """Music ducking processor using Demucs for source separation."""
    
    def __init__(self, model_name: str = "htdemucs", device: str = "auto"):
        """
        Initialize music ducking processor.
        
        Args:
            model_name: Demucs model name (htdemucs, htdemucs_ft, mdx_extra)
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sample_rate = 44100
        
    def _load_model(self):
        """Lazy load Demucs model."""
        if not DEMUCS_AVAILABLE:
            raise ImportError("Demucs not available. Install with: pip install demucs")
        
        if self.model is None:
            print(f"Loading Demucs model: {self.model_name}")
            self.model = get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()
    
    def separate_audio_sources(
        self,
        audio_path: str,
        output_dir: str = None
    ) -> Dict[str, str]:
        """
        Separate audio into stems using Demucs.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Output directory for separated stems
            
        Returns:
            Dictionary mapping stem names to file paths
        """
        self._load_model()
        
        if output_dir is None:
            output_dir = Path(audio_path).parent / "stems"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"Separating audio sources: {audio_path}")
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # Ensure stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]  # Take first 2 channels
        
        # Apply Demucs model
        waveform = waveform.to(self.device)
        with torch.no_grad():
            sources = apply_model(self.model, waveform.unsqueeze(0))
        
        sources = sources.squeeze(0).cpu()
        
        # Save separated stems
        stem_files = {}
        stem_names = ['drums', 'bass', 'other', 'vocals']  # Standard Demucs output order
        
        for i, stem_name in enumerate(stem_names):
            if i < sources.shape[0]:
                stem_path = output_dir / f"{Path(audio_path).stem}_{stem_name}.wav"
                torchaudio.save(str(stem_path), sources[i], self.sample_rate)
                stem_files[stem_name] = str(stem_path)
                print(f"  Saved {stem_name}: {stem_path}")
        
        return stem_files
    
    def analyze_speech_segments(
        self,
        vocals_path: str,
        window_size: float = 0.1  # Analysis window in seconds
    ) -> List[AudioSegment]:
        """
        Analyze vocals track to identify speech segments.
        
        Args:
            vocals_path: Path to separated vocals track
            window_size: Analysis window size in seconds
            
        Returns:
            List of audio segments with speech/silence classification
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa not available. Install with: pip install librosa soundfile")
        
        # Load vocals
        audio, sr = librosa.load(vocals_path, sr=self.sample_rate)
        
        # Calculate frame parameters
        hop_length = int(window_size * sr)
        frame_length = hop_length * 2
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Calculate spectral centroid for speech detection
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        
        # Simple speech detection based on energy and spectral characteristics
        speech_threshold = np.percentile(rms[rms > 0], 25)  # Adaptive threshold
        
        segments = []
        current_start = 0.0
        current_type = None
        
        for i, (energy, centroid) in enumerate(zip(rms, spectral_centroid)):
            time_pos = i * window_size
            
            # Classify frame
            if energy > speech_threshold and 1000 < centroid < 4000:  # Speech-like characteristics
                frame_type = 'speech'
            elif energy > speech_threshold * 0.5:
                frame_type = 'mixed'  # Possible music or background
            else:
                frame_type = 'silence'
            
            # Create segment when type changes
            if current_type != frame_type:
                if current_type is not None:
                    segments.append(AudioSegment(
                        start_time=current_start,
                        end_time=time_pos,
                        audio_type=current_type,
                        volume_level=float(np.mean(rms[int(current_start/window_size):i]))
                    ))
                current_start = time_pos
                current_type = frame_type
        
        # Add final segment
        if current_type is not None:
            segments.append(AudioSegment(
                start_time=current_start,
                end_time=len(audio) / sr,
                audio_type=current_type,
                volume_level=float(np.mean(rms[int(current_start/window_size):]))
            ))
        
        return segments
    
    def generate_ducking_automation(
        self,
        speech_segments: List[AudioSegment],
        profile: DuckingProfile = None
    ) -> List[AudioSegment]:
        """
        Generate music ducking automation based on speech segments.
        
        Args:
            speech_segments: List of analyzed speech segments
            profile: Ducking parameters
            
        Returns:
            List of segments with ducking factors applied
        """
        if profile is None:
            profile = DuckingProfile()
        
        ducked_segments = []
        
        for segment in speech_segments:
            ducking_factor = 1.0  # Default: no ducking
            
            # Apply ducking based on segment type and volume
            if segment.audio_type == 'speech' and segment.volume_level > profile.speech_threshold:
                ducking_factor = profile.music_reduction
            elif segment.audio_type == 'mixed' and segment.volume_level > profile.speech_threshold * 0.5:
                ducking_factor = (profile.music_reduction + 1.0) / 2  # Partial ducking
            
            # Apply minimum duration constraint
            duration = segment.end_time - segment.start_time
            if ducking_factor < 1.0 and duration < profile.min_duck_duration:
                # Check if we can extend to meet minimum duration
                next_segments = [s for s in speech_segments if s.start_time > segment.end_time]
                if next_segments and next_segments[0].start_time - segment.end_time < profile.min_duck_duration:
                    # Extend ducking to next segment
                    ducking_factor = min(ducking_factor, next_segments[0].ducking_factor)
            
            ducked_segments.append(AudioSegment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                audio_type=segment.audio_type,
                volume_level=segment.volume_level,
                ducking_factor=ducking_factor
            ))
        
        return self._smooth_ducking_transitions(ducked_segments, profile)
    
    def _smooth_ducking_transitions(
        self,
        segments: List[AudioSegment],
        profile: DuckingProfile
    ) -> List[AudioSegment]:
        """Smooth ducking transitions to avoid abrupt changes."""
        if len(segments) <= 1:
            return segments
        
        smoothed = [segments[0]]
        
        for i in range(1, len(segments)):
            current = segments[i]
            previous = smoothed[-1]
            
            # Check for abrupt changes
            factor_diff = abs(current.ducking_factor - previous.ducking_factor)
            
            if factor_diff > 0.3:  # Significant change
                # Insert fade transition
                fade_start = max(current.start_time - profile.fade_duration, previous.end_time)
                fade_end = current.start_time + profile.fade_duration
                
                # Adjust previous segment end time
                if fade_start > previous.start_time:
                    smoothed[-1] = AudioSegment(
                        start_time=previous.start_time,
                        end_time=fade_start,
                        audio_type=previous.audio_type,
                        volume_level=previous.volume_level,
                        ducking_factor=previous.ducking_factor
                    )
                
                # Add fade transition segment
                if fade_start < fade_end and fade_end <= current.end_time:
                    fade_factor = (previous.ducking_factor + current.ducking_factor) / 2
                    smoothed.append(AudioSegment(
                        start_time=fade_start,
                        end_time=fade_end,
                        audio_type='transition',
                        volume_level=(previous.volume_level + current.volume_level) / 2,
                        ducking_factor=fade_factor
                    ))
                    
                    # Adjust current segment start time
                    current = AudioSegment(
                        start_time=fade_end,
                        end_time=current.end_time,
                        audio_type=current.audio_type,
                        volume_level=current.volume_level,
                        ducking_factor=current.ducking_factor
                    )
            
            smoothed.append(current)
        
        return smoothed
    
    def apply_ducking_to_music(
        self,
        music_path: str,
        ducking_segments: List[AudioSegment],
        output_path: str
    ) -> bool:
        """
        Apply ducking automation to music track.
        
        Args:
            music_path: Path to music/background audio
            ducking_segments: List of segments with ducking factors
            output_path: Output path for ducked audio
            
        Returns:
            True if successful
        """
        if not LIBROSA_AVAILABLE:
            return self._apply_ducking_with_ffmpeg(music_path, ducking_segments, output_path)
        
        try:
            # Load music
            audio, sr = librosa.load(music_path, sr=self.sample_rate)
            
            # Apply ducking
            ducked_audio = audio.copy()
            
            for segment in ducking_segments:
                start_sample = int(segment.start_time * sr)
                end_sample = int(segment.end_time * sr)
                
                if start_sample < len(ducked_audio) and end_sample > 0:
                    end_sample = min(end_sample, len(ducked_audio))
                    start_sample = max(start_sample, 0)
                    
                    # Apply ducking factor
                    ducked_audio[start_sample:end_sample] *= segment.ducking_factor
            
            # Save ducked audio
            sf.write(output_path, ducked_audio, sr)
            return True
            
        except Exception as e:
            print(f"Error applying ducking: {e}")
            return False
    
    def _apply_ducking_with_ffmpeg(
        self,
        music_path: str,
        ducking_segments: List[AudioSegment],
        output_path: str
    ) -> bool:
        """Fallback ducking implementation using FFmpeg volume filters."""
        try:
            # Create volume filter string
            volume_filters = []
            
            for segment in ducking_segments:
                if segment.ducking_factor != 1.0:
                    volume_db = 20 * np.log10(segment.ducking_factor)
                    volume_filters.append(
                        f"volume={volume_db}dB:enable='between(t,{segment.start_time},{segment.end_time})'"
                    )
            
            if not volume_filters:
                # No ducking needed, just copy
                import shutil
                shutil.copy2(music_path, output_path)
                return True
            
            # Build FFmpeg command
            filter_string = ','.join(volume_filters)
            cmd = [
                'ffmpeg', '-y',
                '-i', music_path,
                '-af', filter_string,
                '-c:a', 'aac',
                '-b:a', '128k',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0 and Path(output_path).exists()
            
        except Exception as e:
            print(f"FFmpeg ducking error: {e}")
            return False
    
    def create_ducked_mix(
        self,
        video_path: str,
        output_path: str,
        background_music_path: str = None,
        ducking_profile: DuckingProfile = None
    ) -> Dict[str, str]:
        """
        Create a complete ducked audio mix for video.
        
        Args:
            video_path: Path to input video
            output_path: Output video path
            background_music_path: Optional background music to add
            ducking_profile: Ducking parameters
            
        Returns:
            Dictionary with paths to generated files
        """
        if ducking_profile is None:
            ducking_profile = DuckingProfile()
        
        work_dir = Path(output_path).parent / "ducking_work"
        work_dir.mkdir(exist_ok=True)
        
        results = {
            "output_video": None,
            "ducked_music": None,
            "separated_stems": None
        }
        
        try:
            # Extract audio from video
            video_audio_path = work_dir / "video_audio.wav"
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate),
                str(video_audio_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Separate audio sources
            stem_files = self.separate_audio_sources(str(video_audio_path), str(work_dir))
            results["separated_stems"] = stem_files
            
            # Analyze speech in vocals
            if 'vocals' in stem_files:
                speech_segments = self.analyze_speech_segments(stem_files['vocals'])
                ducking_segments = self.generate_ducking_automation(speech_segments, ducking_profile)
                
                # Apply ducking to existing music (other + bass)
                music_tracks = []
                if 'other' in stem_files:
                    music_tracks.append(stem_files['other'])
                if 'bass' in stem_files:
                    music_tracks.append(stem_files['bass'])
                
                # Mix music tracks
                if music_tracks:
                    mixed_music_path = work_dir / "mixed_music.wav"
                    if len(music_tracks) == 1:
                        import shutil
                        shutil.copy2(music_tracks[0], mixed_music_path)
                    else:
                        # Mix multiple tracks
                        self._mix_audio_tracks(music_tracks, str(mixed_music_path))
                    
                    # Apply ducking
                    ducked_music_path = work_dir / "ducked_music.wav"
                    self.apply_ducking_to_music(
                        str(mixed_music_path),
                        ducking_segments,
                        str(ducked_music_path)
                    )
                    results["ducked_music"] = str(ducked_music_path)
                    
                    # Remix audio for video
                    final_audio_path = work_dir / "final_audio.wav"
                    mix_tracks = [stem_files['vocals'], stem_files['drums'], str(ducked_music_path)]
                    
                    # Add background music if provided
                    if background_music_path:
                        # Apply ducking to background music too
                        bg_ducked_path = work_dir / "bg_ducked.wav"
                        self.apply_ducking_to_music(
                            background_music_path,
                            ducking_segments,
                            str(bg_ducked_path)
                        )
                        mix_tracks.append(str(bg_ducked_path))
                    
                    self._mix_audio_tracks(mix_tracks, str(final_audio_path))
                    
                    # Replace audio in video
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', video_path,
                        '-i', str(final_audio_path),
                        '-c:v', 'copy',
                        '-c:a', 'aac',
                        '-b:a', '128k',
                        '-map', '0:v:0',
                        '-map', '1:a:0',
                        output_path
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True)
                    if result.returncode == 0:
                        results["output_video"] = output_path
            
            return results
            
        except Exception as e:
            print(f"Error creating ducked mix: {e}")
            return results
        finally:
            # Cleanup temporary files
            import shutil
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
    
    def _mix_audio_tracks(self, track_paths: List[str], output_path: str) -> bool:
        """Mix multiple audio tracks together."""
        if len(track_paths) == 1:
            import shutil
            shutil.copy2(track_paths[0], output_path)
            return True
        
        # Use FFmpeg to mix tracks
        inputs = []
        for track in track_paths:
            inputs.extend(['-i', track])
        
        # Create filter for mixing
        filter_inputs = ''.join(f'[{i}:a]' for i in range(len(track_paths)))
        mix_filter = f'{filter_inputs}amix=inputs={len(track_paths)}:duration=first:dropout_transition=3'
        
        cmd = ['ffmpeg', '-y'] + inputs + ['-filter_complex', mix_filter, output_path]
        
        try:
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
        except Exception:
            return False


def create_music_ducker(model_name: str = "htdemucs", device: str = "auto") -> MusicDucker:
    """
    Create a music ducking processor.
    
    Args:
        model_name: Demucs model name
        device: Device to use for processing
        
    Returns:
        MusicDucker instance
    """
    return MusicDucker(model_name=model_name, device=device)