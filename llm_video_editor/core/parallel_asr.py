"""
Parallel ASR processing module for high-performance speech recognition.
Processes multiple audio streams concurrently with chunk-based processing.
"""
import os
import numpy as np
import concurrent.futures
import multiprocessing
import threading
import queue
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import tempfile
import subprocess

from .asr import ASRProcessor, TranscriptSegment

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class AudioChunk:
    """Container for audio chunk with metadata."""
    chunk_id: int
    start_time: float
    end_time: float
    duration: float
    audio_data: np.ndarray
    sample_rate: int
    file_path: Optional[str] = None  # For temporary chunk files


@dataclass
class ASRJob:
    """Container for ASR processing job."""
    job_id: str
    input_files: List[str]
    output_dir: str
    language: Optional[str] = None
    chunk_duration: float = 30.0  # Chunk size in seconds
    overlap: float = 1.0  # Overlap between chunks
    priority: int = 0
    status: str = "pending"
    progress: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    results: List[TranscriptSegment] = None
    error_message: Optional[str] = None


class ParallelASRProcessor:
    """High-performance parallel ASR processor with chunk-based processing."""
    
    def __init__(
        self,
        model_size: str = "large-v3",
        compute_type: str = "float32",
        max_workers: int = None,
        chunk_duration: float = 30.0,
        chunk_overlap: float = 1.0,
        device: str = "auto"
    ):
        """
        Initialize parallel ASR processor.
        
        Args:
            model_size: Whisper model size
            compute_type: Computation type for Whisper
            max_workers: Maximum worker processes (default: CPU count)
            chunk_duration: Duration of audio chunks in seconds
            chunk_overlap: Overlap between chunks in seconds
            device: Device for processing ('cpu', 'cuda', 'auto')
        """
        self.model_size = model_size
        self.compute_type = compute_type
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 4)
        self.chunk_duration = chunk_duration
        self.chunk_overlap = chunk_overlap
        
        # Device selection
        if device == "auto":
            self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Worker pool and job queue
        self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self.job_queue = queue.Queue()
        self.active_jobs: Dict[str, ASRJob] = {}
        self.completed_jobs: Dict[str, ASRJob] = {}
        
        # Performance statistics
        self.stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "total_audio_duration": 0.0,
            "total_processing_time": 0.0,
            "avg_rtf": 0.0  # Real-time factor
        }
        
        print(f"🎤 ParallelASR initialized: {self.max_workers} workers, {model_size} model, {device} device")
    
    def start_workers(self):
        """Start the worker process pool."""
        if self.executor is not None:
            return
        
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_init_worker,
            initargs=(self.model_size, self.compute_type, self.device)
        )
        print(f"🚀 Started {self.max_workers} ASR worker processes")
    
    def stop_workers(self):
        """Stop the worker process pool."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        print("🛑 ASR workers stopped")
    
    def process_audio_parallel(
        self,
        audio_file: str,
        language: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[TranscriptSegment]:
        """
        Process single audio file with parallel chunk processing.
        
        Args:
            audio_file: Path to audio file
            language: Language code for ASR
            progress_callback: Progress callback function
            
        Returns:
            List of transcript segments
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        print(f"🎤 Processing audio file: {audio_file}")
        start_time = time.time()
        
        # Split audio into chunks
        chunks = self._split_audio_into_chunks(audio_file)
        print(f"📂 Split into {len(chunks)} chunks of {self.chunk_duration}s each")
        
        if not chunks:
            return []
        
        # Ensure workers are started
        self.start_workers()
        
        # Process chunks in parallel
        try:
            with concurrent.futures.as_completed([
                self.executor.submit(_process_audio_chunk, chunk, language)
                for chunk in chunks
            ]) as completed_futures:
                
                chunk_results = []
                for i, future in enumerate(completed_futures):
                    try:
                        segments = future.result(timeout=300)  # 5 minute timeout per chunk
                        chunk_results.append((chunks[i].chunk_id, segments))
                        
                        if progress_callback:
                            progress = (i + 1) / len(chunks)
                            progress_callback(progress)
                        
                        print(f"✅ Chunk {i+1}/{len(chunks)} completed ({len(segments)} segments)")
                        
                    except concurrent.futures.TimeoutError:
                        print(f"⏰ Chunk {i+1}/{len(chunks)} timed out")
                        chunk_results.append((chunks[i].chunk_id, []))
                    except Exception as e:
                        print(f"❌ Chunk {i+1}/{len(chunks)} failed: {e}")
                        chunk_results.append((chunks[i].chunk_id, []))
        
        finally:
            # Cleanup temporary chunk files
            self._cleanup_chunks(chunks)
        
        # Merge and deduplicate results
        merged_segments = self._merge_chunk_results(chunk_results, chunks)
        
        processing_time = time.time() - start_time
        audio_duration = self._get_audio_duration(audio_file)
        rtf = processing_time / audio_duration if audio_duration > 0 else 0
        
        print(f"🎤 Parallel ASR completed: {len(merged_segments)} segments in {processing_time:.1f}s (RTF: {rtf:.2f})")
        
        # Update statistics
        self.stats["total_audio_duration"] += audio_duration
        self.stats["total_processing_time"] += processing_time
        self.stats["avg_rtf"] = self.stats["total_processing_time"] / self.stats["total_audio_duration"]
        
        return merged_segments
    
    def submit_batch_job(
        self,
        job_id: str,
        input_files: List[str],
        output_dir: str,
        language: Optional[str] = None,
        priority: int = 0
    ) -> str:
        """
        Submit batch ASR job for processing multiple files.
        
        Args:
            job_id: Unique job identifier
            input_files: List of audio/video files
            output_dir: Output directory for transcripts
            language: Language code for ASR
            priority: Job priority
            
        Returns:
            Job ID
        """
        asr_job = ASRJob(
            job_id=job_id,
            input_files=input_files,
            output_dir=output_dir,
            language=language,
            chunk_duration=self.chunk_duration,
            overlap=self.chunk_overlap,
            priority=priority
        )
        
        self.job_queue.put(asr_job)
        self.stats["total_jobs"] += 1
        
        print(f"📝 ASR batch job '{job_id}' submitted: {len(input_files)} files")
        
        # Start background processing
        threading.Thread(target=self._process_batch_job, args=(asr_job,), daemon=True).start()
        
        return job_id
    
    def _split_audio_into_chunks(self, audio_file: str) -> List[AudioChunk]:
        """Split audio file into overlapping chunks for parallel processing."""
        try:
            # Get audio duration
            duration = self._get_audio_duration(audio_file)
            if duration <= 0:
                return []
            
            chunks = []
            chunk_id = 0
            start_time = 0.0
            
            while start_time < duration:
                end_time = min(start_time + self.chunk_duration, duration)
                chunk_duration = end_time - start_time
                
                # Extract chunk to temporary file
                chunk_file = self._extract_audio_chunk(audio_file, start_time, chunk_duration)
                
                chunk = AudioChunk(
                    chunk_id=chunk_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration=chunk_duration,
                    audio_data=None,  # Will be loaded in worker process
                    sample_rate=0,    # Will be set in worker process
                    file_path=chunk_file
                )
                
                chunks.append(chunk)
                
                # Move to next chunk with overlap
                start_time += self.chunk_duration - self.chunk_overlap
                chunk_id += 1
            
            return chunks
            
        except Exception as e:
            print(f"❌ Error splitting audio: {e}")
            return []
    
    def _extract_audio_chunk(self, audio_file: str, start_time: float, duration: float) -> str:
        """Extract audio chunk to temporary file."""
        temp_dir = Path(tempfile.gettempdir()) / "parallel_asr_chunks"
        temp_dir.mkdir(exist_ok=True)
        
        chunk_file = temp_dir / f"chunk_{int(start_time * 1000):06d}_{int(duration * 1000):06d}.wav"
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', audio_file,
            '-t', str(duration),
            '-ar', '16000',  # 16kHz for Whisper
            '-ac', '1',      # Mono
            '-f', 'wav',
            str(chunk_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and chunk_file.exists():
                return str(chunk_file)
            else:
                print(f"⚠️ Failed to extract chunk: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print(f"⏰ Chunk extraction timed out: {start_time}s")
            return None
        except Exception as e:
            print(f"❌ Chunk extraction error: {e}")
            return None
    
    def _get_audio_duration(self, audio_file: str) -> float:
        """Get audio file duration using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                audio_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
            else:
                return 0.0
        except:
            return 0.0
    
    def _merge_chunk_results(
        self,
        chunk_results: List[Tuple[int, List[TranscriptSegment]]],
        chunks: List[AudioChunk]
    ) -> List[TranscriptSegment]:
        """Merge and deduplicate transcript segments from chunks."""
        # Sort results by chunk ID
        chunk_results.sort(key=lambda x: x[0])
        
        merged_segments = []
        
        for chunk_id, segments in chunk_results:
            if chunk_id >= len(chunks):
                continue
            
            chunk = chunks[chunk_id]
            
            for segment in segments:
                # Adjust timestamps to global timeline
                adjusted_segment = TranscriptSegment(
                    start=segment.start + chunk.start_time,
                    end=segment.end + chunk.start_time,
                    text=segment.text,
                    words=[
                        {
                            **word,
                            "start": word.get("start", 0) + chunk.start_time,
                            "end": word.get("end", 0) + chunk.start_time
                        }
                        for word in segment.words
                    ] if segment.words else []
                )
                
                merged_segments.append(adjusted_segment)
        
        # Sort by start time and remove duplicates from overlaps
        merged_segments.sort(key=lambda x: x.start)
        deduplicated = self._remove_duplicate_segments(merged_segments)
        
        return deduplicated
    
    def _remove_duplicate_segments(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Remove duplicate segments from overlapping chunks."""
        if not segments:
            return []
        
        deduplicated = [segments[0]]
        
        for segment in segments[1:]:
            last_segment = deduplicated[-1]
            
            # Check for overlap and similar text
            if (segment.start < last_segment.end and 
                self._text_similarity(segment.text, last_segment.text) > 0.8):
                # Merge overlapping segments
                merged_end = max(segment.end, last_segment.end)
                
                # Keep the longer text
                if len(segment.text) > len(last_segment.text):
                    merged_text = segment.text
                    merged_words = segment.words
                else:
                    merged_text = last_segment.text
                    merged_words = last_segment.words
                
                deduplicated[-1] = TranscriptSegment(
                    start=last_segment.start,
                    end=merged_end,
                    text=merged_text,
                    words=merged_words
                )
            else:
                deduplicated.append(segment)
        
        return deduplicated
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple token overlap."""
        if not text1 or not text2:
            return 0.0
        
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def _cleanup_chunks(self, chunks: List[AudioChunk]):
        """Clean up temporary chunk files."""
        for chunk in chunks:
            if chunk.file_path and Path(chunk.file_path).exists():
                try:
                    Path(chunk.file_path).unlink()
                except Exception as e:
                    print(f"⚠️ Failed to cleanup chunk: {e}")
    
    def _process_batch_job(self, asr_job: ASRJob):
        """Process batch ASR job in background thread."""
        asr_job.status = "processing"
        asr_job.start_time = time.time()
        self.active_jobs[asr_job.job_id] = asr_job
        
        try:
            Path(asr_job.output_dir).mkdir(parents=True, exist_ok=True)
            
            all_results = []
            total_files = len(asr_job.input_files)
            
            for i, input_file in enumerate(asr_job.input_files):
                try:
                    # Process single file
                    segments = self.process_audio_parallel(
                        input_file,
                        language=asr_job.language
                    )
                    
                    # Export results
                    file_stem = Path(input_file).stem
                    srt_file = Path(asr_job.output_dir) / f"{file_stem}.srt"
                    vtt_file = Path(asr_job.output_dir) / f"{file_stem}.vtt"
                    
                    asr_processor = ASRProcessor()
                    asr_processor.export_srt(segments, str(srt_file))
                    asr_processor.export_vtt(segments, str(vtt_file))
                    
                    all_results.extend(segments)
                    
                    print(f"✅ File {i+1}/{total_files} completed: {input_file}")
                    
                except Exception as e:
                    print(f"❌ File {i+1}/{total_files} failed: {input_file} - {e}")
                    if not asr_job.error_message:
                        asr_job.error_message = str(e)
                
                # Update progress
                asr_job.progress = (i + 1) / total_files
            
            asr_job.results = all_results
            asr_job.status = "completed" if not asr_job.error_message else "failed"
            
        except Exception as e:
            asr_job.status = "failed"
            asr_job.error_message = str(e)
            self.stats["failed_jobs"] += 1
            print(f"❌ Batch ASR job '{asr_job.job_id}' failed: {e}")
        
        finally:
            asr_job.end_time = time.time()
            
            # Move to completed jobs
            if asr_job.job_id in self.active_jobs:
                del self.active_jobs[asr_job.job_id]
            self.completed_jobs[asr_job.job_id] = asr_job
            
            if asr_job.status == "completed":
                self.stats["completed_jobs"] += 1
                processing_time = asr_job.end_time - asr_job.start_time
                print(f"✅ Batch ASR job '{asr_job.job_id}' completed in {processing_time:.1f}s")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of ASR job."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
        elif job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
        else:
            return None
        
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "start_time": job.start_time,
            "end_time": job.end_time,
            "processing_time": (job.end_time or time.time()) - (job.start_time or 0) if job.start_time else 0,
            "error_message": job.error_message,
            "file_count": len(job.input_files),
            "segment_count": len(job.results) if job.results else 0
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get ASR performance statistics."""
        return {
            **self.stats,
            "config": {
                "model_size": self.model_size,
                "max_workers": self.max_workers,
                "chunk_duration": self.chunk_duration,
                "chunk_overlap": self.chunk_overlap,
                "device": self.device
            }
        }


# Worker process functions (must be at module level for multiprocessing)
def _init_worker(model_size: str, compute_type: str, device: str):
    """Initialize ASR model in worker process."""
    global _worker_asr_processor
    _worker_asr_processor = ASRProcessor(model_size=model_size, compute_type=compute_type)
    print(f"🔧 Worker initialized with {model_size} model on {device}")


def _process_audio_chunk(chunk: AudioChunk, language: Optional[str]) -> List[TranscriptSegment]:
    """Process single audio chunk in worker process."""
    global _worker_asr_processor
    
    try:
        if chunk.file_path and Path(chunk.file_path).exists():
            segments = _worker_asr_processor.transcribe_file(
                chunk.file_path,
                language=language,
                vad_filter=True,
                word_timestamps=True
            )
            return segments
        else:
            print(f"⚠️ Chunk file not found: {chunk.file_path}")
            return []
    except Exception as e:
        print(f"❌ Worker chunk processing error: {e}")
        return []


# Initialize global worker variable
_worker_asr_processor = None