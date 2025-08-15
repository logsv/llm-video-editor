"""
NVENC GPU-accelerated rendering module for high-performance video processing.
Utilizes NVIDIA hardware encoding for faster rendering and batch processing.
"""
import os
import subprocess
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import threading
import queue
import time
import json
from enum import Enum

from .renderer import VideoRenderer, RenderJob
from .planner import EditDecisionList
from ..utils.ffmpeg_utils import FFmpegProcessor


class NVENCPreset(Enum):
    """NVENC preset options for quality vs speed balance."""
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    LOSSLESS = "lossless"


class NVENCProfile(Enum):
    """NVENC profile options for encoding."""
    BASELINE = "baseline"
    MAIN = "main"
    HIGH = "high"
    HIGH444P = "high444p"


@dataclass
class NVENCSettings:
    """NVENC encoding settings configuration."""
    preset: NVENCPreset = NVENCPreset.MEDIUM
    profile: NVENCProfile = NVENCProfile.HIGH
    bitrate: Optional[str] = None  # e.g., "8M", "12M"
    crf: Optional[int] = None  # Constant Rate Factor (0-51)
    rc_mode: str = "vbr"  # Rate control: cbr, vbr, constqp, vbr_minqp
    gpu_id: int = 0  # GPU device ID for multi-GPU systems
    surfaces: int = 32  # Number of encoding surfaces
    async_depth: int = 4  # Async encoding depth
    b_frames: int = 3  # Number of B-frames
    refs: int = 3  # Reference frames


@dataclass
class BatchJob:
    """Container for batch processing job."""
    job_id: str
    input_files: List[str]
    output_dir: str
    edl_configs: List[Dict[str, Any]]
    priority: int = 0  # Higher priority processed first
    status: str = "pending"  # pending, processing, completed, failed
    progress: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None


class NVENCRenderer(VideoRenderer):
    """High-performance GPU-accelerated video renderer using NVENC."""
    
    def __init__(
        self,
        nvenc_settings: Optional[NVENCSettings] = None,
        max_concurrent_jobs: int = 2,
        enable_smart_reframing: bool = True,
        enable_music_ducking: bool = True,
        enable_qc: bool = True
    ):
        """
        Initialize NVENC renderer.
        
        Args:
            nvenc_settings: NVENC encoding configuration
            max_concurrent_jobs: Maximum concurrent rendering jobs
            enable_smart_reframing: Enable YOLO-based smart reframing
            enable_music_ducking: Enable music ducking with Demucs
            enable_qc: Enable quality control checks
        """
        # Initialize base renderer with GPU enabled
        super().__init__(
            use_gpu=True,
            enable_smart_reframing=enable_smart_reframing,
            enable_music_ducking=enable_music_ducking,
            enable_qc=enable_qc
        )
        
        self.nvenc_settings = nvenc_settings or NVENCSettings()
        self.max_concurrent_jobs = max_concurrent_jobs
        
        # Validate NVENC availability
        self._validate_nvenc_support()
        
        # Performance monitoring
        self.render_stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "total_render_time": 0.0,
            "avg_fps": 0.0
        }
    
    def _validate_nvenc_support(self) -> bool:
        """Validate NVENC hardware encoding support."""
        try:
            # Check for NVENC encoder availability
            result = subprocess.run([
                'ffmpeg', '-f', 'lavfi', '-i', 'testsrc2=duration=1:size=320x240:rate=1',
                '-c:v', 'h264_nvenc', '-f', 'null', '-'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("✅ NVENC hardware encoding available")
                return True
            else:
                print("⚠️ NVENC not available, falling back to software encoding")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠️ NVENC check timed out")
            return False
        except Exception as e:
            print(f"⚠️ NVENC validation error: {e}")
            return False
    
    def _build_nvenc_command(
        self,
        input_file: str,
        output_file: str,
        clip_operations: List[Dict[str, Any]] = None,
        duration: Optional[float] = None,
        start_time: Optional[float] = None
    ) -> List[str]:
        """Build optimized FFmpeg command with NVENC encoding."""
        cmd = ['ffmpeg', '-y']
        
        # GPU acceleration and input
        cmd.extend(['-hwaccel', 'cuda'])
        cmd.extend(['-hwaccel_output_format', 'cuda'])
        
        if start_time:
            cmd.extend(['-ss', str(start_time)])
        
        cmd.extend(['-i', input_file])
        
        if duration:
            cmd.extend(['-t', str(duration)])
        
        # Video encoding with NVENC
        cmd.extend(['-c:v', 'h264_nvenc'])
        cmd.extend(['-preset', self.nvenc_settings.preset.value])
        cmd.extend(['-profile:v', self.nvenc_settings.profile.value])
        cmd.extend(['-rc', self.nvenc_settings.rc_mode])
        
        # Rate control
        if self.nvenc_settings.bitrate:
            cmd.extend(['-b:v', self.nvenc_settings.bitrate])
        elif self.nvenc_settings.crf:
            cmd.extend(['-cq', str(self.nvenc_settings.crf)])
        
        # NVENC-specific optimizations
        cmd.extend(['-gpu', str(self.nvenc_settings.gpu_id)])
        cmd.extend(['-surfaces', str(self.nvenc_settings.surfaces)])
        cmd.extend(['-async_depth', str(self.nvenc_settings.async_depth)])
        cmd.extend(['-bf', str(self.nvenc_settings.b_frames)])
        cmd.extend(['-refs', str(self.nvenc_settings.refs)])
        
        # Video filters for operations
        filters = []
        if clip_operations:
            for operation in clip_operations:
                if operation.get('type') == 'reframe':
                    target_aspect = operation.get('params', {}).get('target_aspect')
                    if target_aspect == "9:16":
                        filters.append("hwupload_cuda,scale_cuda=1080:1920")
                    elif target_aspect == "16:9":
                        filters.append("hwupload_cuda,scale_cuda=1920:1080")
                elif operation.get('type') == 'color_correction':
                    params = operation.get('params', {})
                    if any(params.get(k) for k in ['brightness', 'contrast', 'saturation']):
                        eq_params = []
                        if params.get('brightness'):
                            eq_params.append(f"brightness={params['brightness']}")
                        if params.get('contrast'):
                            eq_params.append(f"contrast={params['contrast']}")
                        if params.get('saturation'):
                            eq_params.append(f"saturation={params['saturation']}")
                        filters.append(f"eq={':'.join(eq_params)}")
        
        if filters:
            cmd.extend(['-vf', ','.join(filters)])
        else:
            # Default GPU upload for processing
            cmd.extend(['-vf', 'hwupload_cuda'])
        
        # Audio encoding
        cmd.extend(['-c:a', 'aac', '-b:a', '128k', '-ar', '48000'])
        
        # Output format
        cmd.extend(['-f', 'mp4'])
        cmd.extend(['-movflags', '+faststart'])
        
        cmd.append(output_file)
        
        return cmd
    
    def render_clip_nvenc(
        self,
        input_file: str,
        output_file: str,
        start_time: float,
        duration: float,
        operations: List[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Render single clip with NVENC acceleration.
        
        Returns:
            Tuple of (success, stats)
        """
        start_render_time = time.time()
        
        try:
            cmd = self._build_nvenc_command(
                input_file, output_file, operations, duration, start_time
            )
            
            # Execute with performance monitoring
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration * 10 + 30  # Adaptive timeout
            )
            
            render_time = time.time() - start_render_time
            
            stats = {
                "render_time": render_time,
                "fps": duration / render_time if render_time > 0 else 0,
                "output_size": Path(output_file).stat().st_size if Path(output_file).exists() else 0,
                "success": result.returncode == 0
            }
            
            if result.returncode != 0:
                stats["error"] = result.stderr
                print(f"❌ NVENC render failed: {result.stderr}")
                return False, stats
            
            print(f"✅ NVENC render completed: {duration:.1f}s video in {render_time:.1f}s ({stats['fps']:.1f}x realtime)")
            return True, stats
            
        except subprocess.TimeoutExpired:
            stats = {
                "render_time": time.time() - start_render_time,
                "fps": 0,
                "output_size": 0,
                "success": False,
                "error": "Render timeout"
            }
            return False, stats
        except Exception as e:
            stats = {
                "render_time": time.time() - start_render_time,
                "fps": 0,
                "output_size": 0,
                "success": False,
                "error": str(e)
            }
            return False, stats
    
    def render_edl_parallel(
        self,
        edl: EditDecisionList,
        source_file: str,
        output_dir: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Render EDL with parallel clip processing using NVENC.
        
        Args:
            edl: Edit Decision List
            source_file: Source video file path
            output_dir: Output directory
            progress_callback: Progress callback function
            
        Returns:
            Rendering results with performance stats
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Starting NVENC parallel rendering: {len(edl.clips)} clips")
        start_time = time.time()
        
        # Prepare clip jobs
        clip_jobs = []
        for i, clip in enumerate(edl.clips):
            output_file = os.path.join(output_dir, f"nvenc_clip_{i:03d}.mp4")
            clip_jobs.append({
                "index": i,
                "clip": clip,
                "output_file": output_file,
                "operations": [op.__dict__ if hasattr(op, '__dict__') else op for op in clip.operations]
            })
        
        # Process clips in parallel
        completed_clips = []
        failed_clips = []
        total_render_time = 0.0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_jobs) as executor:
            # Submit all clip jobs
            future_to_job = {
                executor.submit(
                    self.render_clip_nvenc,
                    source_file,
                    job["output_file"],
                    job["clip"].start_time,
                    job["clip"].duration,
                    job["operations"]
                ): job for job in clip_jobs
            }
            
            # Process completed jobs
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    success, stats = future.result()
                    total_render_time += stats["render_time"]
                    
                    if success:
                        completed_clips.append(job["output_file"])
                        print(f"✅ Clip {job['index']+1}/{len(clip_jobs)} completed ({stats['fps']:.1f}x realtime)")
                    else:
                        failed_clips.append({"job": job, "error": stats.get("error", "Unknown error")})
                        print(f"❌ Clip {job['index']+1}/{len(clip_jobs)} failed: {stats.get('error', 'Unknown')}")
                    
                    if progress_callback:
                        progress = (len(completed_clips) + len(failed_clips)) / len(clip_jobs) * 0.8
                        progress_callback(progress)
                        
                except Exception as e:
                    failed_clips.append({"job": job, "error": str(e)})
                    print(f"❌ Clip {job['index']+1}/{len(clip_jobs)} exception: {e}")
        
        # Concatenate successful clips
        final_output = None
        if completed_clips:
            final_output = os.path.join(output_dir, f"nvenc_final_{edl.target_platform}.mp4")
            concat_success = self._concatenate_clips_nvenc(completed_clips, final_output)
            
            if not concat_success:
                print("❌ Failed to concatenate clips")
                final_output = None
        
        total_time = time.time() - start_time
        
        # Update stats
        self.render_stats["total_jobs"] += 1
        self.render_stats["total_render_time"] += total_time
        if not failed_clips:
            self.render_stats["completed_jobs"] += 1
        else:
            self.render_stats["failed_jobs"] += 1
        
        avg_fps = edl.target_duration / total_render_time if total_render_time > 0 else 0
        self.render_stats["avg_fps"] = avg_fps
        
        if progress_callback:
            progress_callback(1.0)
        
        results = {
            "success": len(failed_clips) == 0,
            "final_video": final_output,
            "completed_clips": len(completed_clips),
            "failed_clips": len(failed_clips),
            "total_time": total_time,
            "render_time": total_render_time,
            "speedup": edl.target_duration / total_render_time if total_render_time > 0 else 0,
            "avg_fps": avg_fps,
            "errors": [f["error"] for f in failed_clips]
        }
        
        print(f"🎬 NVENC rendering complete: {results['speedup']:.1f}x realtime speedup")
        return results
    
    def _concatenate_clips_nvenc(self, clip_files: List[str], output_file: str) -> bool:
        """Concatenate clips using GPU-accelerated processing."""
        if not clip_files:
            return False
        
        try:
            # Create file list for concat
            with open(f"{output_file}.txt", 'w') as f:
                for clip_file in clip_files:
                    f.write(f"file '{os.path.abspath(clip_file)}'\n")
            
            # Use GPU-accelerated concat
            cmd = [
                'ffmpeg', '-y',
                '-hwaccel', 'cuda',
                '-f', 'concat',
                '-safe', '0',
                '-i', f"{output_file}.txt",
                '-c:v', 'h264_nvenc',
                '-preset', self.nvenc_settings.preset.value,
                '-c:a', 'copy',
                output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Cleanup
            Path(f"{output_file}.txt").unlink(missing_ok=True)
            
            return result.returncode == 0 and Path(output_file).exists()
            
        except Exception as e:
            print(f"❌ NVENC concatenation error: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get rendering performance statistics."""
        return {
            **self.render_stats,
            "nvenc_settings": {
                "preset": self.nvenc_settings.preset.value,
                "profile": self.nvenc_settings.profile.value,
                "gpu_id": self.nvenc_settings.gpu_id,
                "max_concurrent_jobs": self.max_concurrent_jobs
            }
        }


class BatchProcessor:
    """Batch queue system for processing multiple video jobs."""
    
    def __init__(self, max_workers: int = 2, nvenc_settings: Optional[NVENCSettings] = None):
        """
        Initialize batch processor.
        
        Args:
            max_workers: Maximum concurrent batch jobs
            nvenc_settings: NVENC encoding settings
        """
        self.max_workers = max_workers
        self.nvenc_settings = nvenc_settings or NVENCSettings()
        
        self.job_queue = queue.PriorityQueue()
        self.active_jobs: Dict[str, BatchJob] = {}
        self.completed_jobs: Dict[str, BatchJob] = {}
        
        self.is_running = False
        self.worker_threads: List[threading.Thread] = []
        
        # Statistics
        self.stats = {
            "total_jobs_submitted": 0,
            "total_jobs_completed": 0,
            "total_jobs_failed": 0,
            "total_processing_time": 0.0
        }
    
    def submit_batch_job(
        self,
        job_id: str,
        input_files: List[str],
        output_dir: str,
        edl_configs: List[Dict[str, Any]],
        priority: int = 0
    ) -> str:
        """
        Submit a batch job to the processing queue.
        
        Args:
            job_id: Unique job identifier
            input_files: List of input video files
            output_dir: Output directory for processed videos
            edl_configs: List of EDL configurations for each input
            priority: Job priority (higher = processed first)
            
        Returns:
            Job ID
        """
        if len(input_files) != len(edl_configs):
            raise ValueError("Number of input files must match number of EDL configs")
        
        batch_job = BatchJob(
            job_id=job_id,
            input_files=input_files,
            output_dir=output_dir,
            edl_configs=edl_configs,
            priority=priority,
            status="pending"
        )
        
        # Add to queue (priority queue uses negative priority for max-heap behavior)
        self.job_queue.put((-priority, time.time(), batch_job))
        self.stats["total_jobs_submitted"] += 1
        
        print(f"📝 Batch job '{job_id}' submitted: {len(input_files)} files, priority {priority}")
        return job_id
    
    def start_processing(self):
        """Start the batch processing workers."""
        if self.is_running:
            return
        
        self.is_running = True
        
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.worker_threads.append(worker)
        
        print(f"🚀 Batch processor started with {self.max_workers} workers")
    
    def stop_processing(self):
        """Stop the batch processing workers."""
        self.is_running = False
        
        for worker in self.worker_threads:
            worker.join(timeout=30)
        
        self.worker_threads.clear()
        print("🛑 Batch processor stopped")
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop for processing batch jobs."""
        print(f"👷 Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get next job from queue (timeout to allow checking is_running)
                try:
                    _, _, batch_job = self.job_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the batch job
                self._process_batch_job(batch_job, worker_id)
                self.job_queue.task_done()
                
            except Exception as e:
                print(f"❌ Worker {worker_id} error: {e}")
        
        print(f"👷 Worker {worker_id} stopped")
    
    def _process_batch_job(self, batch_job: BatchJob, worker_id: int):
        """Process a single batch job."""
        print(f"🎬 Worker {worker_id} processing job '{batch_job.job_id}'")
        
        batch_job.status = "processing"
        batch_job.start_time = time.time()
        self.active_jobs[batch_job.job_id] = batch_job
        
        try:
            # Create NVENC renderer for this job
            renderer = NVENCRenderer(
                nvenc_settings=self.nvenc_settings,
                max_concurrent_jobs=1  # Each batch job uses one thread pool
            )
            
            processed_files = []
            total_files = len(batch_job.input_files)
            
            for i, (input_file, edl_config) in enumerate(zip(batch_job.input_files, batch_job.edl_configs)):
                try:
                    # Create EDL from config
                    edl = EditDecisionList(**edl_config)
                    
                    # Create output directory for this file
                    file_output_dir = os.path.join(
                        batch_job.output_dir,
                        f"job_{batch_job.job_id}",
                        f"file_{i:03d}"
                    )
                    Path(file_output_dir).mkdir(parents=True, exist_ok=True)
                    
                    # Render with NVENC
                    results = renderer.render_edl_parallel(
                        edl, input_file, file_output_dir
                    )
                    
                    if results["success"]:
                        processed_files.append(results["final_video"])
                        print(f"✅ File {i+1}/{total_files} completed: {input_file}")
                    else:
                        print(f"❌ File {i+1}/{total_files} failed: {input_file}")
                        batch_job.error_message = f"Failed to process {input_file}: {results.get('errors', 'Unknown error')}"
                    
                    # Update progress
                    batch_job.progress = (i + 1) / total_files
                    
                except Exception as e:
                    print(f"❌ Error processing {input_file}: {e}")
                    batch_job.error_message = str(e)
            
            # Mark job as completed
            batch_job.end_time = time.time()
            batch_job.status = "completed" if not batch_job.error_message else "failed"
            
            # Update statistics
            processing_time = batch_job.end_time - batch_job.start_time
            self.stats["total_processing_time"] += processing_time
            
            if batch_job.status == "completed":
                self.stats["total_jobs_completed"] += 1
                print(f"✅ Batch job '{batch_job.job_id}' completed in {processing_time:.1f}s")
            else:
                self.stats["total_jobs_failed"] += 1
                print(f"❌ Batch job '{batch_job.job_id}' failed: {batch_job.error_message}")
            
        except Exception as e:
            batch_job.end_time = time.time()
            batch_job.status = "failed"
            batch_job.error_message = str(e)
            self.stats["total_jobs_failed"] += 1
            print(f"❌ Batch job '{batch_job.job_id}' failed with exception: {e}")
        
        finally:
            # Move job from active to completed
            if batch_job.job_id in self.active_jobs:
                del self.active_jobs[batch_job.job_id]
            self.completed_jobs[batch_job.job_id] = batch_job
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
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
            "file_count": len(job.input_files)
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status and statistics."""
        return {
            "queue_size": self.job_queue.qsize(),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "is_running": self.is_running,
            "worker_count": len(self.worker_threads),
            "stats": self.stats
        }