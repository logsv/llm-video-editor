from __future__ import annotations
import os, shlex, subprocess, uuid
from pathlib import Path

class OpenSoraClient:
    """Thin wrapper to call Open-Sora v2 via a local script or HTTP server.
    Prefers a local script path set with OPEN_SORA_CMD, else HTTP at 
    OPEN_SORA_HOST.
    Script interface must support: --prompt --fps --seconds --resolution --
    out
    """
    def __init__(self):
        self.cmd = os.environ.get("OPEN_SORA_CMD")
        self.host = os.environ.get("OPEN_SORA_HOST")  # e.g., http://localhost:8888/generate

    def generate(self, prompt: str, seconds: int, fps: int, resolution: str,
                outdir: Path) -> Path:
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"t2v_{uuid.uuid4().hex[:8]}.mp4"
        
        if self.cmd:
            cmd = (
                f"{self.cmd} --prompt {shlex.quote(prompt)} --fps {fps} "
                f"--seconds {seconds} --resolution {resolution} --out "
                f"{shlex.quote(str(outfile))}"
            )
            self._run(cmd)
            return outfile
        elif self.host:
            # Simple HTTP client; adapt to your server's API
            import requests
            r = requests.post(self.host, json={
                "prompt": prompt, "fps": fps, "seconds": seconds,
                "resolution": resolution
            }, timeout=900)
            r.raise_for_status()
            # Assume server writes to outfile path returned by API
            path = Path(r.json().get("path", str(outfile)))
            return path
        else:
            # Fallback: Create a simple test video for development
            print("⚠️  No Open-Sora configured, creating test video...")
            return self._create_test_video(prompt, seconds, fps, resolution, outfile)
    
    def _create_test_video(self, prompt: str, seconds: int, fps: int, resolution: str, outfile: Path) -> Path:
        """Create a simple test video with text overlay for development."""
        import subprocess
        
        # Parse resolution
        width, height = map(int, resolution.split('x'))
        
        # Create a simple colored video with text
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi',
            '-i', f'color=c=0x1e3a8a:size={width}x{height}:duration={seconds}:rate={fps}',
            '-vf', f'drawtext=text=\'{prompt[:50]}...\':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(outfile)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Created test video: {outfile}")
                return outfile
            else:
                print(f"❌ FFmpeg error: {result.stderr}")
                raise RuntimeError(f"Failed to create test video: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg to create test videos.")

    @staticmethod
    def _run(cmd: str):
        proc = subprocess.run(cmd, shell=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}")