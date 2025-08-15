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
            raise RuntimeError("Set OPEN_SORA_CMD or OPEN_SORA_HOST to enable Open-Sora generation")

    @staticmethod
    def _run(cmd: str):
        proc = subprocess.run(cmd, shell=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}")