from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .opensora_client import OpenSoraClient
from .svd_client import SVDClient

@dataclass
class BrollRequest:
    mode: str  # "t2v" or "i2v"
    prompt: str
    seconds: int = 4
    fps: int = 24
    resolution: str = "576x1024"
    image_path: Optional[Path] = None

class BrollGenerator:
    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.opensora = OpenSoraClient()
        self.svd = SVDClient()

    def generate(self, req: BrollRequest) -> Path:
        if req.mode == "t2v":
            return self.opensora.generate(
                prompt=req.prompt, seconds=req.seconds, fps=req.fps,
                resolution=req.resolution, outdir=self.workdir
            )
        elif req.mode == "i2v":
            if not req.image_path:
                raise ValueError("image_path required for i2v")
            return self.svd.generate(
                image_path=req.image_path, prompt=req.prompt,
                seconds=req.seconds, fps=req.fps, outdir=self.workdir
            )
        else:
            raise ValueError(f"Unknown mode: {req.mode}")