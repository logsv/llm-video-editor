from __future__ import annotations
import uuid
from pathlib import Path
from typing import Optional

class SVDClient:
    """Stable Video Diffusion 1.1 (image→video) via Diffusers when 
    available."""
    def __init__(self, model: str = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"):
        self.model = model

    def generate(self, image_path: Path, prompt: str, seconds: int, fps: int,
                outdir: Path) -> Path:
        try:
            import torch
            from diffusers import StableVideoDiffusionPipeline
            from PIL import Image
        except Exception as e:
            raise RuntimeError("Install diffusers/torch/Pillow for SVD (see README)") from e

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableVideoDiffusionPipeline.from_pretrained(self.model,
            torch_dtype=(torch.float16 if device=="cuda" else torch.float32))
        pipe = pipe.to(device)

        img = Image.open(image_path).convert("RGB")
        num_frames = max(8, int(seconds * fps))
        result = pipe(image=img, decode_chunk_size=8, motion_bucket_id=127,
                     noise_aug_strength=0.1, num_frames=num_frames)
        frames = result.frames

        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"i2v_{uuid.uuid4().hex[:8]}.mp4"

        # Save with ffmpeg
        import tempfile, subprocess
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            for i, frame in enumerate(frames):
                frame.save(tdp / f"f_{i:05d}.png")
            cmd = f"ffmpeg -y -framerate {fps} -i {tdp}/f_%05d.png -c:v libx264 -pix_fmt yuv420p {outfile}"
            proc = subprocess.run(cmd, shell=True)
            if proc.returncode != 0:
                raise RuntimeError("ffmpeg failed for SVD output")

        return outfile