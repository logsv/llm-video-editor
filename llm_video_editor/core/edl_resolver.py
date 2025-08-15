from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

from llm_video_editor.generative.broll import BrollGenerator, BrollRequest

"""
Resolve planner EDL entries containing `gen_broll` into concrete `source` 
clips.
- Generates short inserts (3–6 s) using Open-Sora (t2v) or SVD (i2v)
- Writes files under `workdir / "assets"`
- Replaces the `gen_broll` object with a `source` path and removes 
`gen_broll`
"""

def resolve_broll(edl: Dict[str, Any], workdir: Path) -> Dict[str, Any]:
    if not edl or not isinstance(edl, dict):
        return edl
    
    clips: List[Dict[str, Any]] = edl.get("clips", [])
    if not clips:
        return edl
    
    generator = BrollGenerator(workdir=workdir / "assets")
    new_clips: List[Dict[str, Any]] = []
    
    for clip in clips:
        gb = clip.get("gen_broll")
        if gb:
            req = BrollRequest(
                mode=gb.get("mode", "t2v"),
                prompt=gb.get("prompt", "cinematic abstract b-roll"),
                seconds=int(gb.get("seconds", 4)),
                fps=int(gb.get("fps", 24)),
                resolution=gb.get("resolution", "576x1024"),
                image_path=Path(gb["image_path"]) if gb.get("image_path") else None,
            )
            out = generator.generate(req)
            placement = clip.get("insert_when", "after")
            
            # Emit an extra tiny clip before/after, or overlay bookkeeping via marker
            if placement == "before":
                new_clips.append({"source": str(out), "start": 0.0, "end": req.seconds})
                new_clips.append({k: v for k, v in clip.items() if k != "gen_broll"})
            elif placement == "after":
                new_clips.append({k: v for k, v in clip.items() if k != "gen_broll"})
                new_clips.append({"source": str(out), "start": 0.0, "end": req.seconds})
            else:  # under (overlay/music ducking handled in renderer)
                c = {k: v for k, v in clip.items() if k != "gen_broll"}
                c.setdefault("visual_ops", []).append({
                    "type": "broll_under",
                    "asset": str(out),
                    "from": c.get("start", 0),
                    "to": c.get("end", 0)
                })
                new_clips.append(c)
        else:
            new_clips.append(clip)
    
    edl["clips"] = new_clips
    return edl