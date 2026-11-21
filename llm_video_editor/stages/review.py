"""Stage 6 - ``review-gate``: a composited preview sheet before the full encode.

One frame per cut (plus one per B-roll overlay) is rendered at the cut's
midpoint with the exact crop, burned-in caption, overlay placement and the
detected subject box drawn on top, then tiled into ``review/sheet_*.png``.
This takes seconds; the full render waits for ``approve``.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..ffmpeg.graph import PIP_MARGIN, PIP_SIZE, fit_filter
from ..ffmpeg.runner import ffmpeg
from ..ffmpeg.sheets import SheetSpec
from ..schemas import EDL, ClipsManifest, CutAudit, Review, ReviewTile, Transcript
from ..schemas.geometry import BBox
from ..workspace import Workspace
from .captions import group_cues, output_words, write_captions
from .geometry import apply_geometry, edl_hash, load_geometry

TILE_BOX = 360


class ReviewRequired(RuntimeError):
    """The render gate: the current cut has not been approved."""


def geometry_hash(edl: EDL) -> str:
    data = [(c.clip_id, c.crop_center, c.caption_position) for c in edl.clips]
    data += [(o.overlay_id, o.corner, o.zone) for o in edl.overlays]
    return hashlib.sha1(json.dumps(data).encode()).hexdigest()[:16]


def _drawbox(box: BBox | None, w: int, h: int, color: str) -> list[str]:
    if box is None:
        return []
    x, y = int(box[0] * w), int(box[1] * h)
    bw, bh = max(2, int((box[2] - box[0]) * w)), max(2, int((box[3] - box[1]) * h))
    return [f"drawbox=x={x}:y={y}:w={bw}:h={bh}:color={color}:t={max(3, w // 160)}"]


def _label(text: str, h: int) -> str:
    safe = text.replace(":", "\\:").replace("'", "")
    return (f"drawtext=text='{safe}':x=12:y=h-th-12:fontsize={max(18, h // 28)}:"
            "fontcolor=yellow:box=1:boxcolor=black@0.7:boxborderw=6")


def _tile(out: Path, edl: EDL, src: str, src_t: float, t_out: float, center, box: BBox | None,
          label: str, work: Path, has_ass: bool, overlay: tuple[str, float, str, str] | None,
          tile: tuple[int, int], zoom: float = 1.0) -> None:
    W, H = edl.width, edl.height
    base = (f"[0:v]{fit_filter(W, H, edl.fit_mode, center, zoom)},setsar=1,"
            f"setpts=PTS-STARTPTS+{t_out:.3f}/TB")
    chain = []
    args = ["-ss", f"{src_t:.3f}", "-i", src]
    if overlay:
        ov_src, ov_t, zone, corner = overlay
        args += ["-ss", f"{ov_t:.3f}", "-i", ov_src]
        if zone == "pip":
            pw, ph = int(W * PIP_SIZE) // 2 * 2, int(H * PIP_SIZE) // 2 * 2
            ofit = f"scale={pw}:{ph}:force_original_aspect_ratio=decrease"
            mx, my = round(W * PIP_MARGIN), round(H * PIP_MARGIN)
            x = str(mx) if corner[1] == "l" else f"main_w-overlay_w-{mx}"
            y = str(my) if corner[0] == "t" else f"main_h-overlay_h-{my}"
        else:
            ofit, x, y = fit_filter(W, H, edl.fit_mode), "0", "0"
        chain.append(f"[1:v]{ofit},setsar=1,setpts=PTS-STARTPTS+{t_out:.3f}/TB[ov]")
        chain.append(f"{base}[b]")
        base = f"[b][ov]overlay=x={x}:y={y}"
    post = (["ass=captions.ass"] if has_ass else []) + _drawbox(box, W, H, "red")
    post += [_label(label, H), f"scale={tile[0]}:{tile[1]}:flags=area"]
    chain.append(",".join([base, *post]) + "[out]")
    ffmpeg(*args, "-filter_complex", ";".join(chain), "-map", "[out]", "-frames:v", "1",
           str(out.resolve()), timeout=120, cwd=work)


def build_review(ws: Workspace, edl: EDL | None = None, *, cols: int = 6, rows: int = 3,
                 tile_box: int = TILE_BOX) -> Review:
    from .render import overlay_specs, timeline  # local import avoids a cycle

    audited = edl or Workspace.read(ws.edl_audited, EDL)
    gmap = load_geometry(ws, audited)
    edl = apply_geometry(audited, gmap)
    manifest = Workspace.read(ws.clips_manifest, ClipsManifest)
    work = ws.review_dir
    work.mkdir(parents=True, exist_ok=True)
    for old in work.glob("*.png"):
        old.unlink()

    segs, starts = timeline(edl, manifest)
    durations = [s.duration for s in segs]
    transcript = Workspace.read(ws.transcript, Transcript) if ws.transcript.exists() else None
    has_ass = bool(transcript and edl.captions.enabled and write_captions(
        work / "captions.ass", edl, transcript, starts, durations))
    cues = group_cues(output_words(edl, transcript, starts, durations)) if transcript else []
    tile = SheetSpec(tile_box=tile_box).tile_size(edl.width, edl.height)

    def caption_at(t: float) -> str:
        return next((c.text for c in cues if c.start <= t < c.end), "")

    tiles: list[ReviewTile] = []
    warnings: list[str] = []
    placements = {p.clip_id: p for p in (gmap.placements if gmap else [])}
    if gmap is None:
        warnings.append("no geometry_map for this cut: crops are centred, captions at default")
    jobs = []
    for clip, seg, t0 in zip(edl.clips, segs, starts, strict=True):
        mid = seg.duration / 2
        p = placements.get(clip.clip_id)
        box = p.subject_box_out if p else None
        if p and p.caption_overlap > 0:
            warnings.append(f"{clip.clip_id}: caption band still overlaps the subject "
                            f"({p.caption_overlap:.0%})")
        jobs.append((clip, seg.source_file, seg.start + mid, t0 + mid, box, None))
    for spec, ov in zip(overlay_specs(edl, starts, manifest), edl.overlays, strict=False):
        anchor_i = next(i for i, c in enumerate(edl.clips) if c.clip_id == ov.anchor_clip_id)
        t_out = spec.at + spec.duration / 2
        src_t = segs[anchor_i].start + (t_out - starts[anchor_i])
        p = placements.get(ov.anchor_clip_id)
        jobs.append((edl.clips[anchor_i], segs[anchor_i].source_file, src_t, t_out,
                     p.subject_box_out if p else None, (spec, ov)))

    for n, (clip, src, src_t, t_out, box, ovp) in enumerate(jobs):
        overlay = None
        label = f"{clip.clip_id} {t_out:.1f}s"
        if ovp:
            spec, ov = ovp
            overlay = (spec.source_file, spec.src_start + spec.duration / 2, ov.zone, ov.corner)
            label = f"{ov.overlay_id} over {clip.clip_id} {t_out:.1f}s"
        _tile(work / f"tile_{n:03d}.png", edl, src, src_t, t_out, clip.crop_center, box, label,
              work, has_ass, overlay, tile, clip.zoom)
        tiles.append(ReviewTile(index=n, clip_id=clip.clip_id,
                                overlay_id=ovp[1].overlay_id if ovp else None,
                                t_out=round(t_out, 3), source_clip_id=clip.source_clip_id,
                                src_t=round(src_t, 3), caption=caption_at(t_out),
                                caption_position=clip.caption_position or "bottom",
                                subject_box_out=box))

    per = cols * rows
    ffmpeg("-framerate", "1", "-start_number", "0", "-i", str(work / "tile_%03d.png"),
           "-vf", f"tile={cols}x{rows}:padding=6:margin=6:color=0x202020",
           "-start_number", "0", str(work / "sheet_%03d.png"), timeout=120)
    sheets = sorted(work.glob("sheet_*.png"))
    if len(sheets) != -(-len(tiles) // per):
        warnings.append(f"expected {-(-len(tiles) // per)} review sheets, got {len(sheets)}")
    if ws.cut_audit.exists() and not Workspace.read(ws.cut_audit, CutAudit).passed:
        warnings.append("cut_audit.json did not pass")

    review = Review(edl_hash=edl_hash(audited), geometry_hash=geometry_hash(edl),
                    sheets=[ws.rel(s) for s in sheets], tiles=tiles, warnings=warnings)
    Workspace.write(work / "review.json", review)
    return review


def decide(ws: Workspace, approve: bool, notes: str = "") -> Review:
    path = ws.review_dir / "review.json"
    if not path.exists():
        raise FileNotFoundError("No review yet: run `llmve review` first")
    review = Workspace.read(path, Review)
    review = review.model_copy(update={
        "status": "approved" if approve else "rejected", "notes": notes,
        "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    Workspace.write(path, review)
    return review


def check_approved(ws: Workspace, audited: EDL) -> None:
    """Raise ReviewRequired unless the *current* cut and geometry were approved."""
    path = ws.review_dir / "review.json"
    if not path.exists():
        raise ReviewRequired("No review sheet: run `llmve review` and approve it first")
    review = Workspace.read(path, Review)
    current = apply_geometry(audited, load_geometry(ws, audited))
    if review.edl_hash != edl_hash(audited) or review.geometry_hash != geometry_hash(current):
        raise ReviewRequired("The cut changed since it was reviewed: run `llmve review` again")
    if review.status != "approved":
        raise ReviewRequired(f"Review status is {review.status!r}: approve it to render")
