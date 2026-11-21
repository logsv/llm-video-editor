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
          tile: tuple[int, int], zoom: float = 1.0,
          graphic: tuple[str, float, tuple[float, float, float, float] | None] | None = None
          ) -> None:
    W, H = edl.width, edl.height
    # Composite with every input starting at 0 and shift the result to t_out once, just
    # before the captions: shifting each input separately rounds differently per time
    # base (WebM 1/1000 vs MP4 1/15360), and overlay then emits the first base frame
    # before the layer above is "due", so the single captured frame lacks it.
    base = f"[0:v]{fit_filter(W, H, edl.fit_mode, center, zoom)},setsar=1,setpts=PTS-STARTPTS"
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
        chain.append(f"[1:v]{ofit},setsar=1,setpts=PTS-STARTPTS[ov]")
        chain.append(f"{base}[b]")
        base = f"[b][ov]overlay=x={x}:y={y}"
    if graphic:
        g_src, g_t, g_box = graphic
        n = 2 if overlay else 1
        args += ["-c:v", "libvpx-vp9", "-ss", f"{g_t:.3f}", "-i", g_src]  # keeps the alpha
        gx, gy, gw, gh = ((0, 0, W, H) if g_box is None else
                          (round(g_box[0] * W), round(g_box[1] * H),
                           max(2, round(g_box[2] * W)), max(2, round(g_box[3] * H))))
        chain.append(f"[{n}:v]scale={gw}:{gh}:force_original_aspect_ratio=decrease,"
                     f"format=yuva420p,setpts=PTS-STARTPTS[gfx]")
        chain.append(f"{base}[bg]")
        base = f"[bg][gfx]overlay=x={gx}:y={gy}:format=auto"
    post = [f"setpts=PTS-STARTPTS+{t_out:.3f}/TB"]  # captions are timed on the output
    post += (["ass=captions.ass"] if has_ass else []) + _drawbox(box, W, H, "red")
    # one pixel format for every tile, or the sheet's image sequence drops tiles
    post += [_label(label, H), f"scale={tile[0]}:{tile[1]}:flags=area", "format=rgb24"]
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

    gfx_jobs = _graphic_jobs(ws, edl, segs, starts, placements, warnings)
    for n, (clip, src, src_t, t_out, box, ovp) in enumerate(jobs + gfx_jobs):
        overlay = graphic = None
        label = f"{clip.clip_id} {t_out:.1f}s"
        if ovp and ovp[0] == "graphic":
            _, gspec, gr = ovp
            graphic = (gspec.path, t_out - gspec.at, gspec.box)
            label = f"{gr.graphic_id} over {clip.clip_id} {t_out:.1f}s"
        elif ovp:
            spec, ov = ovp
            overlay = (spec.source_file, spec.src_start + spec.duration / 2, ov.zone, ov.corner)
            label = f"{ov.overlay_id} over {clip.clip_id} {t_out:.1f}s"
        _tile(work / f"tile_{n:03d}.png", edl, src, src_t, t_out, clip.crop_center, box, label,
              work, has_ass, overlay, tile, clip.zoom, graphic)
        tiles.append(ReviewTile(index=n, clip_id=clip.clip_id,
                                overlay_id=ovp[1].overlay_id if ovp and overlay else None,
                                graphic_id=ovp[2].graphic_id if graphic else None,
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


def _graphic_jobs(ws: Workspace, edl: EDL, segs, starts, placements, warnings: list[str]):
    """One tile per motion graphic, at a moment where it is fully on screen."""
    if not edl.graphics:
        return []
    from ..hyperframes import GraphicError, HyperFramesUnavailable
    from .graphics import graphic_specs

    jobs = []
    for gr in edl.graphics:
        one = edl.model_copy(update={"graphics": [gr]})
        try:
            specs = graphic_specs(ws, one, starts, quality="draft")
        except (GraphicError, HyperFramesUnavailable) as e:
            warnings.append(f"graphic {gr.graphic_id} could not be rendered: {e}")
            continue
        for spec in specs:
            # 60% in: past the intro animation, before the outro
            t_out = spec.at + min(spec.duration * 0.6, max(0.0, spec.duration - 0.4))
            i = max(k for k, t0 in enumerate(starts) if t0 <= t_out + 1e-6)
            clip, seg = edl.clips[i], segs[i]
            p = placements.get(clip.clip_id)
            jobs.append((clip, seg.source_file, seg.start + (t_out - starts[i]), t_out,
                         p.subject_box_out if p else None, ("graphic", spec, gr)))
    return jobs


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
