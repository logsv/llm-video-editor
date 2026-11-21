"""Stage 4 - ``take-geometry``: subject boxes per scene and safe zones per cut.

For every scene that the edit actually uses, a few keyframes (default 3) are
decoded with ``-ss`` seeks (never a full decode) and run through the face
detector. Each EDL clip then gets:

* ``crop_center`` - duration-weighted focus of its scenes, nudged so the whole
  subject box stays inside the crop window;
* ``subject_box_out`` - the subject box in output-frame coordinates;
* ``caption_position`` - the caption band (bottom/top) that the subject does
  not cover;
* ``pip_corner`` - a picture-in-picture corner clear of the subject.

Output: geometry_map.json. The renderer and review-gate apply it via
``apply_geometry`` so edl_audited.json stays the cut-audit's output.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from ..ffmpeg.runner import ffmpeg_bytes
from ..schemas import (
    EDL,
    ClipGeometry,
    ClipInfo,
    ClipPlacement,
    ClipsManifest,
    Detection,
    GeometryMap,
    SceneGeometry,
)
from ..schemas.geometry import BBox, Corner
from ..workspace import Workspace
from .detectors import Detector, get_detector

ANALYSIS_WIDTH = 640
DEFAULT_FOCUS = (0.5, 0.5)


def edl_hash(edl: EDL) -> str:
    """Fingerprint of the cut itself (geometry-owned fields excluded)."""
    data = edl.model_dump(exclude={"clips": {"__all__": {"crop_center", "caption_position"}},
                                   "overlays": {"__all__": {"corner"}}})
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def grab_frame(source: str, t: float, width: int = ANALYSIS_WIDTH) -> np.ndarray:
    raw = ffmpeg_bytes("-ss", f"{max(0.0, t):.3f}", "-i", source, "-frames:v", "1",
                       "-vf", f"scale={width}:-2,format=rgb24", "-f", "rawvideo", "-")
    if not raw or len(raw) % (width * 3):
        raise ValueError(f"could not decode a frame at {t:.2f}s from {source}")
    return np.frombuffer(raw, np.uint8).reshape(-1, width, 3)


def _union(boxes: list[BBox]) -> BBox | None:
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def summarise_scene(start: float, end: float, dets: list[Detection]) -> SceneGeometry:
    faces = [d for d in dets if d.label == "face"]
    pool = faces or [d for d in dets if d.label == "person"]
    if not pool:
        return SceneGeometry(start=start, end=end, detections=dets)
    # primary subject per sample = largest confident box; union across samples
    primary: dict[float, Detection] = {}
    for d in pool:
        cur = primary.get(d.t)
        if cur is None or _area(d.bbox) * d.score > _area(cur.bbox) * cur.score:
            primary[d.t] = d
    box = _union([d.bbox for d in primary.values()])
    assert box is not None
    if pool is faces:
        focus = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    else:  # person box: aim at the head, roughly the top fifth
        focus = ((box[0] + box[2]) / 2, box[1] + 0.2 * (box[3] - box[1]))
    return SceneGeometry(start=start, end=end, detections=dets,
                         focus=(round(focus[0], 4), round(focus[1], 4)),
                         subject_box=tuple(round(v, 4) for v in box))  # type: ignore[arg-type]


def analyze_clip(info: ClipInfo, spans: list[tuple[float, float]], detector: Detector,
                 samples: int = 3) -> ClipGeometry:
    """Detect subjects in every scene of ``info`` that overlaps a used span."""
    scenes = info.scenes or []
    if not scenes:
        from ..schemas import SceneCut

        scenes = [SceneCut(start=0.0, end=info.duration)]
    out = []
    for sc in scenes:
        used = [(max(sc.start, a), min(sc.end, b)) for a, b in spans if a < sc.end and b > sc.start]
        if not used:
            continue
        lo, hi = min(u[0] for u in used), max(u[1] for u in used)
        times = [round(lo + (hi - lo) * (k + 1) / (samples + 1), 3) for k in range(samples)]
        dets = []
        for t in times:
            for label, bbox, score in detector.detect(grab_frame(info.source_file, t)):
                dets.append(Detection(t=t, label=label, bbox=bbox, score=round(score, 3)))
        out.append(summarise_scene(sc.start, sc.end, dets))
    return ClipGeometry(clip_id=info.clip_id, source_file=info.source_file, width=info.width,
                        height=info.height, scenes=out)


# -- placement maths --------------------------------------------------------------------
def crop_window(src_w: int, src_h: int, out_w: int, out_h: int, center: tuple[float, float],
                mode: str = "cover", zoom: float = 1.0) -> tuple[float, float, float, float]:
    """(scale, x0, y0, letterbox) matching ffmpeg/graph.fit_filter."""
    if mode == "contain":
        s = min(out_w / src_w, out_h / src_h)
        return s, -(out_w - src_w * s) / 2, -(out_h - src_h * s) / 2, 1.0
    s = max(out_w / src_w, out_h / src_h) * max(1.0, zoom)
    sw, sh = src_w * s, src_h * s
    x0 = min(max(0.0, center[0] * sw - out_w / 2), sw - out_w)
    y0 = min(max(0.0, center[1] * sh - out_h / 2), sh - out_h)
    return s, x0, y0, 0.0


def keep_box_inside(center: tuple[float, float], box: BBox | None, src_w: int, src_h: int,
                    out_w: int, out_h: int, zoom: float = 1.0) -> tuple[float, float]:
    """Shift a cover-crop centre so ``box`` fits in the window when it can."""
    if box is None:
        return center
    s = max(out_w / src_w, out_h / src_h) * max(1.0, zoom)
    sw, sh = src_w * s, src_h * s
    cx, cy = center
    for axis, size, scaled in ((0, out_w, sw), (1, out_h, sh)):
        lo_px, hi_px = box[axis] * scaled, box[axis + 2] * scaled
        c_px = (cx if axis == 0 else cy) * scaled
        x0 = c_px - size / 2
        if hi_px - lo_px <= size:
            x0 = min(max(x0, hi_px - size), lo_px)
        new = (x0 + size / 2) / scaled
        cx, cy = (new, cy) if axis == 0 else (cx, new)
    return (round(min(1, max(0, cx)), 4), round(min(1, max(0, cy)), 4))


def box_to_output(box: BBox, src_w: int, src_h: int, out_w: int, out_h: int,
                  center: tuple[float, float], mode: str = "cover",
                  zoom: float = 1.0) -> BBox | None:
    s, x0, y0, _ = crop_window(src_w, src_h, out_w, out_h, center, mode, zoom)
    x1 = (box[0] * src_w * s - x0) / out_w
    y1 = (box[1] * src_h * s - y0) / out_h
    x2 = (box[2] * src_w * s - x0) / out_w
    y2 = (box[3] * src_h * s - y0) / out_h
    clipped = (max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2))
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return tuple(round(v, 4) for v in clipped)  # type: ignore[return-value]


def caption_bands(out_w: int, out_h: int) -> dict[str, tuple[float, float]]:
    """Vertical extent (0..1) of the two-line caption band for each position."""
    vertical = out_h > out_w
    margin = 0.14 if vertical else 0.07
    band = 2 * 1.25 * 0.065 * min(out_w, out_h) / out_h
    return {"bottom": (1 - margin - band, 1 - margin), "top": (margin, margin + band),
            "center": (0.5 - band / 2, 0.5 + band / 2)}


def _overlap(box: BBox | None, y0: float, y1: float, x0: float = 0.08, x1: float = 0.92) -> float:
    if box is None:
        return 0.0
    w = max(0.0, min(box[2], x1) - max(box[0], x0))
    h = max(0.0, min(box[3], y1) - max(box[1], y0))
    return round((w * h) / ((x1 - x0) * (y1 - y0)), 4)


def choose_caption(box: BBox | None, out_w: int, out_h: int) -> tuple[str, float]:
    bands = caption_bands(out_w, out_h)
    scores = {k: _overlap(box, *bands[k]) for k in ("bottom", "top", "center")}
    if scores["bottom"] == 0:
        return "bottom", 0.0
    best = min(scores, key=lambda k: (scores[k], ["bottom", "top", "center"].index(k)))
    return best, scores[best]


PIP = 0.4  # fraction of the frame width/height a PiP inset occupies
PIP_MARGIN = 0.04


def pip_box(corner: Corner) -> BBox:
    x0 = PIP_MARGIN if corner[1] == "l" else 1 - PIP_MARGIN - PIP
    y0 = PIP_MARGIN if corner[0] == "t" else 1 - PIP_MARGIN - PIP
    return (x0, y0, x0 + PIP, y0 + PIP)


def choose_pip(box: BBox | None, avoid: BBox | None = None) -> Corner:
    """Corner for a PiP inset: never over the subject, then clear of ``avoid``."""
    order: list[Corner] = ["br", "bl", "tr", "tl"]

    def cost(c: Corner) -> tuple[float, float, int]:
        x0, y0, x1, y1 = pip_box(c)
        return (_overlap(box, y0, y1, x0, x1), _overlap(avoid, y0, y1, x0, x1), order.index(c))

    return min(order, key=cost)


def place_clips(edl: EDL, manifest: ClipsManifest, geos: dict[str, ClipGeometry]
                ) -> list[ClipPlacement]:
    out = []
    pip_anchors = {o.anchor_clip_id for o in edl.overlays if o.zone == "pip"}
    for clip in edl.clips:
        info, geo = manifest.get(clip.source_clip_id), geos.get(clip.source_clip_id)
        if info is None:
            continue
        weighted, total, boxes = [0.0, 0.0], 0.0, []
        for sc in (geo.scenes if geo else []):
            ov = min(sc.end, clip.end) - max(sc.start, clip.start)
            if ov <= 0 or sc.focus is None:
                continue
            weighted[0] += sc.focus[0] * ov
            weighted[1] += sc.focus[1] * ov
            total += ov
            if sc.subject_box:
                boxes.append(sc.subject_box)
        center = (weighted[0] / total, weighted[1] / total) if total else DEFAULT_FOCUS
        box = _union(boxes)
        if edl.fit_mode == "cover":
            center = keep_box_inside(center, box, info.width, info.height, edl.width, edl.height,
                                     clip.zoom)
        box_out = (box_to_output(box, info.width, info.height, edl.width, edl.height, center,
                                 edl.fit_mode, clip.zoom) if box else None)
        pos, overlap = choose_caption(box_out, edl.width, edl.height)
        bands = caption_bands(edl.width, edl.height)
        corner = choose_pip(box_out, (0.08, *bands[pos][:1], 0.92, bands[pos][1]))
        if clip.clip_id in pip_anchors:
            pb = pip_box(corner)
            if _overlap(pb, *bands[pos]) > 0:  # inset would sit on the captions: move them
                for alt in ("bottom", "top", "center"):
                    if _overlap(box_out, *bands[alt]) == 0 and _overlap(pb, *bands[alt]) == 0:
                        pos, overlap = alt, 0.0
                        break
        out.append(ClipPlacement(clip_id=clip.clip_id, crop_center=(round(center[0], 4),
                                                                    round(center[1], 4)),
                                 subject_box_out=box_out, caption_position=pos,
                                 caption_overlap=overlap, pip_corner=corner))
    return out


def apply_geometry(edl: EDL, gmap: GeometryMap | None) -> EDL:
    """Copy crop centres, caption positions and PiP corners onto the EDL.

    A crop centre or caption position already set on a clip (the editor agent's
    choice) is kept; detection only fills what was left open.
    """
    if gmap is None or gmap.edl_hash != edl_hash(edl):
        return edl
    clips = []
    for c in edl.clips:
        p = gmap.placement(c.clip_id)
        if p is None:
            clips.append(c)
            continue
        clips.append(c.model_copy(update={
            "crop_center": c.crop_center if c.crop_center is not None else p.crop_center,
            "caption_position": c.caption_position or p.caption_position}))
    overlays = []
    for ov in edl.overlays:
        p = gmap.placement(ov.anchor_clip_id)
        overlays.append(ov.model_copy(update={"corner": p.pip_corner}) if p else ov)
    return edl.model_copy(update={"clips": clips, "overlays": overlays})


def load_geometry(ws: Workspace, edl: EDL) -> GeometryMap | None:
    if not ws.geometry_map.exists():
        return None
    gmap = Workspace.read(ws.geometry_map, GeometryMap)
    return gmap if gmap.edl_hash == edl_hash(edl) else None


def run_geometry(ws: Workspace, edl: EDL | None = None, *, detector: Detector | None = None,
                 samples: int = 3) -> GeometryMap:
    edl = edl or Workspace.read(ws.edl_audited, EDL)
    manifest = Workspace.read(ws.clips_manifest, ClipsManifest)
    spans: dict[str, list[tuple[float, float]]] = {}
    for c in edl.clips:
        spans.setdefault(c.source_clip_id, []).append((c.start, c.end))
    own = detector is None
    detector = detector or get_detector()
    try:
        geos = {}
        for cid, sp in spans.items():
            info = manifest.get(cid)
            if info is not None:
                geos[cid] = analyze_clip(info, sp, detector, samples)
    finally:
        if own:
            detector.close()
    gmap = GeometryMap(detector=detector.name, edl_hash=edl_hash(edl), clips=list(geos.values()),
                       placements=place_clips(edl, manifest, geos))
    Workspace.write(ws.geometry_map, gmap)
    ws.record("geometry", ws.stage_key("geometry", gmap.edl_hash, detector.name, samples),
              [ws.geometry_map])
    return gmap
