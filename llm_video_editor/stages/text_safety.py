"""Is on-screen text readable, safe and in keeping with the shot under it?

For each motion graphic, at the moment the review tile shows it, this measures:

* **contrast**: WCAG ratio between the text and what is behind it - the graphic's
  own card when it has one, otherwise the video (white text on a bright sky fails);
* **busy**: how much the shot behind varies (text over foliage or traffic is hard to
  read without a card or shadow);
* **size and safe area**: the painted area's height, and whether it stays inside the
  title-safe margins (larger in 9:16, where the platform UI covers the edges);
* **overlap** with the detected face/subject and with the caption band;
* **palette**: the shot's main colours, and suggested colours - an accent that
  matches the shot, one that contrasts with it, and the text colour that reads best.

It reports; it never changes a graphic. Every built-in template takes ``accent`` and
``text_color`` (``location`` also ``style: card``), so a finding can be fixed with the
suggested colours: ``suggested.text`` -> ``text_color``, ``suggested.match`` or
``suggested.contrast`` -> ``accent``. ``"auto"`` for either asks the graphics stage to
pick it from the shot (``auto_accent`` / ``suggest``).
"""

from __future__ import annotations

import colorsys

import numpy as np

from ..ffmpeg.graph import fit_filter
from ..ffmpeg.runner import ffmpeg_bytes
from ..schemas import EDL, GraphicCheck

ANALYSIS_W = 480
MIN_CONTRAST = 3.0  # WCAG AA for large text; titles are large text
BUSY = 0.2
MIN_HEIGHT = 0.03  # painted area below 3% of the frame height is hard to read on a phone
SAFE_LANDSCAPE = {"left": 0.05, "right": 0.05, "top": 0.05, "bottom": 0.05}
# Shorts/Reels: the platform puts buttons on the right and the title/caption at the bottom
SAFE_PORTRAIT = {"left": 0.05, "right": 0.15, "top": 0.10, "bottom": 0.20}
CAPTION_BANDS = {"bottom": (0.74, 0.96), "top": (0.04, 0.26), "center": (0.40, 0.60)}


def _raw(args: list[str], w: int, channels: int) -> np.ndarray:
    raw = ffmpeg_bytes(*args, "-frames:v", "1", "-f", "rawvideo", "-")
    if not raw or len(raw) % (w * channels):
        raise ValueError("could not decode a frame")
    return np.frombuffer(raw, np.uint8).reshape(-1, w, channels)


def frame_rgb(edl: EDL, src: str, t: float, center=None, zoom: float = 1.0,
              width: int = ANALYSIS_W, color=None) -> np.ndarray:
    """The output frame (fitted, cropped and graded like the render) at analysis size."""
    from .grade import correction_filter, look_filter

    h = round(width * edl.height / edl.width / 2) * 2
    pre, post = correction_filter(color)
    vf = ",".join(x for x in (pre, fit_filter(edl.width, edl.height, edl.fit_mode, center, zoom),
                              post, look_filter(edl.grade), f"scale={width}:{h},format=rgb24")
                  if x)
    return _raw(["-ss", f"{max(0.0, t):.3f}", "-i", src, "-vf", vf], width, 3)


def graphic_rgba(edl: EDL, path: str, t: float, box=None, width: int = ANALYSIS_W) -> np.ndarray:
    """The graphic at time ``t``, placed on a transparent canvas the size of the frame."""
    h = round(width * edl.height / edl.width / 2) * 2
    x, y, bw, bh = (0.0, 0.0, 1.0, 1.0) if box is None else box
    gw, gh = max(2, round(bw * width)), max(2, round(bh * h))
    g = _raw(["-c:v", "libvpx-vp9", "-ss", f"{max(0.0, t):.3f}", "-i", path, "-vf",
              f"scale={gw}:{gh},format=rgba"], gw, 4)  # libvpx keeps the alpha
    canvas = np.zeros((h, width, 4), np.uint8)
    ox, oy = round(x * width), round(y * h)
    ch, cw = min(gh, h - oy), min(gw, width - ox)
    canvas[oy:oy + ch, ox:ox + cw] = g[:ch, :cw]
    return canvas


def luminance(rgb: np.ndarray) -> np.ndarray:
    c = rgb[..., :3].astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def contrast(l1: float, l2: float) -> float:
    hi, lo = max(l1, l2), min(l1, l2)
    return round((hi + 0.05) / (lo + 0.05), 2)


def _hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(round(v)) for v in rgb))


def palette(rgb: np.ndarray, n: int = 5) -> list[tuple[float, float, float]]:
    """Main colours of an image, most common first (coarse quantisation + averaging)."""
    px = rgb[::4, ::4, :3].reshape(-1, 3).astype(np.int32)
    keys = (px[:, 0] >> 5) * 64 + (px[:, 1] >> 5) * 8 + (px[:, 2] >> 5)
    out = []
    for k in np.bincount(keys).argsort()[::-1][:n]:
        sel = px[keys == k]
        if len(sel):
            out.append(tuple(float(v) for v in sel.mean(axis=0)))
    return out


def suggest(rgb: np.ndarray) -> dict[str, str]:
    """Accent that matches the shot, one that contrasts with it, and a text colour."""
    cols = palette(rgb, 8)
    hsv = [colorsys.rgb_to_hsv(*(v / 255 for v in c)) for c in cols]
    # matching: the shot's most colourful main colour, made vivid enough for an accent
    h, s, _ = max(hsv, key=lambda x: x[1] * (0.3 + x[2]))
    match = colorsys.hsv_to_rgb(h, max(0.55, min(0.85, s * 1.4)), 0.95)
    # contrasting: the complement of the dominant hue (sky blue -> warm orange)
    dh, ds, _ = hsv[0]
    comp_h = (dh + 0.5) % 1.0 if ds > 0.12 else 0.12  # grey shot: warm yellow-orange
    comp = colorsys.hsv_to_rgb(comp_h, 0.8, 1.0)
    mean_l = float(luminance(rgb).mean())
    text = "#FFFFFF" if contrast(1.0, mean_l) >= contrast(0.01, mean_l) else "#111111"
    return {"match": _hex(np.array(match) * 255), "contrast": _hex(np.array(comp) * 255),
            "text": text}


def auto_accent(rgb: np.ndarray) -> str:
    """The matching accent, or the contrasting one if the match is too faint here."""
    s = suggest(rgb)
    back = float(luminance(rgb).mean())

    def lum(hx: str) -> float:
        return float(luminance(np.array([[int(hx[i:i + 2], 16) for i in (1, 3, 5)]],
                                        np.uint8))[0])

    return s["match"] if contrast(lum(s["match"]), back) >= 1.8 else s["contrast"]


def _overlap(a, b) -> float:
    """Fraction of box ``a`` covered by box ``b`` (both x1, y1, x2, y2)."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    return w * h / area


def measure(edl: EDL, bg: np.ndarray, gfx: np.ndarray, *, graphic_id: str, t_out: float,
            subject_box=None, caption_position: str | None = None) -> GraphicCheck:
    """Measure one graphic (RGBA canvas) over its background frame (RGB)."""
    alpha = gfx[..., 3].astype(np.float32) / 255.0
    mask = alpha > 0.5
    sug, pal = suggest(bg), [_hex(c) for c in palette(bg)]
    if not mask.any():
        return GraphicCheck(graphic_id=graphic_id, t_out=round(t_out, 3), palette=pal,
                            suggested=sug, issues=["nothing visible at the sampled moment"])
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    x1, y1, x2, y2 = xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h
    bbox = (round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3))
    region = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    g_l = luminance(gfx)[mask]
    bg_l = luminance(bg)
    behind = bg_l[region][~mask[region]]
    behind = behind if behind.size else bg_l[region].ravel()
    busy = float(np.std(behind))
    panel = mask[region].mean() >= 0.5
    if panel:  # text on its own card: contrast within the graphic
        c = contrast(float(np.percentile(g_l, 95)), float(np.percentile(g_l, 5)))
    else:  # text straight on the shot: the text pixels against what is around them
        back = float(np.mean(behind))
        lo, hi = float(np.percentile(g_l, 10)), float(np.percentile(g_l, 90))
        c = max(contrast(lo, back), contrast(hi, back))
    issues = []
    if c < MIN_CONTRAST:
        issues.append(f"low contrast {c}:1 (needs {MIN_CONTRAST}:1): set text_color to "
                      f"{sug['text']} (or \"auto\"), use a card (location style \"card\"), "
                      "or move it to a darker/lighter part of the frame")
    if not panel and busy > BUSY:
        issues.append(f"busy background behind the text ({busy:.2f}): a card (location "
                      "style \"card\", lower-third) or a calmer part of the frame reads better")
    if y2 - y1 < MIN_HEIGHT:
        issues.append(f"very small: {(y2 - y1):.1%} of the frame height")
    safe = SAFE_PORTRAIT if edl.height > edl.width else SAFE_LANDSCAPE
    out = [k for k, lim in (("left", x1 < safe["left"] - 0.005),
                            ("right", x2 > 1 - safe["right"] + 0.005),
                            ("top", y1 < safe["top"] - 0.005),
                            ("bottom", y2 > 1 - safe["bottom"] + 0.005)) if lim]
    full_frame = x1 < 0.01 and y1 < 0.01 and x2 > 0.99 and y2 > 0.99
    if out and not full_frame:
        issues.append(f"outside the safe area ({', '.join(out)}): platform UI or TV "
                      "overscan may cover it")
    if subject_box is not None and _overlap(subject_box, bbox) > 0.1 and not full_frame:
        issues.append(f"covers {_overlap(subject_box, bbox):.0%} of the subject/face")
    if edl.captions.enabled and caption_position in CAPTION_BANDS:
        a, b = CAPTION_BANDS[caption_position]
        if _overlap(bbox, (0.15, a, 0.85, b)) > 0.15 and not full_frame:
            issues.append(f"overlaps the {caption_position} caption band")
    return GraphicCheck(graphic_id=graphic_id, t_out=round(t_out, 3), bbox=bbox,
                        contrast=c, backdrop="panel" if panel else "video",
                        busy=round(busy, 3), issues=issues, palette=pal, suggested=sug)
