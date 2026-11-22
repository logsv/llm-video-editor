"""Colour: per-clip correction (to make clips match) and one look for the whole video.

The agent decides; this module measures and builds filters:

* ``measure_clip`` samples a few frames of a clip and reports exposure (mid grey),
  contrast (black/white points), saturation, colour cast and clipped highlights or
  shadows.
* ``suggest_matches`` proposes per-clip corrections that bring every clip towards the
  edit's median (consistency), clamped so nothing is pushed hard; dark clips get a
  noise warning instead of a big lift.
* ``correction_filter`` (per clip: exposure, contrast, saturation, warmth, tint, HDR
  tone mapping) and ``look_filter`` (one look for the whole video: a preset with a
  strength, or a .cube LUT) build ffmpeg filters used by render, review tiles and the
  text-safety check alike.
* ``color_warnings`` reports what looks overdone: strong corrections, clips that
  disagree with each other, a strong look, lifting dark footage into noise.

No grain, sharpening or other texture filters exist here on purpose: they add noise
on phone and action-camera footage.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np

from ..ffmpeg.runner import ffmpeg_bytes
from ..schemas import EDL, ClipColor, Grade

# Looks at strength 1.0; the edit uses `strength` (default 0.5) of each. Values are
# deliberately gentle: a premium look is consistent and restrained, not loud.
LOOKS: dict[str, dict] = {
    "none": {},
    "natural": {"contrast": 1.05, "saturation": 1.04, "vibrance": 0.12},
    "warm": {"contrast": 1.04, "vibrance": 0.12, "kelvin": 5300},
    "golden-hour": {"contrast": 1.06, "saturation": 1.06, "high": (0.08, 0.0, -0.06),
                    "kelvin": 5600},
    "teal-orange": {"contrast": 1.08, "vibrance": 0.10, "shadow": (-0.07, 0.0, 0.08),
                    "high": (0.08, 0.02, -0.07)},
    "cinematic": {"contrast": 1.04, "saturation": 0.92, "curve": "0/0.05 0.5/0.48 1/0.95",
                  "shadow": (-0.03, 0.0, 0.04)},
    "vivid": {"contrast": 1.08, "vibrance": 0.30},
    "matte": {"saturation": 0.90, "curve": "0/0.08 0.5/0.5 1/0.93"},
    "bw": {"saturation": 0.0, "contrast": 1.10},
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _curve(spec: str, t: float) -> str:
    """Blend a curve with the identity by ``t``."""
    pts = []
    for p in spec.split():
        x, y = (float(v) for v in p.split("/"))
        pts.append(f"{x:g}/{_lerp(x, y, t):.4f}")
    return " ".join(pts)


def look_filter(grade: Grade) -> str:
    """The whole-video look as an ffmpeg filter chain ("" for none)."""
    if grade.lut:
        return f"lut3d=file='{Path(grade.lut).expanduser().resolve()}':interp=tetrahedral"
    p, t = LOOKS.get(grade.look, {}), grade.strength
    if not p or t <= 0:
        return ""
    parts = []
    if "curve" in p:
        parts.append(f"curves=all='{_curve(p['curve'], t)}'")
    if "kelvin" in p:  # white balance shift (6500 K = neutral; lower = warmer)
        parts.append(f"colortemperature=temperature={_lerp(6500, p['kelvin'], t):.0f}:pl=1")
    bal = {}  # split toning: ffmpeg's colorbalance acts on shadows and highlights only
    for zone, key in (("shadow", "s"), ("high", "h")):
        if zone in p:
            for ch, v in zip("rgb", p[zone], strict=True):
                if v:
                    bal[f"{ch}{key}"] = v * t
    if bal:
        parts.append("colorbalance=" + ":".join(f"{k}={v:.4f}" for k, v in bal.items()))
    c, s = _lerp(1, p.get("contrast", 1), t), _lerp(1, p.get("saturation", 1), t)
    if abs(c - 1) > 1e-3 or abs(s - 1) > 1e-3:
        parts.append(f"eq=contrast={c:.4f}:saturation={s:.4f}")
    if p.get("vibrance"):
        parts.append(f"vibrance=intensity={p['vibrance'] * t:.4f}")
    return ",".join(parts)


TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def correction_filter(c: ClipColor | None) -> tuple[str, str]:
    """(before the fit, after the fit): HDR tone mapping, then the correction."""
    if c is None:
        return "", ""
    pre = TONEMAP if c.tonemap else ""
    parts = []
    if abs(c.exposure) > 1e-3:
        parts.append(f"exposure=exposure={c.exposure:.3f}")
    if c.warmth:  # +1 = 4500 K-ish warm, -1 = 8500 K-ish cool
        parts.append(f"colortemperature=temperature={6500 - 2000 * c.warmth:.0f}:pl=1")
    if c.tint:  # + green, - magenta
        parts.append(f"colorchannelmixer=gg={1 + 0.08 * c.tint:.4f}")
    if abs(c.contrast - 1) > 1e-3 or abs(c.saturation - 1) > 1e-3:
        parts.append(f"eq=contrast={c.contrast:.4f}:saturation={c.saturation:.4f}")
    return pre, ",".join(parts)


def chain(*parts: str) -> str:
    return ",".join(p for p in parts if p)


# -- measuring -----------------------------------------------------------------------------
def _frame(src: str, t: float, width: int = 320) -> np.ndarray:
    raw = ffmpeg_bytes("-ss", f"{max(0.0, t):.3f}", "-i", src, "-frames:v", "1", "-vf",
                       f"scale={width}:-2,format=rgb24", "-f", "rawvideo", "-")
    if not raw or len(raw) % (width * 3):
        raise ValueError(f"could not decode a frame at {t:.2f}s from {src}")
    return np.frombuffer(raw, np.uint8).reshape(-1, width, 3)


def frame_stats(rgb: np.ndarray) -> dict:
    x = rgb.astype(np.float32) / 255.0
    luma = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    mx, mn = x.max(axis=2), x.min(axis=2)
    sat = np.where(mx > 1e-4, (mx - mn) / np.maximum(mx, 1e-4), 0)
    r, g, b = (float(x[..., i].mean()) for i in range(3))
    return {"mid": float(np.median(luma)), "black": float(np.percentile(luma, 2)),
            "white": float(np.percentile(luma, 98)), "saturation": float(sat.mean()),
            "warm": r - b, "green": g - (r + b) / 2,
            "clipped_high": float((luma > 0.98).mean()), "clipped_low": float((luma < 0.02).mean())}


def measure_clip(src: str, start: float, end: float, samples: int = 3) -> dict:
    """Average frame statistics over ``samples`` moments of a clip."""
    span = max(0.0, end - start)
    times = [start + span * (k + 1) / (samples + 1) for k in range(samples)]
    stats = [frame_stats(_frame(src, t)) for t in times]
    return {k: round(statistics.fmean(s[k] for s in stats), 4) for k in stats[0]}


def suggest_matches(measured: dict[str, dict]) -> dict[str, dict]:
    """Per-clip corrections towards the median clip (consistency), gently clamped."""
    if not measured:
        return {}
    med = {k: statistics.median(m[k] for m in measured.values())
           for k in ("mid", "saturation", "warm", "green")}
    out = {}
    for cid, m in measured.items():
        notes = []
        ev = float(np.log2(max(med["mid"], 0.02) / max(m["mid"], 0.02)))
        if ev > 0.7 and m["mid"] < 0.15:
            notes.append("dark clip: lifting it far brings up noise; keep exposure <= +0.7")
            ev = 0.7
        sat = med["saturation"] / max(m["saturation"], 0.02)
        sug = {"exposure": round(max(-1.0, min(1.0, ev)), 2),
               "saturation": round(max(0.8, min(1.25, sat)), 2),
               "warmth": round(max(-0.5, min(0.5, (med["warm"] - m["warm"]) * 4)), 2),
               "tint": round(max(-0.5, min(0.5, (med["green"] - m["green"]) * -4)), 2)}
        if m["clipped_high"] > 0.08:
            notes.append(f"{m['clipped_high']:.0%} blown highlights: don't add exposure")
            sug["exposure"] = min(sug["exposure"], 0.0)
        small = all(abs(v) < 0.08 for k, v in sug.items() if k != "saturation") \
            and abs(sug["saturation"] - 1) < 0.06
        out[cid] = {"suggested": None if small else sug, "notes": notes}
    return out


def color_warnings(edl: EDL) -> list[str]:
    """What looks overdone or inconsistent (a report; the agent decides)."""
    w = []
    cols = [(c.clip_id, c.color) for c in edl.clips if c.color is not None]
    for cid, c in cols:
        if abs(c.exposure) > 1.2:
            w.append(f"{cid}: exposure {c.exposure:+.1f} EV is a big push; check for noise "
                     "or blown highlights")
        if c.saturation > 1.35 or c.saturation < 0.6:
            w.append(f"{cid}: saturation x{c.saturation:.2f} looks unnatural")
        if abs(c.warmth) > 0.7 or abs(c.tint) > 0.7:
            w.append(f"{cid}: strong colour shift (warmth {c.warmth:+.1f}, tint {c.tint:+.1f})")
    if len(cols) >= 3:
        sats = [c.saturation for _, c in cols]
        if max(sats) - min(sats) > 0.4:
            w.append("clip saturations differ a lot: clips may not look like one video")
    g = edl.grade
    if g.lut is None and g.look != "none" and g.strength > 0.8:
        w.append(f"look {g.look!r} at strength {g.strength:.1f} is strong: 0.4-0.6 usually "
                 "reads premium")
    if g.lut and not Path(g.lut).expanduser().exists():
        w.append(f"LUT file {g.lut} not found")
    return w
