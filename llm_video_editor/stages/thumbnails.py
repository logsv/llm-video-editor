"""Thumbnails: a sharp frame from a chosen moment plus bold, honest text.

Three designs - ``bold`` (huge words over a zoomed shot), ``question`` (red banner
and an arrow) and ``reveal`` (dark panel and a highlight ring). Rendered with the
HyperFrames still template when the toolchain is installed (real typography,
auto-fitting text), otherwise with ffmpeg drawtext. Output: JPEG under 2 MB at
1280x720 (16:9 edits) or 1080x1920 (vertical edits).
"""

from __future__ import annotations

import html
import shutil
from pathlib import Path

from ..ffmpeg.runner import FFmpegError, ffmpeg

STILL = Path(__file__).resolve().parents[1] / "hyperframes" / "stills" / "thumbnail.html"
MAX_BYTES = 2_000_000
STYLES = ("bold", "question", "reveal")


def size_for(width: int, height: int) -> tuple[int, int]:
    return (1080, 1920) if height > width else (1280, 720)


def pick_frame(source: str, start: float, end: float, out: Path, samples: int = 5) -> Path:
    """The most detailed of ``samples`` frames in [start, end] (JPEG size ~ sharpness)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    span = max(0.2, end - start)
    best, best_size = None, -1
    for i in range(samples):
        t = start + span * (i + 0.5) / samples
        cand = out.with_name(f"{out.stem}_c{i}.jpg")
        try:
            ffmpeg("-ss", f"{t:.3f}", "-i", source, "-frames:v", "1", "-q:v", "2", str(cand),
                   timeout=60)
        except FFmpegError:
            continue
        if cand.exists() and cand.stat().st_size > best_size:
            best, best_size = cand, cand.stat().st_size
    if best is None:
        raise FFmpegError(f"could not extract a frame from {source} at {start:.2f}s")
    shutil.move(best, out)
    for c in out.parent.glob(f"{out.stem}_c*.jpg"):
        c.unlink()
    return out


def _markup(text: str) -> str:
    """Escape, and highlight the last word in the accent colour."""
    words = html.escape(text.strip()).split()
    if len(words) > 1:
        words[-1] = f'<span class="hl">{words[-1]}</span>'
    return " ".join(words)


def _to_jpeg(src: Path, out: Path, w: int, h: int) -> Path:
    for q in (2, 4, 6, 9):
        ffmpeg("-i", str(src), "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,"
               f"crop={w}:{h}", "-frames:v", "1", "-q:v", str(q), str(out), timeout=60)
        if out.stat().st_size <= MAX_BYTES:
            break
    return out


def render_hyperframes(frame: Path, text: str, style: str, out: Path, w: int, h: int,
                       accent: str) -> Path:
    from ..hyperframes import toolchain

    tc = toolchain.ensure()
    work = out.parent / f".{out.stem}_hf"
    shutil.rmtree(work, ignore_errors=True)
    (work / "assets").mkdir(parents=True)
    shutil.copyfile(frame, work / "frame.jpg")
    shutil.copyfile(tc.gsap, work / "assets" / "gsap.min.js")
    page = (STILL.read_text(encoding="utf-8").replace("__W__", str(w)).replace("__H__", str(h))
            .replace("__STYLE__", style).replace("__ACCENT__", html.escape(accent))
            .replace("__TEXT__", _markup(text)))
    (work / "index.html").write_text(page, encoding="utf-8")
    toolchain.hyperframes("snapshot", "--at", "0.5", "--no-end", "--describe", "false",
                          "-o", "snaps", cwd=work, timeout=300)
    pngs = sorted((work / "snaps").glob("frame-*.png"))
    if not pngs:
        raise FFmpegError(f"HyperFrames wrote no snapshot in {work}")
    _to_jpeg(pngs[0], out, w, h)
    shutil.rmtree(work, ignore_errors=True)
    return out


def render_ffmpeg(frame: Path, text: str, style: str, out: Path, w: int, h: int,
                  accent: str) -> Path:
    """Fallback without Node: same three layouts, drawtext typography."""
    words = text.upper().split()
    per = 2 if style == "reveal" else 3  # reveal text must stay inside the left panel
    lines = [" ".join(words[i:i + per]) for i in range(0, len(words), per)] or [""]
    size = round(h * (0.16 if len(lines) <= 2 else 0.12))
    color = accent.lstrip("#")
    base = (f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
            "eq=contrast=1.12:saturation=1.3")
    if style == "bold":
        base = (f"scale={round(w * 1.12)}:{round(h * 1.12)}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},eq=contrast=1.12:saturation=1.3")
    vf = [base]
    if style == "question":
        vf.append(f"drawbox=x=0:y=0:w=iw:h={round(size * len(lines) * 1.25 + h * 0.06)}:"
                  "color=0xE8222B@0.95:t=fill")
    if style == "reveal":
        vf.append("drawbox=x=0:y=0:w=iw*0.5:h=ih:color=black@0.65:t=fill")
    for i, line in enumerate(lines):
        if style == "question":
            y = round(h * 0.03 + i * size * 1.15)
        elif style == "reveal":
            y = round(h * 0.5 - len(lines) * size * 0.6 + i * size * 1.15)
        else:
            y = round(h * 0.92 - (len(lines) - i) * size * 1.1)
        accent_line = (style == "reveal" or i == len(lines) - 1) and style != "question"
        fc = f"0x{color}" if accent_line else "white"
        # text goes through a file: no filtergraph escaping of quotes, colons or backslashes
        (out.parent / f".{out.stem}_l{i}.txt").write_text(line, encoding="utf-8")
        vf.append(f"drawtext=textfile=.{out.stem}_l{i}.txt:expansion=none:x=w*0.04:y={y}:fontsize={size}:"
                  f"fontcolor={fc}:borderw={max(3, size // 14)}:bordercolor=black")
    tmp = out.with_suffix(".png")
    ffmpeg("-i", str(frame.resolve()), "-vf", ",".join(vf), "-frames:v", "1", tmp.name,
           timeout=60, cwd=out.parent)
    _to_jpeg(tmp, out, w, h)
    tmp.unlink(missing_ok=True)
    for f in out.parent.glob(f".{out.stem}_l*.txt"):
        f.unlink()
    return out


def render_thumbnail(frame: Path, text: str, style: str, out: Path, *, width: int, height: int,
                     accent: str = "#FFD400", prefer_hyperframes: bool = True) -> str:
    """Render one thumbnail; returns the renderer used ("hyperframes" or "ffmpeg")."""
    w, h = size_for(width, height)
    style = style if style in STYLES else "bold"
    if prefer_hyperframes:
        from ..hyperframes import HyperFramesUnavailable

        try:
            render_hyperframes(frame, text, style, out, w, h, accent)
            return "hyperframes"
        except (HyperFramesUnavailable, FFmpegError):
            pass
    render_ffmpeg(frame, text, style, out, w, h, accent)
    return "ffmpeg"
