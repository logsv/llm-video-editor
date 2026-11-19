"""Contact sheets: sample frames with ``fps`` and tile them with ffmpeg's ``tile`` filter.

Vision tokens scale with pixel area (~ w*h/750 for Claude), so savings come from
small tiles and few images, not from tiling alone. Defaults keep each sheet at
or under 1568 px on its long edge so it is not downscaled again on upload.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .runner import FFmpegError, ffmpeg

MAX_EDGE = 1568


def estimate_tokens(width: int, height: int) -> int:
    scale = min(1.0, MAX_EDGE / max(width, height))
    return math.ceil((width * scale) * (height * scale) / 750)


@dataclass(frozen=True)
class SheetSpec:
    fps: float = 0.5
    cols: int = 4
    rows: int = 4
    tile_box: int = 384  # tiles fit inside a tile_box x tile_box square
    padding: int = 4
    margin: int = 4
    timestamps: bool = True

    @property
    def per_sheet(self) -> int:
        return self.cols * self.rows

    def tile_size(self, src_w: int, src_h: int) -> tuple[int, int]:
        if src_w >= src_h:
            w, h = self.tile_box, self.tile_box * src_h / max(src_w, 1)
        else:
            w, h = self.tile_box * src_w / max(src_h, 1), self.tile_box
        return max(2, int(round(w / 2)) * 2), max(2, int(round(h / 2)) * 2)

    def sheet_size(self, src_w: int, src_h: int) -> tuple[int, int]:
        tw, th = self.tile_size(src_w, src_h)
        width = self.cols * tw + (self.cols - 1) * self.padding + 2 * self.margin
        height = self.rows * th + (self.rows - 1) * self.padding + 2 * self.margin
        return width, height


@dataclass(frozen=True)
class SheetResult:
    path: Path
    tile_times: list[float]
    tile_size: tuple[int, int]
    sheet_size: tuple[int, int]


def frame_times(start: float, end: float, fps: float) -> list[float]:
    n = max(1, math.ceil((end - start) * fps - 1e-6))
    return [round(start + k / fps, 3) for k in range(n)]


def _filter(spec: SheetSpec, tw: int, th: int, with_text: bool) -> str:
    parts = [f"fps={spec.fps}", f"scale={tw}:{th}:flags=area", "setsar=1"]
    if with_text:
        size = max(12, th // 9)
        parts.append(
            "drawtext=text='%{pts\\:hms}':x=4:y=4:fontsize="
            f"{size}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=3"
        )
    parts.append(
        f"tile={spec.cols}x{spec.rows}:padding={spec.padding}:margin={spec.margin}:color=black"
    )
    return ",".join(parts)


def make_sheets(
    source: str | Path,
    out_dir: Path,
    stem: str,
    *,
    src_size: tuple[int, int],
    start: float,
    end: float,
    spec: SheetSpec = SheetSpec(),
    timeout: float = 600,
) -> list[SheetResult]:
    """Write ``<stem>_%03d.png`` sheets covering ``[start, end)`` of ``source``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{stem}_*.png"):
        old.unlink()
    tw, th = spec.tile_size(*src_size)
    pattern = out_dir / f"{stem}_%03d.png"
    base = ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-copyts", "-i", str(source), "-an"]
    tail = ["-fps_mode", "passthrough", "-start_number", "0", str(pattern)]
    try:
        ffmpeg(*base, "-vf", _filter(spec, tw, th, spec.timestamps), *tail, timeout=timeout)
    except FFmpegError:
        if not spec.timestamps:
            raise
        # drawtext needs a usable font; retry without burned-in timestamps.
        ffmpeg(*base, "-vf", _filter(spec, tw, th, False), *tail, timeout=timeout)

    produced = sorted(out_dir.glob(f"{stem}_*.png"))
    times = frame_times(start, end, spec.fps)
    results = []
    for i, path in enumerate(produced):
        chunk = times[i * spec.per_sheet : (i + 1) * spec.per_sheet]
        results.append(SheetResult(path, chunk, (tw, th), spec.sheet_size(*src_size)))
    return results
