"""Build ``filter_complex`` graphs that normalise mixed sources before ``concat``.

Every segment gets its own ``-ss/-t`` input (no shared-decoder buffering on
long sources) and is normalised to the output frame rate, square pixels, the
target frame size (cover-crop or letterbox, never a stretch), yuv420p, and
48 kHz stereo audio. Segment lengths are quantised to whole output frames and
audio is padded/trimmed to exactly the same length, so ``concat`` cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FitMode = Literal["cover", "contain"]
AUDIO_FMT = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"


@dataclass(frozen=True)
class Segment:
    source_file: str
    start: float  # seconds in the source, already snapped to a source frame
    duration: float  # seconds, already quantised to output frames
    has_audio: bool
    crop_center: tuple[float, float] | None = None


@dataclass(frozen=True)
class OverlaySpec:
    source_file: str
    src_start: float
    at: float  # output seconds
    duration: float
    crop_center: tuple[float, float] | None = None
    zone: Literal["full", "pip"] = "full"
    corner: Literal["tl", "tr", "bl", "br"] = "br"


PIP_SIZE = 0.4
PIP_MARGIN = 0.04


@dataclass
class Graph:
    input_args: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    n_inputs: int = 0

    def add_input(self, *args: str) -> int:
        self.input_args += list(args)
        self.n_inputs += 1
        return self.n_inputs - 1

    @property
    def filter_complex(self) -> str:
        return ";".join(self.filters)


def quantize(seconds: float, fps: float) -> float:
    """Round to a whole number of frames (at least one)."""
    return max(1, round(seconds * fps)) / fps


def snap_to_frame(seconds: float, fps: float) -> float:
    return round(seconds * fps) / fps if fps > 0 else seconds


def fit_filter(width: int, height: int, mode: FitMode = "cover",
               center: tuple[float, float] | None = None) -> str:
    if mode == "contain":
        return (f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black")
    cx, cy = center or (0.5, 0.5)
    x = f"'max(0,min(iw-{width},{cx:.4f}*iw-{width}/2))'"
    y = f"'max(0,min(ih-{height},{cy:.4f}*ih-{height}/2))'"
    return (f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{height}:{x}:{y}")


def _video_chain(idx: int, dur: float, fps: int, fit: str, label: str, shift: float = 0.0) -> str:
    pad = 2 / fps
    tail = f"setpts=PTS-STARTPTS+{shift:.6f}/TB" if shift else "setpts=PTS-STARTPTS"
    return (f"[{idx}:v:0]setpts=PTS-STARTPTS,fps={fps},{fit},setsar=1,format=yuv420p,"
            f"tpad=stop_mode=clone:stop_duration={pad:.6f},trim=duration={dur:.6f},{tail}[{label}]")


def add_segments(g: Graph, segments: list[Segment], *, width: int, height: int, fps: int,
                 mode: FitMode = "cover", audio_only: bool = False, prefix: str = "s"
                 ) -> tuple[str | None, str]:
    """Append per-segment chains plus a ``concat``; returns (video_label, audio_label)."""
    pairs = []
    for i, seg in enumerate(segments):
        idx = g.add_input("-ss", f"{seg.start:.6f}", "-t", f"{seg.duration + 0.5:.6f}",
                          "-i", seg.source_file)
        v, a = f"{prefix}v{i}", f"{prefix}a{i}"
        if not audio_only:
            g.filters.append(_video_chain(idx, seg.duration, fps,
                                          fit_filter(width, height, mode, seg.crop_center), v))
        if seg.has_audio:
            g.filters.append(
                f"[{idx}:a:0]asetpts=PTS-STARTPTS,aresample=48000,{AUDIO_FMT},"
                f"apad=whole_dur={seg.duration:.6f},atrim=duration={seg.duration:.6f}[{a}]")
        else:
            g.filters.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={seg.duration:.6f},"
                             f"{AUDIO_FMT}[{a}]")
        pairs.append((v, a))
    n = len(pairs)
    if audio_only:
        g.filters.append("".join(f"[{a}]" for _, a in pairs) + f"concat=n={n}:v=0:a=1[{prefix}acat]")
        return None, f"{prefix}acat"
    g.filters.append("".join(f"[{v}][{a}]" for v, a in pairs)
                     + f"concat=n={n}:v=1:a=1[{prefix}vcat][{prefix}acat]")
    return f"{prefix}vcat", f"{prefix}acat"


def add_overlays(g: Graph, base: str, overlays: list[OverlaySpec], *, width: int, height: int,
                 fps: int, mode: FitMode = "cover") -> str:
    """Full-frame B-roll cutaways over the base video; returns the new video label."""
    for j, ov in enumerate(overlays):
        idx = g.add_input("-ss", f"{ov.src_start:.6f}", "-t", f"{ov.duration + 0.5:.6f}",
                          "-i", ov.source_file)
        if ov.zone == "pip":
            pw, ph = int(width * PIP_SIZE) // 2 * 2, int(height * PIP_SIZE) // 2 * 2
            fit = f"scale={pw}:{ph}:force_original_aspect_ratio=decrease:flags=lanczos"
            mx, my = round(width * PIP_MARGIN), round(height * PIP_MARGIN)
            x = str(mx) if ov.corner[1] == "l" else f"main_w-overlay_w-{mx}"
            y = str(my) if ov.corner[0] == "t" else f"main_h-overlay_h-{my}"
        else:
            fit, x, y = fit_filter(width, height, mode, ov.crop_center), "0", "0"
        g.filters.append(_video_chain(idx, ov.duration, fps, fit, f"ov{j}", shift=ov.at))
        end = ov.at + ov.duration
        g.filters.append(f"[{base}][ov{j}]overlay=x={x}:y={y}:eof_action=pass:"
                         f"enable='between(t,{ov.at:.6f},{end - 1e-4:.6f})'[vb{j}]")
        base = f"vb{j}"
    return base
