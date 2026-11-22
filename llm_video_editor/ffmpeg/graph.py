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
    zoom: float = 1.0  # punch-in on top of the cover crop


@dataclass(frozen=True)
class OverlaySpec:
    source_file: str
    src_start: float
    at: float  # output seconds
    duration: float
    crop_center: tuple[float, float] | None = None
    zone: Literal["full", "pip"] = "full"
    corner: Literal["tl", "tr", "bl", "br"] = "br"


@dataclass(frozen=True)
class GraphicSpec:
    """A transparent VP9 WebM (HyperFrames output) placed on the output timeline."""

    path: str
    at: float  # output seconds
    duration: float
    box: tuple[float, float, float, float] | None = None  # x, y, w, h (0-1); None = full frame


@dataclass(frozen=True)
class MusicSpec:
    """An audio file placed on the output timeline (music, sfx or voice-over)."""

    path: str
    at: float  # output seconds
    duration: float
    src_start: float = 0.0
    gain_db: float = -18.0
    duck: bool = True
    loop: bool = False
    fade_in: float = 1.0
    fade_out: float = 2.0
    role: str = "music"


@dataclass(frozen=True)
class Ducking:
    """Sidechain settings. ``key_gain_db`` brings measured speech to KEY_LEVEL_DB."""

    depth_db: float = 12.0
    attack_ms: float = 20.0
    release_ms: float = 450.0
    key_gain_db: float = 0.0
    clip_depth_db: float = 0.0  # the clips' own audio dips this far under voice-overs
    vo_key_gain_db: float = 0.0  # brings the voice-over bus to KEY_LEVEL_DB


KEY_LEVEL_DB = -20.0  # speech level the sidechain key is normalised to
DUCK_RATIO = 20.0


def duck_filter(d: Ducking) -> str:
    """``sidechaincompress`` whose threshold yields ``depth_db`` of gain reduction for
    speech at KEY_LEVEL_DB: reduction = (level - threshold) * (1 - 1/ratio)."""
    thr_db = KEY_LEVEL_DB - d.depth_db / (1 - 1 / DUCK_RATIO)
    thr = min(1.0, max(0.000976563, 10 ** (thr_db / 20)))
    return (f"sidechaincompress=threshold={thr:.6f}:ratio={DUCK_RATIO:g}:attack={d.attack_ms:g}:"
            f"release={d.release_ms:g}:makeup=1:detection=rms")


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
               center: tuple[float, float] | None = None, zoom: float = 1.0) -> str:
    """Fit a source into the output frame; ``zoom`` > 1 punches in around ``center``."""
    if mode == "contain":
        return (f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black")
    cx, cy = center or (0.5, 0.5)
    sw, sh = (width, height) if zoom <= 1.0 else (
        2 * round(width * zoom / 2), 2 * round(height * zoom / 2))
    x = f"'max(0,min(iw-{width},{cx:.4f}*iw-{width}/2))'"
    y = f"'max(0,min(ih-{height},{cy:.4f}*ih-{height}/2))'"
    return (f"scale={sw}:{sh}:force_original_aspect_ratio=increase:flags=lanczos,"
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
                                          fit_filter(width, height, mode, seg.crop_center,
                                                     seg.zoom), v))
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


def add_graphics(g: Graph, base: str, graphics: list[GraphicSpec], *, width: int, height: int,
                 fps: int) -> str:
    """Alpha-composite motion graphics over ``base``; returns the new video label.

    The WebM must be decoded with ``libvpx-vp9``: ffmpeg's native VP9 decoder drops
    the alpha plane and the transparent area turns black.
    """
    for k, gr in enumerate(graphics):
        idx = g.add_input("-c:v", "libvpx-vp9", "-i", gr.path)
        if gr.box is None:
            x, y, w, h = 0, 0, width, height
        else:
            x, y = round(gr.box[0] * width), round(gr.box[1] * height)
            w = max(2, round(gr.box[2] * width) // 2 * 2)
            h = max(2, round(gr.box[3] * height) // 2 * 2)
        g.filters.append(
            f"[{idx}:v:0]setpts=PTS-STARTPTS,fps={fps},"
            f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,format=yuva420p,"
            f"tpad=stop_mode=add:stop_duration={gr.duration:.6f}:color=black@0,"
            f"trim=duration={gr.duration:.6f},setpts=PTS-STARTPTS+{gr.at:.6f}/TB[gfx{k}]")
        end = gr.at + gr.duration
        g.filters.append(f"[{base}][gfx{k}]overlay=x={x}:y={y}:format=auto:eof_action=pass:"
                         f"enable='between(t,{gr.at:.6f},{end - 1e-4:.6f})'[vg{k}]")
        base = f"vg{k}"
    return base


def _track_chain(g: Graph, t: MusicSpec, total: float) -> str:
    loop = ("-stream_loop", "-1") if t.loop else ()
    idx = g.add_input(*loop, "-ss", f"{t.src_start:.6f}", "-i", t.path)
    fi, fo = min(t.fade_in, t.duration / 2), min(t.fade_out, t.duration / 2)
    chain = (f"[{idx}:a:0]aresample=48000,{AUDIO_FMT},atrim=duration={t.duration:.6f},"
             f"asetpts=PTS-STARTPTS")
    if fi > 0:
        chain += f",afade=t=in:d={fi:.3f}"
    if fo > 0:
        chain += f",afade=t=out:st={t.duration - fo:.6f}:d={fo:.3f}"
    return chain + (f",volume={t.gain_db:.2f}dB,adelay=delays={round(t.at * 1000)}:all=1,"
                    f"apad=whole_dur={total:.6f}")


def add_speech(g: Graph, voice: str, tracks: list[MusicSpec], *, total: float,
               ducking: Ducking | None = None) -> str:
    """The speech bus: the clips' own audio plus every voice-over track. With
    ``ducking.clip_depth_db`` the clips' own sound dips under the voice-overs."""
    vos = [t for t in tracks if t.role == "voiceover"]
    if not vos:
        return voice
    for i, t in enumerate(vos):
        g.filters.append(_track_chain(g, t, total) + f"[vo{i}]")
    if len(vos) > 1:
        g.filters.append("".join(f"[vo{i}]" for i in range(len(vos)))
                         + f"amix=inputs={len(vos)}:duration=first:normalize=0[vobus]")
    else:
        g.filters.append("[vo0]anull[vobus]")
    if ducking is not None and ducking.clip_depth_db > 0:
        g.filters.append("[vobus]asplit=2[vomix][vokeyraw]")
        g.filters.append(f"[vokeyraw]volume={ducking.vo_key_gain_db:.2f}dB[vokey]")
        clip_duck = Ducking(ducking.clip_depth_db, ducking.attack_ms, ducking.release_ms)
        g.filters.append(f"[{voice}][vokey]{duck_filter(clip_duck)}[scene]")
        g.filters.append("[scene][vomix]amix=inputs=2:duration=first:normalize=0[speech]")
    else:
        g.filters.append(f"[{voice}][vobus]amix=inputs=2:duration=first:normalize=0[speech]")
    return "speech"


def add_music(g: Graph, voice: str, tracks: list[MusicSpec], *, total: float,
              ducking: Ducking = Ducking()) -> str:
    """Speech bus + music/SFX; ducked tracks dip under the speech bus.

    The sidechain key is the speech bus raised by ``ducking.key_gain_db`` (from a
    loudness measurement of the speech), so the dip depends on whether someone is
    talking, not on how loud the recording happens to be.
    """
    speech = add_speech(g, voice, tracks, total=total, ducking=ducking)
    beds = [t for t in tracks if t.role != "voiceover"]
    if not beds:
        return speech
    ducked = [i for i, t in enumerate(beds) if t.duck]
    if ducked:
        g.filters.append(f"[{speech}]asplit=2[mixv][keyraw]")
        g.filters.append(f"[keyraw]volume={ducking.key_gain_db:.2f}dB,"
                         f"asplit={len(ducked)}" + "".join(f"[sc{i}]" for i in ducked))
    else:
        g.filters.append(f"[{speech}]anull[mixv]")
    labels = ["mixv"]
    for i, t in enumerate(beds):
        chain = _track_chain(g, t, total)
        if t.duck:
            g.filters.append(chain + f"[mraw{i}]")
            g.filters.append(f"[mraw{i}][sc{i}]{duck_filter(ducking)}[m{i}]")
        else:
            g.filters.append(chain + f"[m{i}]")
        labels.append(f"m{i}")
    g.filters.append("".join(f"[{x}]" for x in labels)
                     + f"amix=inputs={len(labels)}:duration=first:normalize=0[amixed]")
    return "amixed"
