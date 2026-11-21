"""Footage inspection: facts about every source file, for the agent to decide with.

Nothing here decides anything; it reports what a careful editor would want to know
before touching a timeline: HDR / log-like transfer, bit depth, variable or high
frame rate, rotation, missing or unusual audio, action-cam recordings split across
files, unreadable files, device, recording time and GPS.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..ffmpeg.probe import _rate, _rotation
from ..ffmpeg.runner import FFmpegError, ffprobe_json
from ..stages.source_meta import read_meta
from ..workspace import is_media, resolve_inputs

HDR_TRANSFERS = {"smpte2084": "HDR10/PQ", "arib-std-b67": "HLG"}
# GoPro: G[H|X|L] + 2-digit chapter + 4-digit recording number (GX010042, GX020042, ...)
_GOPRO = re.compile(r"^G([HXL])(\d{2})(\d{4})$", re.IGNORECASE)


@dataclass
class FootageInfo:
    path: str
    name: str
    ok: bool = True
    error: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    video_codec: str | None = None
    pix_fmt: str | None = None
    bit_depth: int = 8
    transfer: str | None = None
    hdr: str | None = None  # "HDR10/PQ" / "HLG" / None
    vfr: bool = False
    rotation: int = 0
    has_audio: bool = False
    audio_channels: int | None = None
    sample_rate: int | None = None
    recorded_at: str | None = None
    device: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    recording_group: str | None = None  # same id = one recording split across files
    chapter: int | None = None
    issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def inspect(path: str | Path) -> FootageInfo:
    p = Path(path)
    info = FootageInfo(path=str(p.resolve()), name=p.name)
    try:
        data = ffprobe_json(p)
    except (FFmpegError, ValueError) as e:
        info.ok, info.error = False, f"unreadable: {str(e).splitlines()[0][:200]}"
        info.issues.append(info.error)
        return info
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        info.ok, info.error = False, "no video stream"
        info.issues.append(info.error)
        return info
    info.duration = round(float(data.get("format", {}).get("duration") or v.get("duration") or 0), 3)
    info.rotation = _rotation(v)
    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    info.width, info.height = (h, w) if info.rotation in (90, 270) else (w, h)
    avg, real = _rate(v.get("avg_frame_rate")), _rate(v.get("r_frame_rate"))
    info.fps = round(avg or real, 3)
    info.vfr = bool(avg and real and abs(avg - real) / real > 0.02)
    info.video_codec = v.get("codec_name")
    info.pix_fmt = v.get("pix_fmt")
    m = re.search(r"(\d{2})(le|be)$", info.pix_fmt or "")
    info.bit_depth = int(m.group(1)) if m else 8
    info.transfer = v.get("color_transfer")
    info.hdr = HDR_TRANSFERS.get(info.transfer or "")
    info.has_audio = a is not None
    if a is not None:
        info.audio_channels = int(a["channels"]) if a.get("channels") else None
        info.sample_rate = int(a["sample_rate"]) if a.get("sample_rate") else None
    meta = read_meta(str(p))
    info.recorded_at, info.device = meta.recorded_at, meta.device
    info.latitude, info.longitude = meta.latitude, meta.longitude
    g = _GOPRO.match(p.stem)
    if g:
        info.recording_group = f"gopro-{g.group(1).upper()}{g.group(3)}"
        info.chapter = int(g.group(2))

    if info.hdr:
        info.issues.append(f"{info.hdr} HDR: tone-map to SDR before grading or mixing with SDR clips")
    if info.bit_depth > 8 and not info.hdr:
        info.issues.append(f"{info.bit_depth}-bit (possibly a log/flat profile): check the look before matching")
    if info.vfr:
        info.issues.append("variable frame rate (phone/screen capture): rendered to constant rate")
    if info.fps >= 50:
        info.issues.append(f"{info.fps:g} fps: can be used as slow motion")
    if info.rotation:
        info.issues.append(f"rotated {info.rotation} degrees in metadata")
    if not info.has_audio:
        info.issues.append("no audio track")
    elif info.audio_channels and info.audio_channels not in (1, 2):
        info.issues.append(f"{info.audio_channels}-channel audio: mixed down to stereo")
    elif info.audio_channels == 1:
        info.issues.append("mono audio")
    if info.duration < 1:
        info.issues.append("shorter than 1 s")
    return info


def group_recordings(infos: list[FootageInfo]) -> dict[str, list[str]]:
    """Recordings split across files (e.g. GoPro chapters), in playback order."""
    groups: dict[str, list[FootageInfo]] = {}
    for i in infos:
        if i.recording_group:
            groups.setdefault(i.recording_group, []).append(i)
    out = {}
    for gid, members in groups.items():
        if len(members) > 1:
            members.sort(key=lambda x: x.chapter or 0)
            out[gid] = [m.name for m in members]
            for m in members:
                m.issues.append(f"part {m.chapter} of one recording split across "
                                f"{len(members)} files ({gid})")
    return out


def select_inputs(inputs: list[str], *, first: int | None = None,
                  order: str = "name") -> list[Path]:
    """Resolve inputs to media files, optionally only the first N (by name or by time)."""
    paths = resolve_inputs(inputs if len(inputs) > 1 else inputs[0])
    paths = [p for p in paths if is_media(p)]
    if order == "time":
        paths.sort(key=lambda p: (read_meta(str(p)).recorded_at or "", p.name))
    elif order != "name":
        raise ValueError("order is 'name' or 'time'")
    return paths[:first] if first else paths
