"""Stage 5 - ``render``: one encode from the audited EDL, then QC.

Small edits render in a single ffmpeg pass. Edits with more clips than
``max_inputs`` are rendered in groups to mezzanine files (x264 CRF 12 + PCM,
so there is no AAC priming drift) that are joined by the concat demuxer in
the final pass. B-roll overlays, captions and two-pass loudnorm always
happen in the final pass.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ..ffmpeg.encoders import (
    AUDIO_ARGS,
    MEZZANINE_AUDIO,
    MEZZANINE_VIDEO,
    Quality,
    detect_encoder,
    video_args,
)
from ..ffmpeg.graph import (
    KEY_LEVEL_DB,
    Ducking,
    Graph,
    MusicSpec,
    OverlaySpec,
    Segment,
    add_graphics,
    add_music,
    add_overlays,
    add_segments,
    add_speech,
    quantize,
    snap_to_frame,
)
from ..ffmpeg.loudnorm import (
    DEFAULT_TARGET,
    LoudnessTarget,
    apply_filter,
    is_silent,
    measure_file,
    measure_filter,
    parse_loudnorm_json,
)
from ..ffmpeg.probe import probe
from ..ffmpeg.runner import ffmpeg, ffprobe_json
from ..schemas import EDL, ClipsManifest, RenderQC, Transcript
from ..workspace import Workspace
from .captions import write_captions, write_srt
from .chapters import chapter_marks, chapter_warnings, description_block, ffmetadata

MAX_INPUTS = 24


@dataclass
class RenderResult:
    output: Path
    qc: RenderQC


def timeline(edl: EDL, manifest: ClipsManifest) -> tuple[list[Segment], list[float]]:
    """Frame-snapped segments and their output start times."""
    segs, starts, t = [], [], 0.0
    for clip in edl.clips:
        info = manifest.get(clip.source_clip_id)
        if info is None:
            raise ValueError(f"{clip.clip_id}: source {clip.source_clip_id} not in manifest")
        start = snap_to_frame(clip.start, info.fps)
        dur = quantize(clip.end - start, edl.fps)
        segs.append(Segment(info.source_file, round(start, 6), dur, info.has_audio,
                            clip.crop_center, clip.zoom))
        starts.append(round(t, 6))
        t += dur
    return segs, starts


def overlay_specs(edl: EDL, starts: list[float], manifest: ClipsManifest) -> list[OverlaySpec]:
    by_id = {c.clip_id: s for c, s in zip(edl.clips, starts, strict=True)}
    out = []
    for ov in edl.overlays:
        if ov.anchor_clip_id not in by_id or manifest.get(ov.source_clip_id) is None:
            continue
        at = quantize(by_id[ov.anchor_clip_id] + ov.offset, edl.fps)
        out.append(OverlaySpec(ov.source_file, ov.src_start, at, quantize(ov.duration, edl.fps),
                               zone=ov.zone, corner=ov.corner))
    return out


def music_specs(edl: EDL, starts: list[float], total: float) -> list[MusicSpec]:
    """Place every music/SFX track on the output timeline."""
    by_id = {c.clip_id: s for c, s in zip(edl.clips, starts, strict=True)}
    out = []
    for m in edl.music:
        if m.anchor_clip_id not in by_id:
            continue
        at = by_id[m.anchor_clip_id] + m.offset
        room = total - at
        if m.duration is not None:
            dur = m.duration
        elif m.role == "music" or m.loop:
            dur = room
        else:  # a sound effect plays once, to the end of its file
            fmt = ffprobe_json(Path(m.file).expanduser()).get("format", {})
            dur = max(0.05, float(fmt.get("duration") or 0) - m.src_start)
        dur = min(dur, room)
        if dur <= 0.05:
            continue
        out.append(MusicSpec(str(Path(m.file).expanduser().resolve()), round(at, 6),
                             round(dur, 6), m.src_start, m.volume_db,
                             m.duck and m.role != "voiceover", m.loop, m.fade_in, m.fade_out,
                             m.role))
    return out


def measure_speech(segs: list[Segment], edl: EDL, work: Path, tracks: list[MusicSpec],
                   total: float, max_inputs: int, workers: int, target: LoudnessTarget,
                   mezz: list[Path] | None) -> tuple[Ducking, list[Path] | None]:
    """Ducking whose sidechain key is levelled to the measured speech loudness.

    Speech = the clips' own audio plus voice-over tracks. Without this, a quiet
    recording never crosses a fixed threshold (no ducking) and a loud one ducks
    all the time.
    """
    m = edl.mix
    base = Ducking(m.duck_depth_db, m.attack_ms, m.release_ms)
    if not any(t.duck for t in tracks):
        return base, mezz
    g, _, a, mezz = _base_graph(segs, edl, work, max_inputs, workers, audio_only=True,
                                mezz=mezz)
    a = add_speech(g, a, tracks, total=total)
    g.filters.append(f"[{a}]{measure_filter(target)}[meas]")
    proc = ffmpeg(*g.input_args, "-filter_complex", g.filter_complex, "-map", "[meas]",
                  "-f", "null", "-", timeout=None)
    speech = parse_loudnorm_json(proc.stderr).get("input_i")
    if speech is None or speech < -70:  # no speech at all: nothing to duck under
        return Ducking(m.duck_depth_db, m.attack_ms, m.release_ms, key_gain_db=-60.0), mezz
    gain = max(-30.0, min(50.0, KEY_LEVEL_DB - speech))
    return Ducking(m.duck_depth_db, m.attack_ms, m.release_ms, key_gain_db=round(gain, 2)), mezz


def _render_mezzanine(i: int, segs: list[Segment], edl: EDL, out_dir: Path) -> Path:
    g = Graph()
    v, a = add_segments(g, segs, width=edl.width, height=edl.height, fps=edl.fps,
                        mode=edl.fit_mode, prefix=f"m{i}")
    out = out_dir / f"mezz_{i:03d}.mkv"
    ffmpeg(*g.input_args, "-filter_complex", g.filter_complex, "-map", f"[{v}]", "-map", f"[{a}]",
           *MEZZANINE_VIDEO, "-r", str(edl.fps), *MEZZANINE_AUDIO, str(out), timeout=None)
    return out


def _base_graph(segs: list[Segment], edl: EDL, work: Path, max_inputs: int, workers: int,
                audio_only: bool = False, mezz: list[Path] | None = None):
    """Graph producing the concatenated timeline (direct or from mezzanine files)."""
    g = Graph()
    if len(segs) <= max_inputs:
        v, a = add_segments(g, segs, width=edl.width, height=edl.height, fps=edl.fps,
                            mode=edl.fit_mode, audio_only=audio_only)
        return g, v, a, mezz
    if mezz is None:
        groups = [segs[i:i + max_inputs] for i in range(0, len(segs), max_inputs)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            mezz = list(pool.map(lambda ia: _render_mezzanine(ia[0], ia[1], edl, work),
                                 enumerate(groups)))
    listfile = work / "mezzanine.txt"
    listfile.write_text("".join(f"file '{p.name}'\n" for p in mezz))
    g.add_input("-f", "concat", "-safe", "0", "-i", str(listfile))
    return g, (None if audio_only else "0:v:0"), "0:a:0", mezz


def _label(x: str) -> str:
    return f"[{x}]"


def render(ws: Workspace, edl: EDL | None = None, *, out: Path | None = None,
           encoder: str = "auto", quality: Quality = "high", burn_captions: bool = True,
           target: LoudnessTarget = DEFAULT_TARGET, max_inputs: int = MAX_INPUTS,
           workers: int = 2, require_review: bool = False) -> RenderResult:
    from .geometry import apply_geometry, load_geometry
    from .review import check_approved

    edl = edl or Workspace.read(ws.edl_audited, EDL)
    if not edl.clips:
        raise ValueError("EDL has no clips to render")
    if require_review:
        check_approved(ws, edl)
    edl = apply_geometry(edl, load_geometry(ws, edl))
    manifest = Workspace.read(ws.clips_manifest, ClipsManifest)
    work = ws.render_dir
    work.mkdir(parents=True, exist_ok=True)
    out = out or work / f"{edl.target_platform}.mp4"
    enc = detect_encoder(encoder)

    segs, starts = timeline(edl, manifest)
    expected = sum(s.duration for s in segs)
    # motion graphics first: a broken composition fails before the long encode
    from .graphics import graphic_specs

    gfx = graphic_specs(ws, edl, starts, quality="draft" if quality == "draft" else "high")

    tracks = music_specs(edl, starts, expected)
    ducking, mezz = measure_speech(segs, edl, work, tracks, expected, max_inputs, workers,
                                   target, None)

    # Pass 1: measure loudness of the programme audio (with music, if any).
    g, _, a, mezz = _base_graph(segs, edl, work, max_inputs, workers, audio_only=True,
                                mezz=mezz)
    if mezz is None or tracks:
        a = add_music(g, a, tracks, total=expected, ducking=ducking)
        g.filters.append(f"[{a}]{measure_filter(target)}[meas]")
        proc = ffmpeg(*g.input_args, "-filter_complex", g.filter_complex, "-map", "[meas]",
                      "-f", "null", "-", timeout=None)
    else:
        proc = ffmpeg(*g.input_args, "-map", "0:a:0", "-af", measure_filter(target),
                      "-f", "null", "-", timeout=None)
    measured = parse_loudnorm_json(proc.stderr)

    # Pass 2: the one real encode.
    g, v, a, _ = _base_graph(segs, edl, work, max_inputs, workers, mezz=mezz)
    base = add_overlays(g, v, overlay_specs(edl, starts, manifest), width=edl.width,
                        height=edl.height, fps=edl.fps, mode=edl.fit_mode)
    base = add_graphics(g, base, gfx, width=edl.width, height=edl.height, fps=edl.fps)
    a = add_music(g, a, tracks, total=expected, ducking=ducking)
    ass = None
    if burn_captions and edl.captions.enabled and ws.transcript.exists():
        ass = write_captions(work / "captions.ass", edl, Workspace.read(ws.transcript, Transcript),
                             starts, [s.duration for s in segs])
    vchain = f"{_label(base)}ass=captions.ass[vout]" if ass else f"{_label(base)}null[vout]"
    g.filters.append(vchain)
    marks = chapter_marks(edl, starts)
    chapter_args: list[str] = []
    if marks:
        meta = work / "chapters.ffmeta"
        meta.write_text(ffmetadata(marks, expected), encoding="utf-8")
        idx = g.add_input("-f", "ffmetadata", "-i", str(meta.resolve()))
        chapter_args = ["-map_chapters", str(idx)]
    g.filters.append(f"{_label(a)}{apply_filter(measured, target)},aresample=48000,"
                     f"atrim=duration={expected:.6f}[aout]")
    ffmpeg(*g.input_args, "-filter_complex", g.filter_complex, "-map", "[vout]", "-map", "[aout]",
           *chapter_args, *video_args(enc, edl.fps, quality), *AUDIO_ARGS,
           "-movflags", "+faststart",
           "-t", f"{expected:.6f}", str(out.resolve()), timeout=None, cwd=work)

    qc = quality_check(out, expected, edl, enc, target, silent=is_silent(measured))
    srt = None
    if ws.transcript.exists():
        srt = write_srt(out.with_suffix(".srt"), edl, Workspace.read(ws.transcript, Transcript),
                        starts, [s.duration for s in segs])
    if marks:
        out.with_suffix(".chapters.txt").write_text(description_block(marks), encoding="utf-8")
    qc = qc.model_copy(update={"warnings": chapter_warnings(marks, expected),
                               "chapters": [(t, title) for t, title in marks],
                               "captions_srt": str(srt) if srt else None})
    Workspace.write(work / "qc_report.json", qc)
    for p in mezz or []:
        p.unlink(missing_ok=True)
    ws.record("render", ws.stage_key("render", edl.model_dump(), enc, quality),
              [out, work / "qc_report.json"])
    return RenderResult(out, qc)


def quality_check(path: Path, expected: float, edl: EDL, encoder: str,
                  target: LoudnessTarget = DEFAULT_TARGET, silent: bool = False) -> RenderQC:
    info = probe(path)
    streams = ffprobe_json(path).get("streams", [])
    vdur = next((float(s.get("duration", 0)) for s in streams if s["codec_type"] == "video"), 0.0)
    adur = next((float(s.get("duration", 0)) for s in streams if s["codec_type"] == "audio"), 0.0)
    frame = 1 / edl.fps
    problems = []
    if abs(info.duration - expected) > 2 * frame:
        problems.append(f"duration {info.duration:.3f}s != expected {expected:.3f}s")
    if abs(vdur - adur) > frame + 1024 / 48000:
        problems.append(f"video/audio stream lengths differ by {abs(vdur - adur) * 1000:.0f} ms")
    if (info.width, info.height) != (edl.width, edl.height):
        problems.append(f"frame size {info.width}x{info.height} != {edl.width}x{edl.height}")
    if not math.isclose(info.fps, edl.fps, abs_tol=0.01):
        problems.append(f"frame rate {info.fps} != {edl.fps}")
    if info.sample_rate != 48000 or info.channels != 2:
        problems.append(f"audio {info.sample_rate} Hz / {info.channels} ch, expected 48000/2")
    lufs = tp = None
    if not silent:
        m = measure_file(path, target)
        lufs, tp = m.get("input_i"), m.get("input_tp")
        if lufs is None or abs(lufs - target.integrated) > 1.0:
            problems.append(f"integrated loudness {lufs} LUFS, target {target.integrated}")
        if tp is not None and tp > target.true_peak + 0.5:
            problems.append(f"true peak {tp} dBTP above {target.true_peak}")
    return RenderQC(
        output=str(path), passed=not problems, encoder=encoder, expected_duration=round(expected, 4),
        duration=round(info.duration, 4), video_duration=round(vdur, 4),
        audio_duration=round(adur, 4), av_offset=round(abs(vdur - adur), 4), width=info.width,
        height=info.height, fps=info.fps, sample_rate=info.sample_rate, channels=info.channels,
        integrated_lufs=lufs, true_peak=tp, target_lufs=target.integrated, problems=problems,
    )
