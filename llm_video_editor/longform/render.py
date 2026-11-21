"""
Render <project>/edl.json into a YouTube-ready file.

Pipeline:
  1. build_plan(): segments -> frame-snapped ranges (optional jump-cut tightening, punch-ins)
  2. one ffmpeg pass per forward-moving run of ranges (split/trim/concat inside a
     filter graph: frame-accurate, no keyframe snapping, no concat-demuxer A/V drift)
  3. join runs, two-pass EBU R128 loudnorm (-14 LUFS, -1 dBTP), AAC 48k, embed chapters,
     +faststart
  4. write captions .srt + chapters .txt for the edited timeline, a cut-check contact
     sheet and report.json
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

from .edl import Plan, Range, build_plan, captions_srt, chapters_text, ffmetadata, hms, load_edl, runs
from .ingest import probe, run

ENCODERS = {
    "x264": ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high"],
    "nvenc": ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0",
              "-profile:v", "high"],
    "videotoolbox": ["-c:v", "h264_videotoolbox", "-q:v", "65", "-profile:v", "high"],
}
PREVIEW_ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "30"]


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def video_chain(r: Range, src: Dict[str, Any], out_w: int, out_h: int, vertical: bool) -> str:
    """crop (vertical and/or punch-in) + scale for one range."""
    W, H = src["width"], src["height"]
    bw, bh = (min(W, _even(H * 9 / 16)), H) if vertical else (W, H)
    z = max(1.0, r.zoom)
    cw, ch = min(W, _even(bw / z)), min(H, _even(bh / z))
    parts = []
    if (cw, ch) != (W, H):
        x = max(0, min(W - cw, int((W - cw) * r.cx)))
        y = max(0, min(H - ch, int((H - ch) * r.cy)))
        parts.append(f"crop={cw}:{ch}:{x}:{y}")
    if (cw, ch) != (out_w, out_h):
        parts.append(f"scale={out_w}:{out_h}:flags=lanczos")
    parts.append("setsar=1")
    return ",".join(parts)


def run_graph(rs: List[Range], src: Dict[str, Any], out_w: int, out_h: int, vertical: bool,
              offset: float) -> str:
    k = len(rs)
    fade = 0.008  # 8 ms fades at every join remove clicks without audible dips
    g = []
    if k > 1:
        g.append(f"[0:v]split={k}" + "".join(f"[vs{i}]" for i in range(k)))
        g.append(f"[0:a]asplit={k}" + "".join(f"[as{i}]" for i in range(k)))
    for i, r in enumerate(rs):
        vin, ain = (f"[vs{i}]", f"[as{i}]") if k > 1 else ("[0:v]", "[0:a]")
        a, b = r.src_in + offset, r.src_out + offset
        d = r.duration
        g.append(f"{vin}trim=start={a:.6f}:end={b:.6f},setpts=PTS-STARTPTS,"
                 f"{video_chain(r, src, out_w, out_h, vertical)}[v{i}]")
        g.append(f"{ain}atrim=start={a:.6f}:end={b:.6f},asetpts=PTS-STARTPTS,"
                 f"afade=t=in:d={fade},afade=t=out:st={max(0, d - fade):.6f}:d={fade}[a{i}]")
    g.append("".join(f"[v{i}][a{i}]" for i in range(k)) + f"concat=n={k}:v=1:a=1[vout][aout]")
    return ";\n".join(g)


def _ff(cmd: List[str], what: str) -> subprocess.CompletedProcess:
    r = run(cmd)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.strip().splitlines()[-25:])
        raise RuntimeError(f"ffmpeg failed during {what}:\n{tail}")
    return r


def measure_loudness(inputs: List[str], target: float, tp: float) -> Optional[Dict[str, str]]:
    r = run(["ffmpeg", "-hide_banner", "-nostats", *inputs, "-vn", "-af",
             f"loudnorm=I={target}:TP={tp}:LRA=11:print_format=json", "-f", "null", "-"])
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    if not m:
        return None
    j = json.loads(m.group(0))
    if j.get("input_i") in ("-inf", None):
        return None
    return j


def stream_durations(path: str) -> Dict[str, float]:
    r = run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", path])
    out = {}
    for s in json.loads(r.stdout or "{}").get("streams", []):
        if s.get("duration"):
            out[s["codec_type"]] = float(s["duration"])
    return out


def cut_check(video: str, plan: Plan, fps: float, out_jpg: str, max_cuts: int = 30) -> Optional[str]:
    """Frame just before and just after each cut, from the FINAL render (snapshots lie)."""
    cuts = [r.out_start for r in plan.ranges[1:]]
    if not cuts:
        return None
    if len(cuts) > max_cuts:
        step = len(cuts) / max_cuts
        cuts = [cuts[int(i * step)] for i in range(max_cuts)]
    tmp = os.path.join(os.path.dirname(out_jpg), "cutframes")
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    n = 0
    for t in cuts:
        for tt in (t - 1.5 / fps, t + 0.5 / fps):
            run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0, tt):.4f}", "-i", video, "-frames:v", "1",
                 "-vf", "scale=320:-2", os.path.join(tmp, f"c{n:03d}.jpg")])
            n += 1
    r = run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(tmp, "c%03d.jpg"), "-vf",
             f"tile=6x{(n + 5) // 6}:padding=6:color=black", "-frames:v", "1", out_jpg])
    with open(os.path.splitext(out_jpg)[0] + ".txt", "w", encoding="utf-8") as f:
        f.write("Pairs (before | after) per cut, 3 pairs per row:\n")
        f.write("\n".join(f"{i + 1:2d}. cut at {hms(t)} ({t:.2f}s)" for i, t in enumerate(cuts)) + "\n")
    return out_jpg if r.returncode == 0 else None


def next_version(renders: str, slug: str, suffix: str) -> str:
    os.makedirs(renders, exist_ok=True)
    v = 1
    while os.path.exists(os.path.join(renders, f"{slug}-v{v}{suffix}.mp4")):
        v += 1
    return os.path.join(renders, f"{slug}-v{v}{suffix}.mp4")


def render(project: str, edl_path: Optional[str] = None, preview: bool = False, vertical: bool = False,
           height: Optional[int] = None, encoder: str = "x264", lufs: float = -14.0,
           true_peak: float = -1.0, output: Optional[str] = None, dry_run: bool = False,
           caption_fixes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    edl_path = edl_path or os.path.join(project, "edl.json")
    edl = load_edl(edl_path)
    src_path = edl.get("source")
    if not src_path:
        with open(os.path.join(project, "source.json"), encoding="utf-8") as f:
            src_path = json.load(f)["path"]
    src = probe(src_path)
    words_path = os.path.join(project, "words.json")
    words = json.load(open(words_path, encoding="utf-8")) if os.path.exists(words_path) else []
    fps = src["fps"]

    plan = build_plan(edl, words, fps, src["duration"])
    if not plan.ranges:
        raise SystemExit("EDL produced no ranges")
    for w in plan.warnings:
        print(f"   WARNING: {w}")
    groups = runs(plan.ranges)
    removed = src["duration"] - plan.duration
    print(f"plan: {len(edl['segments'])} segments -> {len(plan.ranges)} ranges in {len(groups)} pass(es); "
          f"{hms(src['duration'])} -> {hms(plan.duration)} ({removed:.1f}s removed)")

    if vertical:
        out_w, out_h = 1080, 1920
        if preview:
            out_w, out_h = 540, 960
    else:
        out_h = height or (540 if preview else src["height"])
        out_h = min(out_h, src["height"]) if not height else out_h
        out_w = _even(out_h * src["width"] / src["height"])
        out_h = _even(out_h)

    slug = re.sub(r"[^a-z0-9]+", "-", (edl.get("title") or os.path.basename(os.path.abspath(project))).lower()).strip("-")
    slug = slug[:60] or "edit"
    out = output or next_version(os.path.join(project, "renders"), slug,
                                 ("-short" if vertical else "") + ("-preview" if preview else ""))
    stem = os.path.splitext(out)[0]
    work = os.path.join(project, "work")
    os.makedirs(work, exist_ok=True)

    if dry_run:
        for i, r in enumerate(plan.ranges[:40]):
            print(f"   {i:3d}  src {r.src_in:9.2f}-{r.src_out:9.2f}  -> out {hms(r.out_start)}  zoom {r.zoom}")
        if len(plan.ranges) > 40:
            print(f"   ... {len(plan.ranges) - 40} more")
        return {"plan_duration": plan.duration, "ranges": len(plan.ranges)}

    enc = PREVIEW_ENC if preview else ENCODERS[encoder]
    offset = src.get("start_time", 0.0)
    parts = []
    for gi, rs in enumerate(groups):
        seek = max(0.0, rs[0].src_in + offset - 3.0)
        span = rs[-1].src_out + offset - seek + 1.0
        graph_file = os.path.join(work, f"pass{gi:02d}.ffgraph")
        with open(graph_file, "w", encoding="utf-8") as f:
            f.write(run_graph(rs, src, out_w, out_h, vertical, offset))
        part = os.path.join(work, f"pass{gi:02d}.mkv")
        print(f"   pass {gi + 1}/{len(groups)}: {len(rs)} ranges, src {hms(rs[0].src_in)}-{hms(rs[-1].src_out)}",
              flush=True)
        _ff(["ffmpeg", "-y", "-hide_banner", "-v", "error", "-ss", f"{seek:.3f}", "-t", f"{span:.3f}",
             "-copyts", "-i", src["path"], "-filter_complex_script", graph_file,
             "-map", "[vout]", "-map", "[aout]", *enc, "-pix_fmt", "yuv420p",
             "-r", src["fps_fraction"], "-fps_mode", "cfr",
             "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", part], f"pass {gi + 1}")
        parts.append(part)

    if len(parts) == 1:
        inputs = ["-i", parts[0]]
    else:
        lst = os.path.join(work, "parts.txt")
        with open(lst, "w", encoding="utf-8") as f:
            f.write("".join(f"file '{os.path.basename(p)}'\n" for p in parts))
        inputs = ["-f", "concat", "-safe", "0", "-i", lst]

    meta = os.path.join(work, "chapters.ffmeta")
    with open(meta, "w", encoding="utf-8") as f:
        f.write(ffmetadata(plan.chapters, plan.duration, edl.get("title")))

    af = "aresample=48000"
    if not preview:
        print("   measuring loudness...", flush=True)
        m = measure_loudness(inputs, lufs, true_peak)
        if m:
            af = (f"loudnorm=I={lufs}:TP={true_peak}:LRA=11:measured_I={m['input_i']}:"
                  f"measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
                  f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true,"
                  f"aresample=48000")
    print("   muxing final file...", flush=True)
    _ff(["ffmpeg", "-y", "-hide_banner", "-v", "error", *inputs, "-f", "ffmetadata", "-i", meta,
         "-map", "0:v", "-map", "0:a", "-map_metadata", "1", "-map_chapters", "1",
         "-c:v", "copy", "-af", af, "-c:a", "aac", "-b:a", "192k" if preview else "320k",
         "-movflags", "+faststart", out], "final mux")

    with open(stem + ".srt", "w", encoding="utf-8") as f:
        f.write(captions_srt(words, plan.ranges, replacements={**(edl.get("caption_fixes") or {}),
                                                                **(caption_fixes or {})}))
    with open(stem + ".chapters.txt", "w", encoding="utf-8") as f:
        f.write(chapters_text(plan.chapters))
    sheet = cut_check(out, plan, fps, stem + ".cuts.jpg")

    durs = stream_durations(out)
    drift = abs(durs.get("video", 0) - durs.get("audio", 0))
    report = {
        "output": out, "captions": stem + ".srt", "chapters": stem + ".chapters.txt", "cut_check": sheet,
        "source_duration": src["duration"], "planned_duration": plan.duration,
        "video_duration": durs.get("video"), "audio_duration": durs.get("audio"),
        "av_drift": round(drift, 3), "ranges": len(plan.ranges), "passes": len(groups),
        "size": f"{out_w}x{out_h}", "chapters_list": [[round(t, 2), n] for t, n in plan.chapters],
        "warnings": plan.warnings,
    }
    with open(stem + ".report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"done: {out}")
    print(f"   {hms(durs.get('video', 0))} video / audio drift {drift * 1000:.0f} ms"
          + ("  <-- CHECK SYNC" if drift > 0.1 else ""))
    print(f"   captions {stem}.srt, chapters {stem}.chapters.txt")
    if sheet:
        print(f"   cut check {sheet}")
    return report
