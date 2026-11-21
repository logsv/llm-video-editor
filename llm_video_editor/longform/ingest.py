"""
Ingest a long recording for a Claude-driven edit.

Writes into <project>/:
  source.json      path, duration, size, fps, VFR flag
  words.json       [{"w": "word", "s": 1.23, "e": 1.50, "p": 0.98}, ...]  (source seconds)
  silences.json    [[start, end], ...] from ffmpeg silencedetect
  transcript.md    compact, timestamped paragraphs for Claude to read and plan from
  work/sheet.jpg   24 frames across the recording (setups, screen share, b-roll)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from fractions import Fraction
from typing import Any, Dict, List, Optional

from .edl import hms


def run(cmd: List[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)


def probe(path: str) -> Dict[str, Any]:
    r = run(["ffprobe", "-v", "error", "-show_entries",
             "format=duration,start_time:stream=codec_type,width,height,r_frame_rate,avg_frame_rate",
             "-of", "json", path])
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {r.stderr.strip()}")
    j = json.loads(r.stdout)
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), None)
    has_audio = any(s["codec_type"] == "audio" for s in j["streams"])
    if v is None:
        raise RuntimeError(f"{path} has no video stream")
    r_fps = float(Fraction(v["r_frame_rate"]))
    avg = v.get("avg_frame_rate", "0/0")
    avg_fps = float(Fraction(avg)) if avg not in ("0/0", "") else r_fps
    return {
        "path": os.path.abspath(path),
        "duration": float(j["format"]["duration"]),
        "start_time": float(j["format"].get("start_time", 0) or 0),
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": r_fps,
        "fps_fraction": v["r_frame_rate"],
        "vfr": abs(r_fps - avg_fps) > 0.05,
        "has_audio": has_audio,
    }


def extract_audio(src: str, wav: str) -> None:
    r = run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", wav])
    if r.returncode != 0:
        raise RuntimeError(f"audio extraction failed: {r.stderr.strip()}")


def detect_silences(wav: str, noise_db: float = -35, min_dur: float = 0.4) -> List[List[float]]:
    r = run(["ffmpeg", "-hide_banner", "-i", wav, "-af",
             f"silencedetect=n={noise_db}dB:d={min_dur}", "-f", "null", "-"])
    out, start = [], None
    for line in r.stderr.splitlines():
        m = re.search(r"silence_start: ([\d.]+)", line)
        if m:
            start = float(m.group(1))
        m = re.search(r"silence_end: ([\d.]+)", line)
        if m and start is not None:
            out.append([round(start, 3), round(float(m.group(1)), 3)])
            start = None
    return out


def transcribe(wav: str, model_size: str, language: Optional[str], keywords: Optional[str],
               device: str = "auto") -> List[Dict[str, Any]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover - depends on env
        raise SystemExit("faster-whisper is not installed. Run `uv sync` in the repo first.") from e

    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute)
    segments, info = model.transcribe(
        wav, language=language, word_timestamps=True, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        initial_prompt=keywords or None, condition_on_previous_text=False,
    )
    words = []
    last_print = 0.0
    for seg in segments:
        for w in seg.words or []:
            words.append({"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3),
                          "p": round(w.probability, 2)})
        if seg.end - last_print > 300:
            print(f"   transcribed {hms(seg.end)} / {hms(info.duration)}", flush=True)
            last_print = seg.end
    return words


def _sentences(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group words into sentence-ish units (break on punctuation or a pause > 0.8s)."""
    out, cur = [], []
    for i, w in enumerate(words):
        if cur and w["s"] - cur[-1]["e"] > 0.8:
            out.append(cur); cur = []
        cur.append(w)
        if w["w"].endswith((".", "?", "!")) or len(cur) >= 40:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return [{"s": c[0]["s"], "e": c[-1]["e"], "text": " ".join(x["w"] for x in c),
             "low": sum(1 for x in c if x.get("p", 1) < 0.5)} for c in out]


def _key(text: str, n: int = 4) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]", "", text.lower()).split()[:n])


def write_transcript(words: List[Dict[str, Any]], silences: List[List[float]], src: Dict[str, Any],
                     path: str, para_secs: float = 45.0) -> None:
    sents = _sentences(words)
    # Mark likely retakes: a sentence whose opening words repeat in one of the next 3 sentences.
    for i, s in enumerate(sents):
        k = _key(s["text"])
        if len(k.split()) >= 3 and any(_key(n["text"]) == k for n in sents[i + 1:i + 4]):
            s["retake"] = True

    long_sil = [s for s in silences if s[1] - s[0] >= 2.0]
    lines = [
        f"# Transcript: {os.path.basename(src['path'])}",
        "",
        f"Duration {hms(src['duration'])} ({src['duration']:.1f}s), {src['width']}x{src['height']} "
        f"@ {src['fps']:.3f} fps{' (VARIABLE frame rate)' if src['vfr'] else ''}. "
        f"{len(words)} words, {len(long_sil)} pauses of 2s or more.",
        "",
        "Each line: `[start-end]` in SOURCE seconds, then the text. Markers:",
        "`(retake?)` the next lines restart this sentence (keep the last clean take),",
        "`(unclear N)` N low-confidence words (check names/keywords), `-- pause Xs --` dead air.",
        "",
    ]
    para_start = None
    sil_i = 0
    for s in sents:
        while sil_i < len(long_sil) and long_sil[sil_i][1] <= s["s"]:
            a, b = long_sil[sil_i]
            lines.append(f"-- pause {b - a:.1f}s at {a:.2f} --")
            sil_i += 1
        if para_start is None or s["s"] - para_start > para_secs:
            lines.append("")
            lines.append(f"## {hms(s['s'])}")
            para_start = s["s"]
        tags = (" (retake?)" if s.get("retake") else "") + (f" (unclear {s['low']})" if s["low"] else "")
        lines.append(f"[{s['s']:.2f}-{s['e']:.2f}] {s['text']}{tags}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def contact_sheet(src: Dict[str, Any], out_jpg: str, n: int = 24, cols: int = 6) -> Optional[str]:
    tmp = os.path.join(os.path.dirname(out_jpg), "sheet_frames")
    os.makedirs(tmp, exist_ok=True)
    dur = src["duration"]
    for i in range(n):
        t = dur * (i + 0.5) / n
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", src["path"], "-frames:v", "1",
             "-vf", "scale=320:-2", os.path.join(tmp, f"f{i:03d}.jpg")])
    r = run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(tmp, "f%03d.jpg"), "-vf",
             f"tile={cols}x{(n + cols - 1) // cols}:padding=4", "-frames:v", "1", out_jpg])
    shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode != 0:
        return None
    with open(os.path.splitext(out_jpg)[0] + ".txt", "w", encoding="utf-8") as f:
        f.write("Contact sheet frames, left to right, top to bottom:\n")
        f.write("\n".join(f"{i + 1:2d}. {hms(dur * (i + 0.5) / n)}" for i in range(n)) + "\n")
    return out_jpg


def ingest(source: str, project: str, model: str = "small", language: Optional[str] = None,
           keywords: Optional[str] = None, words_json: Optional[str] = None,
           device: str = "auto", sheet: bool = True) -> Dict[str, Any]:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SystemExit("ffmpeg/ffprobe not found on PATH (macOS: brew install ffmpeg, Ubuntu: apt install ffmpeg)")
    work = os.path.join(project, "work")
    os.makedirs(work, exist_ok=True)

    src = probe(source)
    print(f"source: {src['path']}\n   {hms(src['duration'])}, {src['width']}x{src['height']} @ {src['fps']:.3f}fps")
    if src["vfr"]:
        print("   WARNING: variable frame rate (typical of phone/screen recordings). Cuts are still "
              "frame-accurate because render re-encodes to constant frame rate, but check A/V sync.")
    if not src["has_audio"]:
        raise SystemExit("source has no audio track; nothing to transcribe")
    with open(os.path.join(project, "source.json"), "w", encoding="utf-8") as f:
        json.dump(src, f, indent=2)

    wav = os.path.join(work, "audio16k.wav")
    print("extracting audio...")
    extract_audio(src["path"], wav)
    silences = detect_silences(wav)
    with open(os.path.join(project, "silences.json"), "w", encoding="utf-8") as f:
        json.dump(silences, f)

    words_path = os.path.join(project, "words.json")
    if words_json:
        if os.path.abspath(words_json) != os.path.abspath(words_path):
            shutil.copyfile(words_json, words_path)
    elif os.path.exists(words_path):
        print("words.json already exists; reusing it (delete it to re-transcribe)")
    else:
        print(f"transcribing with faster-whisper '{model}' (long recordings take a while)...")
        words = transcribe(wav, model, language, keywords, device)
        with open(words_path, "w", encoding="utf-8") as f:
            json.dump(words, f)
    with open(words_path, encoding="utf-8") as f:
        words = json.load(f)

    write_transcript(words, silences, src, os.path.join(project, "transcript.md"))
    sheet_path = contact_sheet(src, os.path.join(work, "sheet.jpg")) if sheet else None
    print(f"done: {len(words)} words, {len(silences)} silences")
    print(f"   read {os.path.join(project, 'transcript.md')}")
    if sheet_path:
        print(f"   look at {sheet_path}")
    return src
