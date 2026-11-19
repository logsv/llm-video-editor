"""Stage 1 - ``watch``: probe, transcribe, score scenes and build contact sheets.

Outputs (in the workspace):
  transcript.json       word-level transcript per clip
  clips_manifest.json   per-clip metadata + talking_head / broll classification
  sheets/index.json     tile -> source timestamp map for every contact sheet
  sheets/*.png          overview sheets (whole clip, low fps)
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from ..ffmpeg.probe import ProbeResult, probe
from ..ffmpeg.scenes import SceneAnalysis, analyze_scenes
from ..ffmpeg.sheets import SheetResult, SheetSpec, estimate_tokens, make_sheets
from ..schemas import (
    ClipInfo,
    ClipsManifest,
    ClipTranscript,
    SceneCut,
    Sheet,
    SheetIndex,
    Tile,
    Transcript,
)
from ..workspace import SourceFingerprint, Workspace
from .asr import FasterWhisperTranscriber, Transcriber, extract_audio
from .chunked_asr import ChunkConfig, transcribe_chunked

OVERVIEW = SheetSpec(fps=0.5, cols=4, rows=4)
DENSE = SheetSpec(fps=2.0, cols=4, rows=4)


@dataclass
class WatchResult:
    manifest: ClipsManifest
    transcript: Transcript
    sheets: SheetIndex
    cached: bool


def _sheet_models(ws: Workspace, clip_id: str, kind: str, spec: SheetSpec,
                  results: list[SheetResult], start: float, end: float) -> list[Sheet]:
    out = []
    for r in results:
        out.append(Sheet(
            path=ws.rel(r.path), clip_id=clip_id, kind=kind, fps=spec.fps,
            cols=spec.cols, rows=spec.rows, tile_width=r.tile_size[0], tile_height=r.tile_size[1],
            start=r.tile_times[0] if r.tile_times else start,
            end=min(end, (r.tile_times[-1] + 1 / spec.fps) if r.tile_times else end),
            tiles=[Tile(index=i, t=t) for i, t in enumerate(r.tile_times)],
            est_tokens=estimate_tokens(*r.sheet_size),
        ))
    return out


def _transcribe_clip(ws: Workspace, fp: SourceFingerprint, info: ProbeResult,
                     transcriber: Transcriber, language: str | None,
                     chunking: ChunkConfig | None = None) -> ClipTranscript:
    cdir = ws.clip_dir(fp.clip_id)
    cached = cdir / "transcript.json"
    chunked = chunking is not None and info.has_audio and chunking.applies(info.duration)
    parts: list = ["asr", fp.key, transcriber.name, language]
    if chunked:  # only appended when used, so unchunked clips keep their existing cache
        parts.append(chunking.cache_key())
    key = ws.stage_key(*parts)
    if ws.is_fresh(f"asr:{fp.clip_id}", key):
        return Workspace.read(cached, ClipTranscript)
    lang, segments = None, []
    if info.has_audio:
        wav = cdir / "audio_16k.wav"
        try:
            extract_audio(fp.path, wav)
            if chunked:
                lang, segments = transcribe_chunked(transcriber, wav, language, chunking,
                                                    cdir / "asr_chunks")
            else:
                lang, segments = transcriber.transcribe(wav, language)
        finally:
            wav.unlink(missing_ok=True)
    ct = ClipTranscript(clip_id=fp.clip_id, source_file=fp.path, language=lang,
                        duration=info.duration, segments=segments)
    Workspace.write(cached, ct)
    ws.record(f"asr:{fp.clip_id}", key, [cached])
    return ct


def watch(
    ws: Workspace,
    *,
    transcriber: Transcriber | None = None,
    language: str | None = None,
    overview: SheetSpec = OVERVIEW,
    scene_threshold: float = 0.3,
    workers: int = 4,
    chunking: ChunkConfig | None = None,
) -> WatchResult:
    """``chunking`` enables chunked/parallel ASR for clips >= ``chunking.min_duration``."""
    transcriber = transcriber or FasterWhisperTranscriber()
    fps = ws.fingerprints()
    parts: list = ["watch", [f.key for f in fps], transcriber.name, language,
                   overview, scene_threshold]
    if chunking is not None:
        parts.append(chunking.cache_key())
    key = ws.stage_key(*parts)
    if ws.is_fresh("watch", key):
        return WatchResult(
            Workspace.read(ws.clips_manifest, ClipsManifest),
            Workspace.read(ws.transcript, Transcript),
            Workspace.read(ws.sheets_index, SheetIndex),
            cached=True,
        )

    probes = {f.clip_id: probe(f.path) for f in fps}
    # ffmpeg-bound work (scene scoring, sheets) runs in threads while ASR runs here.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scene_jobs: dict[str, Future[SceneAnalysis]] = {}
        sheet_jobs: dict[str, Future[list[SheetResult]]] = {}
        for f in fps:
            p = probes[f.clip_id]
            scene_jobs[f.clip_id] = pool.submit(
                analyze_scenes, f.path, p.duration, threshold=scene_threshold)
            sheet_jobs[f.clip_id] = pool.submit(
                make_sheets, f.path, ws.sheets_dir, f"{f.clip_id}_overview",
                src_size=(p.width, p.height), start=0.0, end=p.duration, spec=overview)
        transcripts = [
            _transcribe_clip(ws, f, probes[f.clip_id], transcriber, language, chunking)
            for f in fps
        ]
        scenes = {cid: job.result() for cid, job in scene_jobs.items()}
        sheets = {cid: job.result() for cid, job in sheet_jobs.items()}

    clips, sheet_models = [], []
    for f, ct in zip(fps, transcripts, strict=True):
        p, sa = probes[f.clip_id], scenes[f.clip_id]
        models = _sheet_models(ws, f.clip_id, "overview", overview, sheets[f.clip_id],
                               0.0, p.duration)
        sheet_models += models
        speech = sum(s.end - s.start for s in ct.segments)
        clips.append(ClipInfo(
            clip_id=f.clip_id, source_file=f.path, duration=round(p.duration, 3),
            width=p.width, height=p.height, fps=p.fps, has_audio=p.has_audio,
            video_codec=p.video_codec, audio_codec=p.audio_codec,
            kind="broll" if not ct.segments else "talking_head",
            speech_seconds=round(speech, 3),
            speech_ratio=round(speech / p.duration, 4) if p.duration else 0.0,
            scene_change_rate=sa.cuts_per_minute, motion_score=sa.motion_score,
            scenes=[SceneCut(start=a, end=b) for a, b in sa.scenes],
            overview_sheets=[m.path for m in models],
        ))

    manifest = ClipsManifest(batch_hash=ws.batch_hash, clips=clips)
    transcript = Transcript(clips=transcripts)
    index = SheetIndex(sheets=sheet_models)
    Workspace.write(ws.clips_manifest, manifest)
    Workspace.write(ws.transcript, transcript)
    Workspace.write(ws.sheets_index, index)
    ws.record("watch", key, [ws.clips_manifest, ws.transcript, ws.sheets_index])
    return WatchResult(manifest, transcript, index, cached=False)


def watch_window(
    ws: Workspace, clip_id: str, start: float, end: float, *, spec: SheetSpec = DENSE
) -> list[Sheet]:
    """Dense sheets (default 2 FPS) for one candidate window; appended to the index."""
    manifest = Workspace.read(ws.clips_manifest, ClipsManifest)
    clip = manifest.get(clip_id)
    if clip is None:
        raise KeyError(f"Unknown clip_id {clip_id}")
    start, end = max(0.0, start), min(clip.duration, end)
    if end <= start:
        raise ValueError("Empty window")
    stem = f"{clip_id}_dense_{start:07.2f}_{end:07.2f}".replace(".", "p")
    results = make_sheets(clip.source_file, ws.sheets_dir, stem, src_size=(clip.width, clip.height),
                          start=start, end=end, spec=spec)
    models = _sheet_models(ws, clip_id, "dense", spec, results, start, end)
    index = Workspace.read(ws.sheets_index, SheetIndex)
    index.sheets = [s for s in index.sheets if not s.path.startswith(f"sheets/{stem}_")] + models
    Workspace.write(ws.sheets_index, index)
    return models


__all__ = ["DENSE", "OVERVIEW", "WatchResult", "watch", "watch_window"]
