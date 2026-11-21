from llm_video_editor.schemas import ClipsManifest, SheetIndex, Transcript
from llm_video_editor.stages.watch import watch, watch_window
from llm_video_editor.workspace import Workspace
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeTranscriber


@requires_ffmpeg
def test_watch_multi_clip_folder(media_dir, tmp_path):
    ws = Workspace.open(media_dir, home=tmp_path)
    asr = FakeTranscriber()
    res = watch(ws, transcriber=asr)
    assert not res.cached
    kinds = {c.source_file.rsplit("/", 1)[-1]: c.kind for c in res.manifest.clips}
    assert kinds == {
        "a_talk_1080p30.mp4": "talking_head",
        "b_talk_720p60.mov": "talking_head",
        "c_broll_720p24.mp4": "broll",
    }
    assert len(asr.calls) == 2  # silent B-roll clip has no audio stream to transcribe
    for c in res.manifest.clips:
        assert c.overview_sheets and (ws.root / c.overview_sheets[0]).exists()
        assert c.scenes and c.scenes[-1].end == c.duration
        assert c.motion_score > 0  # testsrc2 moves
    # checkpoints on disk validate against their schemas
    Workspace.read(ws.clips_manifest, ClipsManifest)
    Workspace.read(ws.transcript, Transcript)
    idx = Workspace.read(ws.sheets_index, SheetIndex)
    assert idx.est_tokens > 0
    # nothing in the manifest or transcripts uses source paths as identifiers
    assert all(c.clip_id.startswith("v") for c in res.manifest.clips)

    again = watch(ws, transcriber=asr)
    assert again.cached and len(asr.calls) == 2


@requires_ffmpeg
def test_speechless_audio_is_broll(media_dir, tmp_path):
    ws = Workspace.open(media_dir / "a_talk_1080p30.mp4", home=tmp_path)
    res = watch(ws, transcriber=FakeTranscriber(silent=True))
    assert res.manifest.clips[0].kind == "broll"
    assert res.manifest.clips[0].speech_ratio == 0


@requires_ffmpeg
def test_watch_window_appends_dense_sheet(media_dir, tmp_path):
    ws = Workspace.open(media_dir / "a_talk_1080p30.mp4", home=tmp_path)
    res = watch(ws, transcriber=FakeTranscriber())
    cid = res.manifest.clips[0].clip_id
    dense = watch_window(ws, cid, 1.0, 3.0)
    assert len(dense) == 1 and dense[0].kind == "dense" and dense[0].fps == 2.0
    assert len(dense[0].tiles) == 4
    idx = Workspace.read(ws.sheets_index, SheetIndex)
    assert any(s.kind == "dense" for s in idx.sheets)
    watch_window(ws, cid, 1.0, 3.0)  # idempotent
    assert sum(s.kind == "dense" for s in Workspace.read(ws.sheets_index, SheetIndex).sheets) == 1


@requires_ffmpeg
def test_watch_chunked_asr_matches_unchunked_and_caches(media_dir, tmp_path):
    from llm_video_editor.stages.chunked_asr import ChunkConfig
    from tests.fakes import TimelineTranscriber, words_for

    truth = words_for(" ".join(f"w{i}" for i in range(12)), 0.15)  # 0.15 .. 3.75 s
    cfg = ChunkConfig(min_duration=2.0, chunk_seconds=1.5, overlap=0.6, search_window=0.2,
                      workers=1)
    ws = Workspace.open(media_dir / "a_talk_1080p30.mp4", home=tmp_path)
    asr = TimelineTranscriber(truth)
    res = watch(ws, transcriber=asr, chunking=cfg)
    assert len(asr.calls) == 3 and all(p.name.startswith("chunk_") for p in asr.calls)
    ct = res.transcript.clips[0]
    assert [s.id for s in ct.segments] == list(range(len(ct.segments)))
    assert [(w.text, w.start, w.end) for w in ct.words()] == [
        (w.text, w.start, w.end) for w in truth]
    Workspace.read(ws.transcript, Transcript)  # schema unchanged
    assert not list(ws.clip_dir(ct.clip_id).glob("asr_chunks*"))

    # cached: same settings -> no new ASR calls, even with a different worker count
    watch(ws, transcriber=asr, chunking=ChunkConfig(**{**cfg.__dict__, "workers": 2}))
    assert len(asr.calls) == 3
    # chunking settings are part of the key -> re-transcribe
    watch(ws, transcriber=asr, chunking=ChunkConfig(**{**cfg.__dict__, "overlap": 0.4}))
    assert len(asr.calls) == 6
    # same words with chunking off (one call over the whole clip)
    plain = watch(ws, transcriber=asr)
    assert len(asr.calls) == 7
    assert [w.text for w in plain.transcript.clips[0].words()] == [w.text for w in truth]


@requires_ffmpeg
def test_watch_short_clip_not_chunked_keeps_cache_key(media_dir, tmp_path):
    from llm_video_editor.stages.chunked_asr import ChunkConfig

    ws = Workspace.open(media_dir / "a_talk_1080p30.mp4", home=tmp_path)
    asr = FakeTranscriber()
    watch(ws, transcriber=asr)
    watch(ws, transcriber=asr, chunking=ChunkConfig())  # 4 s clip < 10 min threshold
    assert len(asr.calls) == 1  # per-clip ASR cache reused; whole file, not chunks
