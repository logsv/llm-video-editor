import wave

import pytest

from llm_video_editor.schemas import Segment, Word
from llm_video_editor.stages.chunked_asr import (
    Chunk,
    ChunkConfig,
    merge_chunks,
    parse_silences,
    plan_chunks,
    transcribe_chunked,
)
from tests.conftest import requires_ffmpeg
from tests.fakes import TimelineTranscriber, words_for

CFG = ChunkConfig(min_duration=0, chunk_seconds=10, overlap=2.0, search_window=1.0)


def timeline(duration: float, step: float = 0.3) -> list[Word]:
    n = int((duration - 0.2) / step)
    return words_for(" ".join(f"w{i}" for i in range(n)), 0.1, step=step)


def asr_view(chunk: Chunk, words: list[Word], jitter=lambda c, w: 0.0, per_segment=5):
    """What a model sees for one chunk: whole words inside its audio, maybe shifted."""
    seen = []
    for w in words:
        if w.start >= chunk.start and w.end <= chunk.end:
            d = jitter(chunk, w)
            seen.append(Word(start=round(w.start + d, 3), end=round(w.end + d, 3), text=w.text))
    segs = []
    for i in range(0, len(seen), per_segment):
        g = seen[i:i + per_segment]
        segs.append(Segment(id=len(segs), start=g[0].start, end=g[-1].end,
                            text=" ".join(w.text for w in g), words=g))
    return segs


def check(merged: list[Segment], truth: list[Word]):
    words = [w for s in merged for w in s.words]
    assert [w.text for w in words] == [w.text for w in truth]  # nothing duplicated or dropped
    assert [s.id for s in merged] == list(range(len(merged)))
    for a, b in zip(words, words[1:], strict=False):
        assert a.start <= b.start and a.end <= b.start + 1e-9
    for s in merged:
        assert s.start <= s.end and s.words[0].start >= s.start - 1e-9
        assert s.text == " ".join(w.text for w in s.words)
    assert all(a.start <= b.start for a, b in zip(merged, merged[1:], strict=False))
    return words


def test_plan_fixed_cuts_without_silence():
    chunks = plan_chunks(30.0, [], CFG)
    assert [(c.lo, c.hi) for c in chunks] == [(0, 10), (10, 20), (20, 30)]
    assert [(c.start, c.end) for c in chunks] == [(0, 11), (9, 21), (19, 30)]


def test_plan_snaps_to_longest_nearby_silence_and_absorbs_short_tail():
    sil = [(9.2, 9.4), (10.4, 11.0), (14.0, 15.0)]
    chunks = plan_chunks(21.0, sil, CFG)
    assert [c.lo for c in chunks] == [0.0, 10.7]  # longest pause within +-1 s of 10
    assert chunks[-1].hi == 21.0  # 10.3 s tail <= chunk + window, so no 3rd chunk
    assert len(plan_chunks(10.9, [], CFG)) == 1


def test_parse_silences():
    log = """
[silencedetect @ 0x1] silence_start: 1.5
[silencedetect @ 0x1] silence_end: 2.25 | silence_duration: 0.75
[silencedetect @ 0x1] silence_start: 9.8
"""
    assert parse_silences(log, 10.0) == [(1.5, 2.25), (9.8, 10.0)]


def test_merge_exact_timestamps():
    truth = timeline(30.0)
    chunks = plan_chunks(30.0, [], CFG)
    merged = merge_chunks([(c, asr_view(c, truth)) for c in chunks])
    words = check(merged, truth)
    assert [(w.start, w.end) for w in words] == [(w.start, w.end) for w in truth]


# Time-only matching resolves drift between two chunks' timestamps up to about half a word
# (here 0.25 s words): beyond that a re-seen word and its neighbour are indistinguishable.
@pytest.mark.parametrize("amp", [0.02, 0.05, 0.06])
def test_merge_with_jitter_between_chunks(amp):
    truth = timeline(30.0)
    chunks = plan_chunks(30.0, [], CFG)

    def jitter(c, w):  # each chunk's timestamps drift a different way
        return amp if c.index % 2 else -amp

    check(merge_chunks([(c, asr_view(c, truth, jitter)) for c in chunks]), truth)


@pytest.mark.parametrize("seed", range(20))
def test_merge_with_random_per_word_jitter(seed):
    import random

    rng = random.Random(seed)
    truth = timeline(45.0)
    chunks = plan_chunks(45.0, [], CFG)
    check(merge_chunks([(c, asr_view(c, truth, lambda c, w: rng.uniform(-0.05, 0.05)))
                        for c in chunks]), truth)


def _two_chunks():
    return (Chunk(0, 0.0, 11.0, 0.0, 10.0), Chunk(1, 9.0, 20.0, 10.0, 20.0))


def _seg(words):
    return [Segment(id=0, start=words[0].start, end=words[-1].end,
                    text=" ".join(w.text for w in words), words=words)]


def test_merge_word_seen_on_both_sides_of_cut_is_kept_once():
    a, b = _two_chunks()
    left = [Word(start=9.5, end=9.75, text="before"), Word(start=9.8, end=10.08, text="edge")]
    right = [Word(start=9.88, end=10.16, text="edge"), Word(start=10.3, end=10.6, text="after")]
    merged = merge_chunks([(a, _seg(left)), (b, _seg(right))])
    assert [w.text for s in merged for w in s.words] == ["before", "edge", "after"]
    assert [s.id for s in merged] == [0, 1]


def test_merge_word_neither_side_claims_is_recovered():
    a, b = _two_chunks()
    # Left model puts "edge" just after the cut, right model just before it.
    left = [Word(start=9.5, end=9.75, text="before"), Word(start=9.9, end=10.2, text="edge")]
    right = [Word(start=9.8, end=10.1, text="edge"), Word(start=10.3, end=10.6, text="after")]
    merged = merge_chunks([(a, _seg(left)), (b, _seg(right))])
    words = [w for s in merged for w in s.words]
    assert [w.text for w in words] == ["before", "edge", "after"]
    assert words[1].start >= words[0].end


def test_merge_ignores_overlap_hallucination_far_from_cut():
    a, b = _two_chunks()
    left = [Word(start=9.0, end=9.3, text="real")]
    right = [Word(start=9.05, end=9.2, text="ghost"), Word(start=10.3, end=10.6, text="after")]
    merged = merge_chunks([(a, _seg(left)), (b, _seg(right))])
    assert [w.text for s in merged for w in s.words] == ["real", "after"]


def test_merge_wordless_segments_by_midpoint():
    a, b = _two_chunks()
    s = lambda t0, t1, txt: Segment(id=0, start=t0, end=t1, text=txt)  # noqa: E731
    merged = merge_chunks([(a, [s(1, 2, "x"), s(9.6, 10.6, "y")]),
                           (b, [s(9.62, 10.62, "y"), s(15, 16, "z")])])
    assert [m.text for m in merged] == ["x", "y", "z"]
    assert [m.id for m in merged] == [0, 1, 2]


def _silent_wav(path, seconds, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(seconds * rate))
    return path


@pytest.mark.parametrize("workers", [1, 2])
def test_transcribe_chunked_end_to_end(tmp_path, workers):
    wav = _silent_wav(tmp_path / "a.wav", 31.0)
    truth = timeline(31.0)
    asr = TimelineTranscriber(truth)
    cfg = ChunkConfig(min_duration=0, chunk_seconds=10, overlap=2.0, search_window=1.0,
                      workers=workers)
    lang, segs = transcribe_chunked(asr, wav, None, cfg, tmp_path / "chunks")
    assert lang == "en"
    words = check(segs, truth)
    assert [(w.start, w.end) for w in words] == [(w.start, w.end) for w in truth]
    assert not (tmp_path / "chunks").exists()
    if workers == 1:
        assert len(asr.calls) == 3


def test_cache_key_ignores_workers_only():
    assert ChunkConfig(workers=1).cache_key() == ChunkConfig(workers=8).cache_key()
    assert ChunkConfig(overlap=1).cache_key() != ChunkConfig().cache_key()


@requires_ffmpeg
def test_detect_silences_and_snap_cut(tmp_path):
    from llm_video_editor.ffmpeg.runner import ffmpeg
    from llm_video_editor.stages.chunked_asr import detect_silences

    wav = tmp_path / "tone.wav"
    expr = "if(between(t,9.2,10.0),0,0.5*sin(2*PI*440*t))"
    ffmpeg("-f", "lavfi", "-i", f"aevalsrc='{expr}':s=16000:d=21", "-ac", "1",
           "-c:a", "pcm_s16le", str(wav))
    sil = detect_silences(wav, 21.0, CFG)
    assert len(sil) == 1 and abs(sil[0][0] - 9.2) < 0.05 and abs(sil[0][1] - 10.0) < 0.05
    assert abs(plan_chunks(21.0, sil, CFG)[1].lo - 9.6) < 0.05  # cut mid-pause, not at 10.0
