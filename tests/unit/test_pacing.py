import pytest

from llm_video_editor.schemas import (
    EDL,
    ClipInfo,
    ClipsManifest,
    ClipTranscript,
    EDLClip,
    Overlay,
    Pacing,
    Transcript,
    Word,
)
from llm_video_editor.stages.cut_audit import audit
from llm_video_editor.stages.pacing import speech_islands
from tests.fakes import make_segments


def W(s, e, t="w"):
    return Word(start=s, end=e, text=t)


def test_islands_remove_only_long_pauses():
    words = [W(1.0, 1.4), W(1.6, 2.0), W(4.0, 4.5)]  # 0.2s gap kept, 2.0s gap removed
    p = Pacing(tighten=True, max_gap=0.6, lead=0.1, tail=0.1)
    assert speech_islands(0, 10, words, p) == [(0.9, 2.1), (3.9, 4.6)]


def test_span_without_words_is_kept_whole():
    assert speech_islands(5, 9, [W(1, 2)], Pacing(tighten=True)) == [(5, 9)]


def test_fillers_become_pauses_when_asked():
    words = [W(1.0, 1.4, "So"), W(1.9, 2.3, "um,"), W(2.8, 3.2, "right.")]
    assert len(speech_islands(0, 5, words, Pacing(tighten=True, drop_fillers=True))) == 2
    assert len(speech_islands(0, 5, words, Pacing(tighten=True))) == 1


# "one." ends 1.45; "two." sentence starts 4.0 -> a 2.5s pause inside one audited clip
SCRIPT = [(0.0, "This is sentence number one."), (4.0, "Here comes sentence two."),
          (8.0, "And finally sentence three.")]


def _world():
    info = ClipInfo(clip_id="va", source_file="/raw/a.mp4", duration=12.0, width=1920,
                    height=1080, fps=30, has_audio=True)
    broll = ClipInfo(clip_id="vb", source_file="/raw/b.mp4", duration=5.0, width=1920,
                     height=1080, fps=30, has_audio=False, kind="broll")
    tr = Transcript(clips=[ClipTranscript(clip_id="va", source_file=info.source_file,
                                          duration=12.0, segments=make_segments(SCRIPT))])
    return ClipsManifest(batch_hash="x", clips=[info, broll]), tr


def _edl(pacing, overlays=()):
    return EDL(target_platform="youtube", width=1920, height=1080, pacing=pacing,
               clips=[EDLClip(clip_id="c001", source_clip_id="va", source_file="/raw/a.mp4",
                              start=0.0, end=9.6, chapter="Intro", priority=2)],
               overlays=list(overlays))


def test_cut_audit_tightens_and_alternates_punch_in():
    manifest, tr = _world()
    edl, rep = audit(_edl(Pacing(tighten=True, punch_in="alternate", punch_zoom=1.15)),
                     tr, manifest)
    assert rep.passed
    assert [c.clip_id for c in edl.clips] == ["c001", "c001.2", "c001.3"]
    assert [c.zoom for c in edl.clips] == [1.0, 1.15, 1.0]
    assert [c.chapter for c in edl.clips] == ["Intro", None, None]
    assert {c.priority for c in edl.clips} == {2}
    assert rep.pauses_removed > 4.0
    assert edl.total_duration == pytest.approx(rep.duration_after)


def test_pacing_off_leaves_cut_alone():
    manifest, tr = _world()
    edl, rep = audit(_edl(Pacing()), tr, manifest)
    assert len(edl.clips) == 1 and rep.pauses_removed == 0


def test_overlay_follows_the_piece_that_plays_its_moment():
    manifest, tr = _world()
    ov = Overlay(overlay_id="o1", source_clip_id="vb", source_file="/raw/b.mp4",
                 anchor_clip_id="c001", offset=4.5, duration=1.0)  # source 4.5s: sentence two
    edl, rep = audit(_edl(Pacing(tighten=True), [ov]), tr, manifest)
    (moved,) = edl.overlays
    assert moved.anchor_clip_id == "c001.2"
    piece = next(c for c in edl.clips if c.clip_id == "c001.2")
    assert piece.start + moved.offset == pytest.approx(4.5, abs=1e-3)


def test_cold_open_replay_is_kept():
    manifest, tr = _world()
    hook = EDLClip(clip_id="c000", source_clip_id="va", source_file="/raw/a.mp4",
                   start=7.95, end=9.6, chapter="Hook")
    edl = _edl(Pacing())
    edl = edl.model_copy(update={"clips": [hook, *edl.clips]})
    out, rep = audit(edl, tr, manifest)
    assert [(c.clip_id, c.start) for c in out.clips][1] == ("c001", 0.0)  # main cut untouched
    (v,) = [v for v in rep.violations if v.kind == "overlap"]
    assert v.severity == "warning" and "cold open" in v.message and rep.passed


def test_budget_counts_the_tightened_length_not_the_pauses():
    # one 9.6 s clip that holds ~7 s of speech once its two long pauses are removed
    manifest, tr = _world()

    def edl(pacing):
        return EDL(target_platform="youtube", width=1920, height=1080, pacing=pacing,
                   target_duration=9.0, clips=[
                       EDLClip(clip_id=f"c{i}", source_clip_id="va", source_file="/raw/a.mp4",
                               start=0.0, end=9.6) for i in range(1)])

    cut, rep = audit(edl(Pacing()), tr, manifest)
    assert "duration_budget" not in [v.kind for v in rep.violations if v.fixed]  # 9.6 s ~ 9 s
    cut, rep = audit(edl(Pacing(tighten=True)), tr, manifest)
    assert rep.passed and len(cut.clips) == 3  # tightened, nothing dropped for the pauses
    assert cut.total_duration < 9.0 * 1.05
