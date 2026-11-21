import pytest

from llm_video_editor.schemas import (
    EDL,
    ClipInfo,
    ClipsManifest,
    ClipTranscript,
    EDLClip,
    Overlay,
    Transcript,
)
from llm_video_editor.stages.cut_audit import AuditConfig, audit
from tests.fakes import make_segments

# words are 0.25s long on a 0.3s grid: "one." 1.2-1.45, "two." 2.9-3.15, "three." 4.9-5.15
SCRIPT = [(0.0, "This is sentence number one."), (2.0, "Here comes sentence two."),
          (4.0, "And finally sentence three.")]


def _world(duration=10.0):
    a = ClipInfo(clip_id="va", source_file="/raw/a.mp4", duration=duration, width=1920,
                 height=1080, fps=30, has_audio=True)
    b = ClipInfo(clip_id="vb", source_file="/raw/b.mp4", duration=4.0, width=1280, height=720,
                 fps=24, has_audio=False, kind="broll")
    tr = Transcript(clips=[ClipTranscript(clip_id="va", source_file=a.source_file,
                                          duration=duration, segments=make_segments(SCRIPT))])
    return ClipsManifest(batch_hash="x", clips=[a, b]), tr


def _edl(*spans, target=None, overlays=()):
    clips = [EDLClip(clip_id=f"c{i}", source_clip_id=src, source_file=f"/raw/{src[1]}.mp4",
                     start=s, end=e, priority=p)
             for i, (src, s, e, p) in enumerate(spans)]
    return EDL(target_platform="youtube", width=1920, height=1080, target_duration=target,
               clips=clips, overlays=list(overlays))


def kinds(report):
    return [v.kind for v in report.violations]


def test_mid_word_cuts_snap_to_word_edges():
    manifest, tr = _world()
    # 0.4 is inside "is" (0.3-0.55); 1.3 is inside "one." (1.2-1.45)
    edl, rep = audit(_edl(("va", 0.4, 1.3, 1)), tr, manifest)
    assert kinds(rep) == ["mid_word_in", "mid_word_out"]
    c = edl.clips[0]
    assert c.start == pytest.approx(0.25, abs=1e-3)  # lead-in stops at the previous word
    assert c.end == pytest.approx(1.45 + 0.15, abs=1e-3)
    assert rep.passed


def test_mid_sentence_end_is_extended():
    manifest, tr = _world()
    # ends after "comes" (2.3-2.55): sentence ends at "two." (2.9-3.15)
    edl, rep = audit(_edl(("va", 1.95, 2.56, 1)), tr, manifest)
    assert "mid_sentence_out" in kinds(rep)
    assert edl.clips[0].end == pytest.approx(3.15 + 0.15, abs=1e-3)
    assert rep.passed


def test_mid_sentence_trims_back_when_extension_too_far():
    manifest, tr = _world()
    edl, rep = audit(_edl(("va", 0.0, 2.56, 1)), tr, manifest, cfg=AuditConfig(max_extend=0.1))
    v = [v for v in rep.violations if v.kind == "mid_sentence_out"][0]
    assert v.fixed and "trimmed back" in v.fix
    assert edl.clips[0].end == pytest.approx(1.45 + 0.15, abs=1e-3)


def test_out_of_bounds_clamped_per_source_clip():
    manifest, tr = _world(duration=5.0)
    edl, rep = audit(_edl(("va", 3.9, 9.0, 1), ("vb", 1.0, 30.0, 1)), tr, manifest)
    oob = [v for v in rep.violations if v.kind == "out_of_bounds"]
    assert [v.source_file for v in oob] == ["/raw/a.mp4", "/raw/b.mp4"]
    assert edl.clips[0].end <= 5.0 and edl.clips[1].end == 4.0  # b clamped to its own 4s


def test_overlap_merge_and_too_short():
    manifest, tr = _world()
    edl, rep = audit(_edl(("va", 0.0, 1.9, 1), ("va", 1.0, 3.3, 2), ("vb", 0.0, 0.3, 1)),
                     tr, manifest)
    assert "overlap" in kinds(rep) and "too_short" in kinds(rep)
    assert len(edl.clips) == 1 and edl.clips[0].end == pytest.approx(3.3, abs=1e-3)


def test_duration_budget_drops_lowest_priority():
    manifest, tr = _world()
    edl, rep = audit(_edl(("va", 0.0, 1.9, 1), ("va", 2.0, 3.3, 3), ("va", 4.0, 5.4, 2),
                          target=3.4), tr, manifest)
    dropped = [v for v in rep.violations if v.kind == "duration_budget"]
    assert dropped[0].fix == "dropped priority-3 clip"
    assert [c.clip_id for c in edl.clips] == ["c0", "c2"]
    assert rep.passed


def test_unknown_source_and_unfixable_budget_fail():
    manifest, tr = _world()
    edl, rep = audit(_edl(("vz", 0, 2, 1), ("va", 0.0, 5.4, 1), target=2.0), tr, manifest)
    assert kinds(rep)[0] == "unknown_source"
    assert not rep.passed and rep.unresolved[0].kind == "duration_budget"


def test_overlays_follow_their_anchor():
    manifest, tr = _world()
    ov = Overlay(overlay_id="o1", source_clip_id="vb", source_file="/raw/b.mp4", src_start=3.5,
                 anchor_clip_id="c1", offset=0.2, duration=9)
    edl, rep = audit(_edl(("va", 0.0, 1.9, 1), ("va", 2.0, 3.3, 1), overlays=[ov]), tr, manifest)
    o = edl.overlays[0]
    assert o.duration <= edl.clips[1].duration and o.src_start + o.duration <= 4.0
    assert edl.overlay_start(o) == pytest.approx(edl.clips[0].duration + 0.2)
    edl2, rep2 = audit(_edl(("va", 0.0, 1.9, 1), overlays=[ov]), tr, manifest)
    assert edl2.overlays == [] and "overlay_out_of_bounds" in kinds(rep2)
