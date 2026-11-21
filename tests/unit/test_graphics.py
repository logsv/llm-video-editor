import json

import pytest
from pydantic import ValidationError

from llm_video_editor.ffmpeg.graph import Graph, GraphicSpec, add_graphics
from llm_video_editor.hyperframes import compose, toolchain
from llm_video_editor.schemas import (
    EDL,
    ClipInfo,
    ClipsManifest,
    ClipTranscript,
    EDLClip,
    Graphic,
    Pacing,
    Transcript,
)
from llm_video_editor.stages.cut_audit import audit
from tests.fakes import make_segments


def test_graphic_needs_exactly_one_source():
    with pytest.raises(ValidationError):
        Graphic(graphic_id="g1", anchor_clip_id="c1", duration=2)
    with pytest.raises(ValidationError):
        Graphic(graphic_id="g1", template="callout", composition="/x", anchor_clip_id="c1",
                duration=2)
    Graphic(graphic_id="g1", template="callout", anchor_clip_id="c1", duration=2)


def test_builtin_templates_are_filled_for_any_frame():
    t = compose.templates()
    assert {"lower-third", "title-card", "callout"} <= set(t)
    assert {v["id"] for v in t["lower-third"].variables} >= {"title", "subtitle", "accent"}
    for info in t.values():
        html = compose.fill_template(info.path.read_text(), 1080, 1920, 3.5)
        assert not any(p in html for p in ("__W__", "__H__", "__D__", "__U__")), info.name
        assert 'data-width="1080"' in html and 'data-duration="3.500"' in html
        assert 'window.__timelines["main"]' in html and "assets/gsap.min.js" in html


def test_prepare_project_copies_gsap_and_never_touches_the_source(tmp_path):
    gsap = tmp_path / "gsap.min.js"
    gsap.write_text("/* gsap */")
    src = tmp_path / "mycomp"
    src.mkdir()
    (src / "index.html").write_text("<html></html>")
    (src / "node_modules").mkdir()
    out = compose.prepare_project(tmp_path / "out", template=None, composition=src,
                                  width=1920, height=1080, duration=2, gsap=gsap)
    assert (out / "assets" / "gsap.min.js").read_text() == "/* gsap */"
    assert not (out / "node_modules").exists() and not (src / "assets").exists()
    with pytest.raises(compose.GraphicError):
        compose.prepare_project(tmp_path / "o2", template="nope", composition=None,
                                width=10, height=10, duration=1, gsap=None)


def test_cache_key_changes_with_inputs():
    a = compose.cache_key("t", {"title": "A"}, 1920, 1080, 3, 30)
    assert a == compose.cache_key("t", {"title": "A"}, 1920, 1080, 3, 30)
    assert a != compose.cache_key("t", {"title": "B"}, 1920, 1080, 3, 30)
    assert a != compose.cache_key("t", {"title": "A"}, 1080, 1920, 3, 30)


def test_lint_errors_are_readable():
    out = json.dumps({"ok": False, "findings": [
        {"code": "missing_timeline_registry", "severity": "error", "message": "Missing x",
         "fixHint": "Register it"},
        {"code": "w", "severity": "warning", "message": "meh"}]})
    assert compose._lint_errors("noise\n" + out) == [
        "- missing_timeline_registry: Missing x (fix: Register it)"]
    assert compose._lint_errors("not json") == []


def test_graphics_are_decoded_with_libvpx_to_keep_alpha():
    g = Graph()
    label = add_graphics(g, "base", [GraphicSpec("/g/a.webm", 2.0, 3.0),
                                     GraphicSpec("/g/b.webm", 5.0, 1.0, (0.5, 0.0, 0.5, 0.25))],
                         width=1920, height=1080, fps=30)
    assert g.input_args[:4] == ["-c:v", "libvpx-vp9", "-i", "/g/a.webm"]
    fc = g.filter_complex
    assert "format=yuva420p" in fc and "overlay=x=960:y=0:format=auto" in fc
    assert "scale=960:270" in fc and "between(t,2.000000,4.999900)" in fc
    assert label == "vg1"


def test_toolchain_refuses_old_node_and_keeps_telemetry_off(monkeypatch):
    monkeypatch.setattr(toolchain, "node_major", lambda: 20)
    with pytest.raises(toolchain.HyperFramesUnavailable, match="Node.js 22"):
        toolchain.ensure()
    env = toolchain.env()
    assert env["HYPERFRAMES_NO_TELEMETRY"] == "1" and env["DO_NOT_TRACK"] == "1"
    assert env["npm_config_onnxruntime_node_install_cuda"] == "skip"


SCRIPT = [(0.0, "This is sentence number one."), (4.0, "Here comes sentence two.")]


def _world():
    info = ClipInfo(clip_id="va", source_file="/raw/a.mp4", duration=8.0, width=1920,
                    height=1080, fps=30, has_audio=True)
    tr = Transcript(clips=[ClipTranscript(clip_id="va", source_file=info.source_file,
                                          duration=8.0, segments=make_segments(SCRIPT))])
    return ClipsManifest(batch_hash="x", clips=[info]), tr


def _edl(graphics, pacing=None):
    return EDL(target_platform="youtube", width=1920, height=1080, pacing=pacing or Pacing(),
               clips=[EDLClip(clip_id="c001", source_clip_id="va", source_file="/raw/a.mp4",
                              start=0.0, end=5.4)], graphics=graphics)


def G(gid, **kw):
    kw.setdefault("template", "lower-third")
    return Graphic(graphic_id=gid, anchor_clip_id=kw.pop("anchor", "c001"),
                   duration=kw.pop("duration", 2.0), **kw)


def test_cut_audit_validates_graphics(tmp_path):
    manifest, tr = _world()
    edl, rep = audit(_edl([G("g1"), G("g2", template="nope"), G("g3", anchor="c999"),
                           G("g4", offset=4.0, duration=10.0),
                           G("g5", template=None, composition=str(tmp_path))]), tr, manifest)
    assert [g.graphic_id for g in edl.graphics] == ["g1", "g4"]
    assert edl.graphics[1].duration == pytest.approx(edl.total_duration - 4.0, abs=1e-3)
    kinds = sorted(v.kind for v in rep.violations)
    assert kinds.count("unknown_graphic") == 2 and kinds.count("graphic_out_of_bounds") == 2
    assert rep.passed


def test_graphics_follow_tightened_pieces():
    manifest, tr = _world()
    edl, _ = audit(_edl([G("g1", offset=4.2)], Pacing(tighten=True)), tr, manifest)
    (g,) = edl.graphics
    piece = next(c for c in edl.clips if c.clip_id == g.anchor_clip_id)
    assert g.anchor_clip_id == "c001.2" and piece.start + g.offset == pytest.approx(4.2, abs=1e-3)
