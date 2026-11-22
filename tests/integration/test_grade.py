"""Colour: one restrained look per video plus per-clip correction for consistency;
measured, suggested and reported, applied in render, review and thumbnails."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from llm_video_editor.agent.tools import Context, call
from llm_video_editor.ffmpeg.runner import ffmpeg, ffmpeg_bytes
from llm_video_editor.schemas import EDL, ClipColor, EDLClip, Grade
from llm_video_editor.stages import grade as gr
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeTranscriber
from tests.fixtures.make_fixture import make_clip

pytestmark = [requires_ffmpeg, pytest.mark.integration]


def _apply(img: Path, chain: str) -> np.ndarray:
    raw = ffmpeg_bytes("-i", str(img), "-vf", f"{chain + ',' if chain else ''}format=rgb24",
                       "-f", "rawvideo", "-")
    return np.frombuffer(raw, np.uint8).reshape(-1, 320, 3).astype(np.float32) / 255


@pytest.fixture(scope="module")
def frame(tmp_path_factory):
    p = tmp_path_factory.mktemp("g") / "scene.png"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x180:d=1", "-frames:v", "1", str(p))
    return p


def test_every_look_is_a_valid_gentle_filter(frame):
    base = _apply(frame, "")
    for look in gr.LOOKS:
        f = gr.look_filter(Grade(look=look, strength=0.5))
        out = _apply(frame, f)
        assert out.shape == base.shape, look
        diff = float(np.abs(out - base).mean())
        if look == "none":
            assert f == "" and diff == 0
        elif look != "bw":
            assert 0 < diff < 0.08, (look, diff)  # visible but restrained at 0.5
    assert gr.look_filter(Grade(look="warm", strength=0.0)) == ""


def test_warm_and_cool_go_the_right_way_and_strength_scales(frame):
    def warmth(chain):
        x = _apply(frame, chain)
        return float(x[..., 0].mean() - x[..., 2].mean())

    base = warmth("")
    half = warmth(gr.look_filter(Grade(look="warm", strength=0.5)))
    full = warmth(gr.look_filter(Grade(look="warm", strength=1.0)))
    assert base < half < full
    _, cool = gr.correction_filter(ClipColor(warmth=-0.8))
    assert warmth(cool) < base
    _, brighter = gr.correction_filter(ClipColor(exposure=0.7))
    assert _apply(frame, brighter).mean() > _apply(frame, "").mean()
    pre, _ = gr.correction_filter(ClipColor(tonemap=True))
    assert pre.startswith("zscale")


def test_matching_suggests_gentle_corrections_and_warns_on_dark_clips():
    m = {"a": {"mid": 0.40, "saturation": 0.30, "warm": 0.02, "green": 0.0,
               "clipped_high": 0.0, "clipped_low": 0.0},
         "b": {"mid": 0.42, "saturation": 0.31, "warm": 0.03, "green": 0.0,
               "clipped_high": 0.0, "clipped_low": 0.0},
         "dark": {"mid": 0.06, "saturation": 0.20, "warm": -0.06, "green": 0.0,
                  "clipped_high": 0.0, "clipped_low": 0.3}}
    s = gr.suggest_matches(m)
    assert s["a"]["suggested"] is None  # already matches
    d = s["dark"]
    assert d["suggested"]["exposure"] == 0.7 and "noise" in d["notes"][0]
    assert d["suggested"]["warmth"] > 0 and d["suggested"]["saturation"] <= 1.25


def test_warnings_for_overdone_colour():
    def clip(i, **c):
        return EDLClip(clip_id=f"c{i}", source_clip_id="s", source_file="/x", start=0, end=1,
                       color=ClipColor(**c))

    edl = EDL(target_platform="youtube", width=1920, height=1080,
              grade=Grade(look="vivid", strength=0.9),
              clips=[clip(1, exposure=1.5), clip(2, saturation=1.5), clip(3, saturation=0.9)])
    w = gr.color_warnings(edl)
    assert any("big push" in x for x in w) and any("unnatural" in x for x in w)
    assert any("differ a lot" in x for x in w) and any("strong" in x for x in w)


def ok(res):
    assert "error" not in res, res
    return res


def test_agent_colour_tools_and_graded_render(tmp_path):
    raw = tmp_path / "trip"
    make_clip(raw / "a.mp4", duration=4, size=(640, 360))
    make_clip(raw / "b.mp4", duration=4, size=(640, 360))
    dark = raw / "c_dark.mp4"
    ffmpeg("-i", str(raw / "b.mp4"), "-vf", "eq=brightness=-0.35", "-c:a", "copy", str(dark))
    c = Context(home=tmp_path / "home", transcriber_factory=lambda m: FakeTranscriber())
    ws = ok(call("open_project", {"inputs": [str(raw)]}, c))["workspace"]
    ids = {Path(x["file"]).stem: x["clip_id"] for x in ok(call("analyze", {"workspace": ws},
                                                              c))["clips"]}
    ok(call("new_edit", {"workspace": ws, "target_platform": "youtube", "reason": "start"}, c))
    ok(call("batch", {"workspace": ws, "reason": "three shots", "operations": [
        {"op": "add_clip", "params": {"source_clip_id": ids[k], "start": 0.2, "end": 3.5,
                                      "role": "broll"}} for k in ("a", "b", "c_dark")]}, c))
    rep = ok(call("color_report", {"workspace": ws}, c))
    by = {x["clip_id"]: x for x in rep["clips"]}
    assert by["c003"]["mid"] < by["c001"]["mid"] and by["c003"]["suggested"]["exposure"] > 0
    assert rep["spread"]["mid"] > 0.02

    ok(call("batch", {"workspace": ws, "reason": "match the dark clip, warm look",
                      "operations": [
                          {"op": "set_color", "params": {"clip_id": "c003",
                                                         **by["c003"]["suggested"]}},
                          {"op": "set_grade", "params": {"look": "warm", "strength": 0.5}}]}, c))
    st = ok(call("edit_state", {"workspace": ws}, c))
    assert st["version"] == 3
    edl = EDL.model_validate_json((Path(ws) / "edl_raw.json").read_text())
    assert edl.grade.look == "warm" and edl.clips[2].color.exposure > 0
    assert ok(call("set_color", {"workspace": ws, "clip_id": "c003", "fields": {},
                                 "reason": "back to the original"}, c))["version"] == 4
    assert ok(call("audit", {"workspace": ws}, c))["color"] == "no colour issues"

    from llm_video_editor.stages.render import render
    from llm_video_editor.workspace import Workspace
    w = Workspace.load(Path(ws))
    ungraded = edl.model_copy(update={"grade": Grade(), "clips": [
        c.model_copy(update={"color": None}) for c in edl.clips]})
    plain = render(w, ungraded, quality="draft",
                   burn_captions=False, out=tmp_path / "out" / "plain.mp4")
    warm = render(w, edl, quality="draft", burn_captions=False,
                  out=tmp_path / "out" / "warm.mp4")
    assert plain.qc.passed and warm.qc.passed

    def rb(path, t):
        x = np.frombuffer(ffmpeg_bytes("-ss", str(t), "-i", str(path), "-frames:v", "1",
                                       "-vf", "scale=160:90,format=rgb24", "-f", "rawvideo",
                                       "-"), np.uint8).reshape(90, 160, 3).astype(float)
        return x[..., 0].mean() - x[..., 2].mean(), x.mean()

    assert rb(warm.output, 1.0)[0] > rb(plain.output, 1.0)[0] + 2  # warmer
    assert rb(warm.output, 8.0)[1] > rb(plain.output, 8.0)[1] + 5  # dark clip lifted
