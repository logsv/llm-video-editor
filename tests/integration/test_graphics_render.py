"""Motion graphics through render and review.

The fake graphic is opaque red in its top-left quarter and *transparent green*
everywhere else, over a black clip: correct alpha shows black there, a decoder
that drops alpha (ffmpeg's native VP9) would show green.

``test_real_hyperframes_*`` runs the actual HyperFrames CLI and is skipped unless
the pinned toolchain is installed (``llmve graphics doctor --install``).
"""

from __future__ import annotations

import subprocess

import pytest
from typer.testing import CliRunner

from llm_video_editor import hyperframes as hf_pkg
from llm_video_editor.cli.app import app
from llm_video_editor.ffmpeg.runner import ffmpeg, ffprobe_json
from llm_video_editor.hyperframes import toolchain
from llm_video_editor.hyperframes.compose import has_alpha
from llm_video_editor.schemas import EDL, EDLClip, Graphic
from llm_video_editor.stages.render import render
from llm_video_editor.stages.review import build_review
from llm_video_editor.stages.watch import watch
from llm_video_editor.workspace import Workspace
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeTranscriber
from tests.fixtures.make_fixture import make_clip

pytestmark = [requires_ffmpeg, pytest.mark.integration]


def fake_graphic(path, w, h, duration):
    alpha = f"if(lt(X,{w // 2})*lt(Y,{h // 2}),255,0)"
    ffmpeg("-f", "lavfi", "-i", f"color=c=green:s={w}x{h}:r=30:d={duration}", "-vf",
           f"format=yuva420p,geq=r='if(lt(X,{w // 2})*lt(Y,{h // 2}),255,0)':"
           f"g='if(lt(X,{w // 2})*lt(Y,{h // 2}),0,255)':b=0:a='{alpha}'",
           "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0", str(path),
           timeout=120)
    return path


@pytest.fixture
def black_ws(tmp_path):
    raw = tmp_path / "raw"
    make_clip(raw / "black.mp4", duration=4, size=(640, 360), fps=30, markers=True)
    ws = Workspace.open(raw, home=tmp_path / "home")
    info = watch(ws, transcriber=FakeTranscriber()).manifest.clips[0]
    return ws, info


@pytest.fixture
def fake_hyperframes(monkeypatch, tmp_path):
    calls = []

    def render_graphic(cache_dir, *, template=None, composition=None, variables=None, width,
                       height, duration, fps=30, quality="standard"):
        calls.append((template, variables, width, height, duration))
        out = cache_dir / f"fake-{len(calls)}.webm"
        return fake_graphic(out, width, height, duration)

    monkeypatch.setattr(hf_pkg, "render_graphic", render_graphic)
    return calls


def rgb_at(path, t, x, y):
    """One pixel as (r, g, b); binary stdout, so not through the text-mode runner."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t}", "-i", str(path),
                          "-frames:v", "1", "-vf", f"crop=1:1:{x}:{y},format=rgb24",
                          "-f", "rawvideo", "-"], capture_output=True, check=True,
                         timeout=60).stdout
    return tuple(raw[:3])


def _edl(info, graphics):
    return EDL(target_platform="youtube", width=640, height=360, captions={"enabled": False},
               clips=[EDLClip(clip_id="c1", source_clip_id=info.clip_id,
                              source_file=info.source_file, start=0.0, end=3.0)],
               graphics=graphics)


def test_fake_graphic_really_has_alpha(tmp_path):
    assert has_alpha(fake_graphic(tmp_path / "g.webm", 64, 36, 1))


def test_render_alpha_composites_graphics(black_ws, fake_hyperframes):
    ws, info = black_ws
    edl = _edl(info, [Graphic(graphic_id="g1", template="lower-third", anchor_clip_id="c1",
                              offset=0.2, duration=2.0, variables={"title": "Hi"})])
    res = render(ws, edl, quality="draft", burn_captions=False)
    assert res.qc.passed, res.qc.problems
    assert fake_hyperframes[0][:4] == ("lower-third", {"title": "Hi"}, 640, 360)
    # t=1.2s: graphic on, marker flash (0.5-0.6s of each second) off
    r, g, b = rgb_at(res.output, 1.2, 40, 40)
    assert r > 180 and g < 80, (r, g, b)  # opaque red quarter
    r, g, b = rgb_at(res.output, 1.2, 600, 330)
    assert g < 60, (r, g, b)  # transparent area shows the black clip, not green
    r, g, b = rgb_at(res.output, 2.8, 40, 40)
    assert r < 60, (r, g, b)  # graphic over after 2.2s


def test_graphic_box_places_and_scales(black_ws, fake_hyperframes):
    ws, info = black_ws
    edl = _edl(info, [Graphic(graphic_id="g1", template="callout", anchor_clip_id="c1",
                              duration=2.0, box=(0.5, 0.5, 0.5, 0.5))])
    res = render(ws, edl, quality="draft", burn_captions=False)
    assert fake_hyperframes[0][2:4] == (320, 180)  # rendered at box size
    assert rgb_at(res.output, 1.2, 340, 200)[0] > 180  # red quarter of the box
    assert rgb_at(res.output, 1.2, 40, 40)[0] < 60  # outside the box: untouched


def test_review_shows_graphic_tiles(black_ws, fake_hyperframes):
    ws, info = black_ws
    edl = _edl(info, [Graphic(graphic_id="g1", template="title-card", anchor_clip_id="c1",
                              duration=2.0)])
    review = build_review(ws, edl)
    gtiles = [t for t in review.tiles if t.graphic_id == "g1"]
    assert len(gtiles) == 1 and gtiles[0].clip_id == "c1"
    assert not review.warnings or all("graphic" not in w for w in review.warnings)
    # the pixels, not just the metadata: the red quarter is on the graphic tile ...
    tile = ws.review_dir / f"tile_{gtiles[0].index:03d}.png"
    r, g, b = rgb_at(tile, 0, 20, 20)
    assert r > 180 and g < 80, (r, g, b)
    assert rgb_at(tile, 0, 300, 160)[1] < 60  # ... and the transparent part stays see-through
    # ... and the sheet keeps the clip tile too (mixed pixel formats used to drop tiles)
    sheet = ws.root / review.sheets[0]
    assert len(review.tiles) == 2
    tw = ffprobe_json(tile)["streams"][0]["width"]
    r, g, b = rgb_at(sheet, 0, 6 + 20, 6 + 20)  # clip tile: black or the white marker flash
    assert abs(r - g) < 30 and abs(g - b) < 30, ("slot 0 should be the clip tile", (r, g, b))
    r, g, b = rgb_at(sheet, 0, 6 + tw + 6 + 20, 6 + 20)
    assert r > 180 and g < 80, ("slot 1 should be the graphic tile", (r, g, b))


def test_review_warns_when_hyperframes_is_missing(black_ws, monkeypatch):
    ws, info = black_ws

    def unavailable(*a, **k):
        raise hf_pkg.HyperFramesUnavailable("HyperFrames needs Node.js 22+")

    monkeypatch.setattr(hf_pkg, "render_graphic", unavailable)
    edl = _edl(info, [Graphic(graphic_id="g1", template="callout", anchor_clip_id="c1",
                              duration=1.0)])
    review = build_review(ws, edl)
    assert any("g1 could not be rendered" in w for w in review.warnings)


def test_cli_add_and_chapter_cards(black_ws):
    ws, info = black_ws
    edl = _edl(info, []).model_copy(update={"clips": [
        EDLClip(clip_id="c1", source_clip_id=info.clip_id, source_file=info.source_file,
                start=0, end=1.5, chapter="Intro"),
        EDLClip(clip_id="c2", source_clip_id=info.clip_id, source_file=info.source_file,
                start=1.5, end=3, chapter="Setup")]})
    Workspace.write(ws.edl_raw, edl)
    cli = CliRunner()
    added = cli.invoke(app, ["graphics", "add", str(ws.root), "--at", "c2", "-t", "lower-third",
                             "--var", "title=Vikas", "--var", "subtitle=Editor",
                             "--offset", "0.2", "--duration", "1"])
    assert added.exit_code == 0, added.output
    cards = cli.invoke(app, ["graphics", "chapters", str(ws.root), "--duration", "1"])
    assert cards.exit_code == 0, cards.output
    graphics = Workspace.read(ws.edl_raw, EDL).graphics
    assert [(g.graphic_id, g.template) for g in graphics] == [
        ("g001", "lower-third"), ("ch01", "title-card"), ("ch02", "title-card")]
    assert graphics[2].variables == {"title": "Setup", "kicker": "Chapter 2"}
    bad = cli.invoke(app, ["graphics", "add", str(ws.root), "--at", "c9", "-t", "callout"])
    assert bad.exit_code != 0


real = pytest.mark.skipif(not (toolchain.toolchain().installed
                               and (toolchain.node_major() or 0) >= 22),
                          reason="HyperFrames toolchain not installed")


@real
@pytest.mark.hyperframes
def test_real_hyperframes_template_renders_with_alpha(tmp_path):
    from llm_video_editor.hyperframes import render_graphic

    out = render_graphic(tmp_path, template="lower-third",
                         variables={"title": "Real", "subtitle": "HyperFrames"},
                         width=640, height=360, duration=1.5, quality="draft")
    assert has_alpha(out)
    again = render_graphic(tmp_path, template="lower-third",
                           variables={"title": "Real", "subtitle": "HyperFrames"},
                           width=640, height=360, duration=1.5, quality="draft")
    assert again == out  # cached


@real
@pytest.mark.hyperframes
def test_real_hyperframes_end_to_end_render(black_ws):
    ws, info = black_ws
    edl = _edl(info, [Graphic(graphic_id="g1", template="callout", anchor_clip_id="c1",
                              offset=0.3, duration=2.0,
                              variables={"text": "HELLO", "position": "center"})])
    res = render(ws, edl, quality="draft", burn_captions=False)
    assert res.qc.passed, res.qc.problems
    r, g, b = rgb_at(res.output, 1.2, 320, 180)  # centre of the white pill
    assert min(r, g, b) > 150, (r, g, b)
    assert max(rgb_at(res.output, 1.2, 10, 350)) < 60  # corner: still the black clip
