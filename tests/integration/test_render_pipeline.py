"""End-to-end render checks on synthetic multi-camera media.

* A/V sync: every clip carries a white flash and a 1 kHz beep that start
  together; after cutting, re-timing (30/60 fps -> 30 fps), cover-cropping to
  9:16 and loudnorm, every beep must still line up with its flash.
* Loudness/format: mixed 1080p30 / 720p60 / silent 720p24 sources with a
  B-roll overlay come out 1080x1920 @ 30 fps, 48 kHz stereo, -14 LUFS.
"""

from __future__ import annotations

import re

import pytest

from llm_video_editor.ffmpeg.runner import ffmpeg
from llm_video_editor.schemas import EDL, ClipsManifest, EDLClip, Overlay
from llm_video_editor.stages import render as render_mod
from llm_video_editor.stages.render import render
from llm_video_editor.stages.watch import watch
from llm_video_editor.workspace import Workspace
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeTranscriber
from tests.fixtures.make_fixture import make_clip

pytestmark = [requires_ffmpeg, pytest.mark.integration]
FRAME = 1 / 30


def _watched(folder, tmp_path, asr=None):
    ws = Workspace.open(folder, home=tmp_path / "home")
    res = watch(ws, transcriber=asr or FakeTranscriber())
    return ws, {c.source_file.rsplit("/", 1)[-1]: c for c in res.manifest.clips}


def _clip(cid, info, start, end, role="a_roll"):
    return EDLClip(clip_id=cid, source_clip_id=info.clip_id, source_file=info.source_file,
                   start=start, end=end, role=role)


def flash_onsets(path) -> list[float]:
    proc = ffmpeg("-i", str(path), "-an", "-vf",
                  "signalstats,metadata=print:key=lavfi.signalstats.YAVG", "-f", "null", "-")
    times, vals, t = [], [], None
    for line in proc.stderr.splitlines():
        if m := re.search(r"pts_time:([0-9.]+)", line):
            t = float(m.group(1))
        elif (m := re.search(r"YAVG=([0-9.]+)", line)) and t is not None:
            times.append(t)
            vals.append(float(m.group(1)))
    return [times[i] for i in range(1, len(vals)) if vals[i] > 128 >= vals[i - 1]]


def beep_onsets(path, duration: float) -> list[float]:
    proc = ffmpeg("-i", str(path), "-vn", "-af", "silencedetect=n=-35dB:d=0.04", "-f", "null", "-")
    ends = [float(x) for x in re.findall(r"silence_end: ([0-9.]+)", proc.stderr)]
    return [t for t in ends if t < duration - 0.05]  # silencedetect also reports EOF


def test_av_sync_survives_cut_retime_and_concat(tmp_path):
    raw = tmp_path / "sync"
    make_clip(raw / "cam_a.mp4", duration=4, size=(1920, 1080), fps=30, audio="beeps",
              markers=True)
    make_clip(raw / "cam_b.mov", duration=4, size=(1280, 720), fps=60, audio="beeps",
              markers=True, sample_rate=48000)
    ws, m = _watched(raw, tmp_path)
    a, b = m["cam_a.mp4"], m["cam_b.mov"]
    # cut points deliberately off the frame grid of both sources
    edl = EDL(target_platform="reels", width=1080, height=1920, clips=[
        _clip("c1", a, 0.37, 2.21), _clip("c2", b, 1.113, 3.29), _clip("c3", a, 2.71, 3.95)])
    res = render(ws, edl, quality="draft", burn_captions=False)
    assert res.qc.av_offset <= FRAME, res.qc

    flashes, beeps = flash_onsets(res.output), beep_onsets(res.output, res.qc.duration)
    assert len(beeps) >= 4, beeps
    for t in beeps:
        nearest = min(flashes, key=lambda f: abs(f - t))
        assert abs(nearest - t) <= FRAME + 0.005, (t, nearest, flashes)


def test_mixed_sources_render_to_spec_at_minus_14_lufs(media_dir, tmp_path):
    ws, m = _watched(media_dir, tmp_path)
    a, b, c = m["a_talk_1080p30.mp4"], m["b_talk_720p60.mov"], m["c_broll_720p24.mp4"]
    edl = EDL(target_platform="reels", width=1080, height=1920, clips=[
        _clip("c1", a, 0.2, 1.95), _clip("c2", b, 2.13, 3.7), _clip("c3", c, 0.5, 2.0, "broll")],
        overlays=[Overlay(overlay_id="o1", source_clip_id=c.clip_id, source_file=c.source_file,
                          anchor_clip_id="c2", offset=0.3, duration=1.0)])
    res = render(ws, edl, quality="draft")
    qc = res.qc
    assert qc.passed, qc.problems
    assert (qc.width, qc.height, qc.fps) == (1080, 1920, 30.0)
    assert (qc.sample_rate, qc.channels) == (48000, 2)
    assert qc.integrated_lufs == pytest.approx(-14.0, abs=1.0)
    assert qc.true_peak <= -1.0
    assert qc.duration == pytest.approx(qc.expected_duration, abs=FRAME)
    assert qc.av_offset <= FRAME
    assert (ws.render_dir / "captions.ass").exists()


def test_long_edits_render_through_mezzanine_groups(media_dir, tmp_path, monkeypatch):
    ws, m = _watched(media_dir, tmp_path)
    a, b, c = m["a_talk_1080p30.mp4"], m["b_talk_720p60.mov"], m["c_broll_720p24.mp4"]
    edl = EDL(target_platform="youtube", width=1920, height=1080, fit_mode="contain", clips=[
        _clip("c1", a, 0.0, 1.0), _clip("c2", b, 0.5, 1.5), _clip("c3", c, 0.0, 1.0, "broll"),
        _clip("c4", a, 2.0, 3.0), _clip("c5", b, 3.0, 3.9)])
    res = render(ws, edl, quality="draft", max_inputs=2, burn_captions=False)
    assert res.qc.passed, res.qc.problems
    assert res.qc.duration == pytest.approx(4.9, abs=FRAME)
    assert not list(ws.render_dir.glob("mezz_*.mkv"))  # cleaned up
    assert render_mod.MAX_INPUTS == 24


def test_render_rejects_unknown_sources(media_dir, tmp_path):
    ws, _ = _watched(media_dir, tmp_path)
    edl = EDL(target_platform="reels", width=1080, height=1920, clips=[
        EDLClip(clip_id="c1", source_clip_id="vnope", source_file="/x.mp4", start=0, end=1)])
    with pytest.raises(ValueError, match="not in manifest"):
        render(ws, edl, quality="draft")
    assert Workspace.read(ws.clips_manifest, ClipsManifest).clips
