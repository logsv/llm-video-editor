"""Music beds and sound effects through cut-audit and render.

The "voice" is the fixture's 440 Hz tone, the "music" a 220 Hz tone. Measuring both
bands in the same output makes the checks independent of the final loudness pass:
ducking must lower music relative to voice.
"""

from __future__ import annotations

import re

import pytest

from llm_video_editor.ffmpeg.runner import ffmpeg
from llm_video_editor.schemas import EDL, EDLClip, MusicTrack
from llm_video_editor.stages.cut_audit import audit
from llm_video_editor.stages.render import render
from llm_video_editor.stages.watch import watch
from llm_video_editor.workspace import Workspace
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeTranscriber
from tests.fixtures.make_fixture import make_clip

pytestmark = [requires_ffmpeg, pytest.mark.integration]
FRAME = 1 / 30


@pytest.fixture
def ws_info(tmp_path):
    raw = tmp_path / "raw"
    make_clip(raw / "talk.mp4", duration=5, size=(640, 360), fps=30, audio="sine")
    ws = Workspace.open(raw, home=tmp_path / "home")
    info = watch(ws, transcriber=FakeTranscriber()).manifest.clips[0]
    return ws, info


def tone(path, freq, seconds):
    ffmpeg("-f", "lavfi", "-i", f"sine=f={freq}:sample_rate=48000:d={seconds}",
           "-af", "volume=-6dB", "-ac", "2", str(path), timeout=60)
    return path


def band_db(path, freq, start=0.5, dur=3.0):
    proc = ffmpeg("-ss", str(start), "-t", str(dur), "-i", str(path), "-vn", "-af",
                  f"bandpass=f={freq}:width_type=h:w=40,astats=measure_overall=RMS_level"
                  ":measure_perchannel=none", "-f", "null", "-", timeout=60)
    vals = re.findall(r"RMS level dB: (-?[0-9.]+|-inf)", proc.stderr)
    return float(vals[-1]) if vals and vals[-1] != "-inf" else -120.0


def _edl(info, music):
    return EDL(target_platform="youtube", width=640, height=360, captions={"enabled": False},
               clips=[EDLClip(clip_id="c1", source_clip_id=info.clip_id,
                              source_file=info.source_file, start=0.0, end=4.0)],
               music=music)


def M(path, **kw):
    return MusicTrack(track_id=kw.pop("track_id", "m1"), file=str(path),
                      anchor_clip_id=kw.pop("anchor", "c1"), **kw)


def gap(path):
    """Voice band minus music band, in dB, inside one output."""
    return band_db(path, 440) - band_db(path, 220)


def test_music_ducks_under_the_clips_own_voice(ws_info, tmp_path):
    ws, info = ws_info
    bed = tone(tmp_path / "bed.wav", 220, 10)
    gaps = {}
    for duck in (False, True):
        res = render(ws, _edl(info, [M(bed, volume_db=0, duck=duck, fade_in=0, fade_out=0)]),
                     quality="draft", burn_captions=False)
        qc = res.qc
        assert qc.passed, qc.problems
        assert qc.duration == pytest.approx(4.0, abs=2 * FRAME)  # music never lengthens the cut
        assert qc.integrated_lufs == pytest.approx(-14.0, abs=1.0)
        gaps[duck] = gap(res.output)
    assert gaps[True] - gaps[False] > 8, gaps  # default depth: 12 dB under speech


@pytest.fixture
def silent_ws(tmp_path):
    raw = tmp_path / "silent"
    make_clip(raw / "broll.mp4", duration=5, size=(640, 360), fps=30, audio=None)
    ws = Workspace.open(raw, home=tmp_path / "home2")
    info = watch(ws, transcriber=FakeTranscriber(silent=True)).manifest.clips[0]
    return ws, info


def test_music_ducks_under_a_voiceover(silent_ws, tmp_path):
    ws, info = silent_ws
    bed = tone(tmp_path / "bed.wav", 220, 10)
    vo = tone(tmp_path / "narration.wav", 440, 10)
    gaps = {}
    for duck in (False, True):
        edl = _edl(info, [M(bed, volume_db=0, duck=duck, fade_in=0, fade_out=0),
                          M(vo, track_id="vo", role="voiceover")])
        res = render(ws, edl, quality="draft", burn_captions=False)
        assert res.qc.passed, res.qc.problems
        gaps[duck] = gap(res.output)
    assert gaps[True] - gaps[False] > 8, gaps


def test_ducking_follows_speech_not_recording_level(silent_ws, tmp_path):
    """A quiet and a loud narration duck the bed by about the same amount."""
    ws, info = silent_ws
    bed = tone(tmp_path / "bed.wav", 220, 10)
    vo = tone(tmp_path / "narration.wav", 440, 10)
    dips = {}
    for level in (-24.0, 0.0):
        out = {}
        for duck in (False, True):
            edl = _edl(info, [M(bed, volume_db=level, duck=duck, fade_in=0, fade_out=0),
                              M(vo, track_id="vo", role="voiceover", volume_db=level)])
            out[duck] = gap(render(ws, edl, quality="draft", burn_captions=False).output)
        dips[level] = out[True] - out[False]
    assert all(d > 8 for d in dips.values()), dips
    assert abs(dips[-24.0] - dips[0.0]) < 3, dips


def test_sound_effect_plays_once_at_its_offset(ws_info, tmp_path):
    ws, info = ws_info
    sfx = tone(tmp_path / "boing.wav", 220, 0.5)
    res = render(ws, _edl(info, [M(sfx, role="sfx", offset=2.0, volume_db=0, duck=False,
                                   fade_in=0, fade_out=0)]),
                 quality="draft", burn_captions=False)
    assert res.qc.passed, res.qc.problems
    assert band_db(res.output, 220, start=2.1, dur=0.3) - band_db(res.output, 220, start=0.5,
                                                                  dur=1.0) > 20


def test_cut_audit_drops_bad_tracks_and_clamps(ws_info, tmp_path):
    ws, info = ws_info
    bed = tone(tmp_path / "bed.wav", 220, 10)
    from llm_video_editor.schemas import ClipsManifest, Transcript
    manifest = Workspace.read(ws.clips_manifest, ClipsManifest)
    tr = Workspace.read(ws.transcript, Transcript)
    edl, rep = audit(_edl(info, [M(bed, track_id="ok", duration=30.0),
                                 M(tmp_path / "missing.mp3", track_id="gone"),
                                 M(bed, track_id="orphan", anchor="c9")]), tr, manifest)
    assert [m.track_id for m in edl.music] == ["ok"]
    assert edl.music[0].duration == pytest.approx(edl.total_duration, abs=1e-3)
    kinds = sorted(v.kind for v in rep.violations)
    assert "unknown_music" in kinds and kinds.count("music_out_of_bounds") == 2
