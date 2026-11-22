"""AI voice-over: spoken text, cached clean synthesis, the agent's voiceover tool, mix
(music ducks, the scene dips), captions and publish disclosure."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from llm_video_editor.agent.tools import Context, call
from llm_video_editor.ffmpeg.runner import ffmpeg
from llm_video_editor.voice import engines as eng_mod
from llm_video_editor.voice.engines import cache_dir
from llm_video_editor.voice.text import apply_lexicon, number_words, ordinal_words, spoken
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeTranscriber
from tests.fixtures.make_fixture import make_clip


def test_spoken_form_numbers_units_and_names():
    assert number_words(380) == "three hundred and eighty"
    assert ordinal_words(11) == "eleventh" and ordinal_words(22) == "twenty-second"
    lex = {"khajuraho": "Kha-joo-raa-ho"}
    assert spoken("380 km to Khajuraho, built in the 11th century", "en-us", lex) == \
        "three hundred and eighty kilometres to Kha-joo-raa-ho, built in the eleventh century"
    assert spoken("At 60 km/h for 2 hrs, 1 km left", "en-gb", {}) == \
        "At sixty kilometres per hour for two hours, one kilometre left"
    assert spoken("Rs 50 chai in 2026", "en-us", {}) == "fifty rupees chai in twenty twenty-six"
    assert spoken("380 km", "hi", {}) == "380 km"  # only English gets number words
    assert apply_lexicon("KHAJURAHO!", lex) == "Kha-joo-raa-ho!"


class FakeEngine:
    """Writes a tone whose length follows the text, with silence around it."""

    name = "fake"

    def __init__(self):
        self.calls = []

    def model_id(self):
        return "fake/1"

    def voices(self):
        return [eng_mod.VoiceInfo("fake", "af_heart", "English (US)", "en-us", "female", "MIT")]

    def synth(self, text, voice, out_wav, *, speed=1.0, lang=None):
        self.calls.append(text)
        dur = 0.06 * len(text) / speed
        ffmpeg("-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono:d=0.4", "-f", "lavfi", "-i",
               f"sine=f=330:sample_rate=24000:d={dur:.3f}", "-filter_complex",
               "[0][1][0]concat=n=3:v=0:a=1", str(out_wav), timeout=60)
        return out_wav


@pytest.fixture
def fake_engine(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr(eng_mod, "_LOADED", {"kokoro": fake})
    return fake


SCRIPT = [(0.3, "So today I ride from Bhopal."), (2.3, "The road is long.")]


def ok(res):
    assert "error" not in res, res
    return res


@requires_ffmpeg
def test_voiceover_tool_adds_clean_cached_lines_and_reports_fit(tmp_path, fake_engine,
                                                                monkeypatch):
    monkeypatch.setenv("LLMVE_CONFIG", str(tmp_path / "cfg"))
    raw = tmp_path / "trip"
    make_clip(raw / "talk.mp4", duration=5, size=(640, 360), audio="sine")
    make_clip(raw / "ride.mp4", duration=8, size=(640, 360), audio=None)
    c = Context(home=tmp_path / "home", transcriber_factory=lambda m: FakeTranscriber(SCRIPT))
    ws = ok(call("open_project", {"inputs": [str(raw)]}, c))["workspace"]
    clips = {Path(x["file"]).stem: x["clip_id"] for x in ok(call("analyze", {"workspace": ws},
                                                                 c))["clips"]}
    ok(call("new_edit", {"workspace": ws, "target_platform": "youtube", "reason": "start"}, c))
    ok(call("batch", {"workspace": ws, "reason": "talk then ride", "operations": [
        {"op": "add_clip", "params": {"source_clip_id": clips["talk"], "start": 0, "end": 4}},
        {"op": "add_clip", "params": {"source_clip_id": clips["ride"], "start": 0,
                                      "end": 7.5}}]}, c))
    ok(call("pronounce", {"workspace": ws, "reason": "place name",
                          "say_as": {"Khajuraho": "Kha-joo-raa-ho"}}, c))

    aud = ok(call("voiceover", {"workspace": ws, "reason": "try a line", "audition": True,
                                "lines": [{"text": "380 km to Khajuraho.",
                                           "anchor_clip_id": "c002", "offset": 0.5}]}, c))
    assert aud["lines"][0]["spoken"] == "three hundred and eighty kilometres to Kha-joo-raa-ho."
    assert ok(call("edit_state", {"workspace": ws}, c))["music"] == []  # audition: no change

    res = ok(call("voiceover", {"workspace": ws, "reason": "context over the riding", "lines": [
        {"text": "380 km to Khajuraho.", "anchor_clip_id": "c002", "offset": 0.5},
        {"text": "Me again, over myself.", "anchor_clip_id": "c001", "offset": 0.0}]}, c))
    assert len(fake_engine.calls) == 2  # the first line came from the cache
    line = res["lines"][0]
    # the 0.4 s of silence on each side is trimmed: only the spoken tone remains
    assert line["seconds"] == pytest.approx(0.06 * len(line["spoken"]), abs=0.15)
    fit = {f["track_id"]: f for f in res["fit"]}
    assert fit["a001"]["issues"] == [] and "over" in fit["a002"]["issues"][0]
    music = ok(call("edit_state", {"workspace": ws}, c))["music"]
    assert music[0]["role"] == "voiceover" and music[0]["text"] == "380 km to Khajuraho."
    assert music[0]["generator"] == "fake:af_heart" and music[0]["volume_db"] == 0
    au = ok(call("audit", {"workspace": ws}, c))
    assert [v["track_id"] for v in au["voiceover"]] == ["a002"]

    # captions carry the narration as written; publish discloses the AI voice
    from llm_video_editor.schemas import EDL, Transcript
    from llm_video_editor.stages.captions import output_words
    from llm_video_editor.workspace import Workspace
    w = Workspace.load(Path(ws))
    edl = EDL.model_validate_json((Path(ws) / "edl_raw.json").read_text())
    words = output_words(edl, Workspace.read(w.transcript, Transcript), edl.output_starts(),
                         [x.duration for x in edl.clips])
    vo_words = [x[2] for x in words if x[0] >= 4.4]
    assert vo_words[:4] == ["380", "km", "to", "Khajuraho."]

    from llm_video_editor.stages.publish import ai_voices, build_description, used_credits
    assert ai_voices(edl) == ["fake:af_heart"]
    _, missing = used_credits(w, edl)
    assert not missing  # generated narration needs no licence record
    copy = type("C", (), {"hook": "h", "body": "b"})()
    assert "Narration: AI-generated voice (Fake)" in build_description(copy, [], [], [],
                                                                       ai_voices(edl))


def _band(path, freq, start, dur):
    proc = ffmpeg("-ss", str(start), "-t", str(dur), "-i", str(path), "-vn", "-af",
                  f"bandpass=f={freq}:width_type=h:w=40,astats=measure_overall=RMS_level"
                  ":measure_perchannel=none", "-f", "null", "-", timeout=60)
    vals = re.findall(r"RMS level dB: (-?[0-9.]+|-inf)", proc.stderr)
    return float(vals[-1]) if vals and vals[-1] != "-inf" else -120.0


@requires_ffmpeg
def test_scene_sound_dips_under_the_narration(tmp_path):
    """The clip's own sound (440 Hz here) is lower while the voice-over plays."""
    from llm_video_editor.schemas import EDL, EDLClip, MixSettings, MusicTrack
    from llm_video_editor.stages.render import render
    from llm_video_editor.stages.watch import watch
    from llm_video_editor.workspace import Workspace

    raw = tmp_path / "raw"
    make_clip(raw / "ride.mp4", duration=8, size=(640, 360), audio="sine")
    ws = Workspace.open(raw, home=tmp_path / "home")
    info = watch(ws, transcriber=FakeTranscriber(silent=True)).manifest.clips[0]
    vo = tmp_path / "vo.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=f=250:sample_rate=48000:d=3", str(vo), timeout=60)
    levels = {}
    for depth in (0.0, 8.0):
        edl = EDL(target_platform="youtube", width=640, height=360,
                  captions={"enabled": False}, mix=MixSettings(clip_duck_db=depth),
                  clips=[EDLClip(clip_id="c1", source_clip_id=info.clip_id,
                                 source_file=info.source_file, start=0, end=7)],
                  music=[MusicTrack(track_id="v", file=str(vo), role="voiceover",
                                    anchor_clip_id="c1", offset=3.0)])
        out = render(ws, edl, quality="draft", burn_captions=False).output
        levels[depth] = _band(out, 440, 3.5, 2.0) - _band(out, 440, 0.5, 2.0)
    assert levels[8.0] < levels[0.0] - 4, levels


KOKORO = cache_dir() / "kokoro"
has_kokoro = pytest.mark.skipif(
    not (KOKORO / "voices-v1.0.bin").exists()
    or not any((KOKORO / m).exists() for m in eng_mod.KOKORO_MODELS)
    or os.environ.get("LLMVE_SKIP_TTS"), reason="Kokoro model files not in the cache")


@has_kokoro
@requires_ffmpeg
def test_real_kokoro_english_and_hindi(tmp_path):
    from llm_video_editor.voice import get_engine, synthesize

    eng = get_engine("kokoro")
    en = synthesize(tmp_path, eng, "380 km later, Khajuraho.", "am_michael")
    hi = synthesize(tmp_path, eng, "खजुराहो के मंदिर बहुत सुंदर हैं।", "hm_omega")
    assert 2.0 < en.duration < 6.0 and 1.5 < hi.duration < 6.0
    assert en.spoken.startswith("three hundred and eighty kilometres")
    assert {v.voice for v in eng.voices()} >= {"af_heart", "hm_omega", "bf_emma"}
