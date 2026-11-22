"""publish: upload.md + upload.json + 3 thumbnails next to the render."""

from __future__ import annotations

import pytest

from llm_video_editor.ffmpeg.runner import ffmpeg, ffprobe_json
from llm_video_editor.hyperframes import toolchain
from llm_video_editor.media.library import register
from llm_video_editor.schemas import EDL, EDLClip, MusicTrack, Overlay, UploadPackage
from llm_video_editor.stages import publish as pub
from llm_video_editor.stages.cut_audit import run_cut_audit
from llm_video_editor.stages.render import render
from llm_video_editor.stages.watch import watch
from llm_video_editor.workspace import Workspace
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeProvider, FakeTranscriber
from tests.fixtures.make_fixture import make_clip

pytestmark = [requires_ffmpeg, pytest.mark.integration]
SCRIPT = [(0.5, "I tested five cheap microphones for a week."),
          (12.0, "One of them sounded better than my studio mic."),
          (24.0, "The worst one cost the most."), (36.0, "Here is the winner.")]


def copy(schema, prompt):
    copy.prompts.append(prompt)
    return schema.model_validate({
        "titles": ["I Tested 5 Cheap Mics: One Beat My Studio Mic", "5 Budget Mics <Tested>"],
        "hook": "A week with five budget microphones.", "body": "Same voice, same room.",
        "tags": ["budget microphone", "mic test"], "hashtags": ["microphone", "budget tech"],
        "category": "Science & Technology", "pinned_comment": "Which one would you pick?",
        "thumbnails": [
            {"clip_id": "c002", "segment_id": 1, "text": "Beat my studio mic?",
             "style": "question", "why": "curiosity"},
            {"clip_id": "c003", "segment_id": 2, "text": "Priciest = worst", "style": "question",
             "why": "contrast"},
            {"clip_id": "c999", "segment_id": 1, "text": "nope", "style": "bold", "why": "x"},
            {"clip_id": "c004", "segment_id": 3, "text": "The winner", "style": "reveal",
             "why": "payoff"}]})


copy.prompts = []


@pytest.fixture
def rendered(tmp_path):
    tmp = make_clip(tmp_path / "tmp.mp4", duration=48, size=(1280, 720))
    raw = tmp_path / "raw"
    raw.mkdir()
    ffmpeg("-i", str(tmp), "-c", "copy", "-metadata", "creation_time=2026-03-14T09:30:00Z",
           "-metadata", "location=+28.6139+077.2090+216.000/",
           "-metadata", "com.apple.quicktime.model=iPhone 16 Pro",
           "-movflags", "use_metadata_tags", str(raw / "vlog.mov"))
    broll = make_clip(tmp_path / "stock" / "street.mp4", duration=4, size=(1280, 720), audio=None)
    ws = Workspace.open(raw, home=tmp_path / "home")
    info = watch(ws, transcriber=FakeTranscriber(SCRIPT)).manifest.clips[0]
    ffmpeg("-f", "lavfi", "-i", "sine=f=220:d=60", "-ac", "2", str(tmp_path / "bed.mp3"))
    bed = register(ws, tmp_path / "bed.mp3", kind="music", title="Night Drive",
                   license="YouTube Audio Library", provider="youtube-audio-library",
                   attribution="Night Drive by Artist X")

    from llm_video_editor.schemas import ClipsManifest
    from llm_video_editor.stages.cutaways import _register
    manifest = Workspace.read(ws.clips_manifest, ClipsManifest)
    stock = _register(manifest, broll, "street")
    Workspace.write(ws.clips_manifest, manifest)

    def mk(cid, a, b, ch):
        return EDLClip(clip_id=cid, source_clip_id=info.clip_id, source_file=info.source_file,
                       start=a, end=b, chapter=ch)

    Workspace.write(ws.edl_raw, EDL(
        target_platform="youtube", width=1280, height=720, captions={"enabled": False},
        clips=[mk("c001", 0, 12, "The test"), mk("c002", 12, 24, "Surprise winner"),
               mk("c003", 24, 36, "The flop"), mk("c004", 36, 47.9, "Winner")],
        music=[MusicTrack(track_id="a001", file=bed.file, anchor_clip_id="c001")],
        overlays=[Overlay(overlay_id="o1", source_clip_id=stock.clip_id,
                          source_file=stock.source_file, anchor_clip_id="c002", offset=1,
                          duration=2)]))
    run_cut_audit(ws)
    out = tmp_path / "out" / "final.mp4"
    out.parent.mkdir()
    render(ws, quality="draft", burn_captions=False, out=out)
    return ws, out


def test_publish_writes_package_next_to_the_video(rendered, monkeypatch):
    ws, video = rendered
    copy.prompts.clear()
    monkeypatch.setattr(pub, "reverse_geocode", lambda lat, lon: "New Delhi, Delhi, India")
    pkg = pub.publish(ws, FakeProvider(copy), video=video, geocode=True,
                      prefer_hyperframes=False)
    folder = video.parent
    assert (folder / "final.upload.md").exists() and (folder / "final.upload.json").exists()
    assert Workspace.read(folder / "final.upload.json", UploadPackage) == pkg

    assert pkg.titles == ["I Tested 5 Cheap Mics: One Beat My Studio Mic", "5 Budget Mics Tested"]
    assert pkg.hashtags == ["#microphone", "#budgettech"]
    d = pkg.description
    assert d.startswith("A week with five budget microphones.")
    assert "Chapters\n0:00 The test" in d and "- Night Drive by Artist X" in d
    assert d.endswith("#microphone #budgettech")

    assert pkg.source.latitude == pytest.approx(28.6139) and pkg.source.device == "iPhone 16 Pro"
    assert pkg.source.place == "New Delhi, Delhi, India"
    assert "FILMED IN" not in copy.prompts[0]  # location only reaches the copy with use_location
    assert "street.mp4" not in copy.prompts[0]  # never a media path in a prompt

    audited = Workspace.read(ws.edl_audited, EDL)
    assert any("no licence/credit recorded for street.mp4" in w for w in pkg.warnings), (
        pkg.warnings, audited.overlays)
    assert any("c999#1 is not in the cut" in w for w in pkg.warnings)

    assert [t.style for t in pkg.thumbnails] == ["question", "bold", "reveal"]  # styles unique
    for t in pkg.thumbnails:
        s = ffprobe_json(t.path)["streams"][0]
        assert (s["width"], s["height"]) == (1280, 720)
        assert (folder / t.path.rsplit("/", 1)[-1]).stat().st_size < 2_000_000
        assert t.renderer == "ffmpeg"
    assert not list(folder.glob(".final.frame*"))  # temporary frames cleaned up

    md = (folder / "final.upload.md").read_text()
    for heading in ("## Title", "## Description", "## Tags", "## Thumbnails", "## Chapters",
                    "## Music and footage credits", "## Video details", "## Pinned comment"):
        assert heading in md, heading
    assert "New Delhi, Delhi, India" in md and "iPhone 16 Pro" in md and "final.srt" in md


def test_location_reaches_the_copy_only_when_asked(rendered, monkeypatch):
    ws, video = rendered
    copy.prompts.clear()
    monkeypatch.setattr(pub, "reverse_geocode", lambda lat, lon: "New Delhi, Delhi, India")
    pub.publish(ws, FakeProvider(copy), video=video, geocode=True, use_location=True,
                thumbnails=False)
    assert "FILMED IN: New Delhi, Delhi, India" in copy.prompts[0]


def test_no_geocoding_unless_asked(rendered, monkeypatch):
    ws, video = rendered

    def boom(*a):
        raise AssertionError("geocoded without --geocode")

    monkeypatch.setattr(pub, "reverse_geocode", boom)
    pkg = pub.publish(ws, FakeProvider(copy), video=video, thumbnails=False)
    assert pkg.source.place is None and pkg.source.latitude is not None


@pytest.mark.hyperframes
@pytest.mark.skipif(not (toolchain.toolchain().installed and (toolchain.node_major() or 0) >= 22),
                    reason="HyperFrames toolchain not installed")
def test_real_hyperframes_thumbnails(rendered):
    ws, video = rendered
    pkg = pub.publish(ws, FakeProvider(copy), video=video)
    assert [t.renderer for t in pkg.thumbnails] == ["hyperframes"] * 3


def test_thumbnails_by_time_for_footage_without_speech(rendered):
    ws, video = rendered
    cut = {c.clip_id: c for c in Workspace.read(ws.edl_audited, EDL).clips}
    mid1 = round((cut["c001"].start + cut["c001"].end) / 2, 2)
    mid3 = round((cut["c003"].start + cut["c003"].end) / 2, 2)
    copy_ = pub.UploadCopy.model_validate({
        "titles": ["A ride"], "hook": "h", "body": "b", "tags": ["t"], "hashtags": ["x"],
        "category": "Travel & Events", "pinned_comment": "", "thumbnails": [
            {"clip_id": "c001", "at": mid1, "text": "The road", "style": "bold", "why": "w"},
            {"clip_id": "c003", "at": mid3, "text": "Temple", "style": "reveal", "why": "w"},
            {"clip_id": "c004", "at": 2.0, "text": "nope", "style": "question", "why": "w"}]})
    pkg = pub.publish(ws, None, video=video, copy=copy_, prefer_hyperframes=False)
    assert [t.text for t in pkg.thumbnails] == ["The road", "Temple"], pkg.warnings
    assert pkg.thumbnails[0].t_out == pytest.approx(mid1 - 0.5 - cut["c001"].start, abs=0.6)
    assert any("c004@2.0s is outside the clip" in w for w in pkg.warnings)
