import pytest
from typer.testing import CliRunner

from llm_video_editor.cli.app import app
from llm_video_editor.ffmpeg.runner import ffmpeg
from llm_video_editor.media import library
from llm_video_editor.schemas import EDL, EDLClip
from llm_video_editor.stages.watch import watch
from llm_video_editor.workspace import Workspace
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeTranscriber
from tests.fixtures.make_fixture import make_clip

pytestmark = [requires_ffmpeg, pytest.mark.integration]


def test_media_and_music_commands(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    make_clip(raw / "talk.mp4", duration=4, size=(640, 360))
    ws = Workspace.open(raw, home=tmp_path / "home")
    info = watch(ws, transcriber=FakeTranscriber()).manifest.clips[0]
    Workspace.write(ws.edl_raw, EDL(target_platform="youtube", width=640, height=360, clips=[
        EDLClip(clip_id="c1", source_clip_id=info.clip_id, source_file=info.source_file,
                start=0, end=3)]))
    track = tmp_path / "Night Drive.mp3"
    ffmpeg("-f", "lavfi", "-i", "sine=f=220:d=5", str(track))
    cli, root = CliRunner(), str(ws.root)

    added = cli.invoke(app, ["media", "add", root, str(track), "--title", "Night Drive",
                             "--license", "YouTube Audio Library",
                             "--provider", "youtube-audio-library",
                             "--attribution", "Night Drive by Artist X"])
    assert added.exit_code == 0, added.output
    stored = library.load_credits(ws).items[0].file
    assert "youtube-audio-library-night-drive.mp3" in stored
    credits = cli.invoke(app, ["media", "credits", root])
    assert "Night Drive by Artist X" in credits.output and "credit required" in credits.output

    for args in (["--at", "c1"], ["--at", "c1", "--role", "sfx", "--offset", "1"],
                 ["--at", "c1", "--role", "voiceover"]):
        r = cli.invoke(app, ["music", "add", root, stored, *args])
        assert r.exit_code == 0, r.output
    music = Workspace.read(ws.edl_raw, EDL).music
    assert [(m.track_id, m.role, m.volume_db, m.duck) for m in music] == [
        ("a001", "music", -10.0, True), ("a002", "sfx", -4.0, False),
        ("a003", "voiceover", 0.0, False)]
    assert cli.invoke(app, ["music", "add", root, stored, "--at", "c9"]).exit_code != 0

    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.setattr(library, "PROVIDERS", {"pixabay": library.PROVIDERS["pixabay"],
                                               "pexels": library.PROVIDERS["pexels"]})
    res = cli.invoke(app, ["media", "search", root, "city", "--kind", "video",
                           "-p", "pixabay", "-p", "pexels"])
    assert res.exit_code == 0 and "PIXABAY_API_KEY" in res.output and "PEXELS_API_KEY" in res.output


def test_media_place_puts_one_cutaway_at_an_exact_moment(tmp_path):
    from llm_video_editor.schemas import ClipsManifest
    from llm_video_editor.stages.cut_audit import run_cut_audit

    raw = tmp_path / "raw"
    make_clip(raw / "talk.mp4", duration=6, size=(640, 360))
    ws = Workspace.open(raw, home=tmp_path / "home")
    info = watch(ws, transcriber=FakeTranscriber()).manifest.clips[0]
    Workspace.write(ws.edl_raw, EDL(target_platform="youtube", width=640, height=360, clips=[
        EDLClip(clip_id="c1", source_clip_id=info.clip_id, source_file=info.source_file,
                start=0, end=4)]))
    clip = make_clip(tmp_path / "dl" / "cat.mp4", duration=5, size=(640, 360), audio=None)
    res = CliRunner().invoke(app, ["media", "place", str(ws.root), str(clip), "--at", "c1",
                                   "--offset", "1", "--duration", "2"])
    assert res.exit_code == 0, res.output
    (ov,) = Workspace.read(ws.edl_raw, EDL).overlays
    assert (ov.overlay_id, ov.anchor_clip_id, ov.offset, ov.duration) == ("p001", "c1", 1.0, 2.0)
    assert Workspace.read(ws.clips_manifest, ClipsManifest).get(ov.source_clip_id).kind == "broll"
    audited, rep = run_cut_audit(ws)
    assert rep.passed and len(audited.overlays) == 1
