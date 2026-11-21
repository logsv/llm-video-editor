import json
import re

import pytest
from typer.testing import CliRunner

from llm_video_editor import llm as llm_pkg
from llm_video_editor.cli.app import app
from llm_video_editor.stages import asr as asr_mod
from llm_video_editor.stages.plan import ReduceResult
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeProvider, FakeTranscriber

pytestmark = [requires_ffmpeg, pytest.mark.integration]


def test_llmve_run_end_to_end(media_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(asr_mod, "FasterWhisperTranscriber", lambda *a, **k: FakeTranscriber())

    def responder(schema, prompt):
        ids = sorted(set(re.findall(r"chunk (v[0-9a-f]{8})#", prompt)))
        return ReduceResult.model_validate({"selections": [
            {"clip_id": ids[0], "first_segment": 0, "last_segment": 1, "purpose": "all",
             "priority": 1}]})

    monkeypatch.setattr(llm_pkg, "get_provider", lambda *a, **k: FakeProvider(responder))
    out = tmp_path / "final.mp4"
    args = ["run", str(media_dir), "-p", "4s reel: the intro", "-t", "reels", "--draft",
            "-o", str(out), "--home", str(tmp_path / "home"), "--detector", "none"]
    stopped = CliRunner().invoke(app, args)
    assert stopped.exit_code == 2 and "review sheet:" in stopped.output  # gate holds
    assert not out.exists()
    result = CliRunner().invoke(app, [*args, "--approve"])
    assert result.exit_code == 0, result.output
    assert out.exists() and "QC PASSED" in result.output
    ws_root = next((tmp_path / "home").iterdir())
    for name in ("transcript.json", "clips_manifest.json", "asset_plan.json", "edl_raw.json",
                 "cut_audit.json", "edl_audited.json", "geometry_map.json",
                 "review/review.json", "review/sheet_000.png", "render/qc_report.json"):
        assert (ws_root / name).exists(), name
    assert json.loads((ws_root / "cut_audit.json").read_text())["passed"] is True

    schema = CliRunner().invoke(app, ["schema", "edl"])
    assert schema.exit_code == 0 and '"clips"' in schema.output


def test_step_commands_from_the_readme(media_dir, tmp_path, monkeypatch):
    """`review <ws>`, `review reject|approve <ws> --notes`, `transcript`, `render`."""
    monkeypatch.setattr(asr_mod, "FasterWhisperTranscriber", lambda *a, **k: FakeTranscriber())

    def responder(schema, prompt):
        ids = sorted(set(re.findall(r"chunk (v[0-9a-f]{8})#", prompt)))
        return ReduceResult.model_validate({"selections": [
            {"clip_id": ids[0], "first_segment": 0, "last_segment": 1, "purpose": "all",
             "priority": 1}]})

    monkeypatch.setattr(llm_pkg, "get_provider", lambda *a, **k: FakeProvider(responder))
    cli = CliRunner()
    home = tmp_path / "home"
    stopped = cli.invoke(app, ["run", str(media_dir), "-p", "4s reel", "-t", "reels",
                               "--home", str(home), "--detector", "none"])
    assert stopped.exit_code == 2, stopped.output
    ws = str(next(home.iterdir()))

    transcript = cli.invoke(app, ["transcript", ws])
    assert transcript.exit_code == 0 and "[0]" in (home / ws / "transcript.md").read_text()

    built = cli.invoke(app, ["review", ws])
    assert built.exit_code == 0 and "sheet_000.png" in built.output, built.output
    rejected = cli.invoke(app, ["review", "reject", ws, "--notes", "tighter"])
    assert rejected.exit_code == 0 and "rejected" in rejected.output, rejected.output
    blocked = cli.invoke(app, ["render", ws, "--draft"])
    assert blocked.exit_code == 2  # a rejected review does not unlock render

    cli.invoke(app, ["review", ws])
    approved = cli.invoke(app, ["review", "approve", ws, "--notes", "ok"])
    assert approved.exit_code == 0 and "approved" in approved.output, approved.output
    out = tmp_path / "final.mp4"
    rendered = cli.invoke(app, ["render", ws, "--draft", "-o", str(out)])
    assert rendered.exit_code == 0 and "QC PASSED" in rendered.output, rendered.output
    assert out.with_suffix(".srt").exists()

    assert cli.invoke(app, ["review"]).exit_code in (0, 2)  # bare group prints usage
