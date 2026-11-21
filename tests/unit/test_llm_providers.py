import json
import os
import stat
import sys

import pytest

from llm_video_editor.llm import BoundaryError, ProviderError, get_provider, structured
from llm_video_editor.llm.claude_cli import ClaudeCLIProvider
from llm_video_editor.stages.plan import ReduceResult
from tests.fakes import FakeProvider

GOOD = {"selections": [{"clip_id": "vabc", "first_segment": 0, "last_segment": 1,
                        "purpose": "hook", "priority": 1}], "broll": [], "notes": ""}


def _fake_claude(tmp_path, envelope, code=0):
    record = tmp_path / "record.json"
    script = tmp_path / "bin" / "claude"
    script.parent.mkdir()
    script.write_text(f"""#!{sys.executable}
import json, os, sys
json.dump({{"argv": sys.argv[1:], "stdin": sys.stdin.read(), "cwd": os.getcwd()}},
          open({str(record)!r}, "w"))
print({json.dumps(envelope)!r})
sys.exit({code})
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script, record


def test_claude_cli_contract(tmp_path):
    envelope = {"type": "result", "is_error": False, "result": "", "structured_output": GOOD}
    script, record = _fake_claude(tmp_path, envelope)
    provider = ClaudeCLIProvider(executable=str(script))
    out = provider.complete("system text", "pick segments for clip vabc", ReduceResult)
    assert out.selections[0].clip_id == "vabc"

    rec = json.loads(record.read_text())
    argv = rec["argv"]
    assert argv[0] == "-p"
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--tools") + 1] == ""
    assert "--no-session-persistence" in argv
    schema = json.loads(argv[argv.index("--json-schema") + 1])
    assert schema == ReduceResult.model_json_schema()
    assert rec["stdin"] == "pick segments for clip vabc"  # prompt goes via stdin, not argv
    assert "pick segments" not in " ".join(argv)
    assert os.path.basename(rec["cwd"]).startswith("llmve-planner-")
    assert not os.path.exists(rec["cwd"])  # clean temp dir, removed afterwards


def test_claude_cli_is_error_and_bad_payload(tmp_path):
    script, _ = _fake_claude(tmp_path, {"type": "result", "is_error": True, "result": "boom"})
    with pytest.raises(ProviderError, match="boom"):
        ClaudeCLIProvider(executable=str(script)).complete("s", "p", ReduceResult)


def test_claude_cli_list_envelope():
    env = ClaudeCLIProvider.parse_envelope(json.dumps([
        {"type": "system"}, {"type": "result", "is_error": False, "structured_output": GOOD}]))
    assert env["structured_output"] == GOOD


def test_missing_cli_reported():
    with pytest.raises(ProviderError, match="not found"):
        ClaudeCLIProvider(executable="no-such-claude-binary").complete("s", "p", ReduceResult)


@pytest.mark.parametrize("leak", [
    "use /home/me/raw/interview.mp4 please", "clip C0042.MOV", "audio at x.wav",
])
def test_zero_upload_boundary(leak):
    provider = FakeProvider(lambda schema, prompt: pytest.fail("provider must not be called"))
    with pytest.raises(BoundaryError):
        structured(provider, "system", leak, ReduceResult)
    with pytest.raises(BoundaryError):
        ClaudeCLIProvider(executable="claude").complete("s", leak, ReduceResult)


def test_structured_retries_once_on_invalid_output():
    calls = []

    def responder(schema, prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return schema.model_validate({"selections": "nope"})
        return schema.model_validate(GOOD)

    out = structured(FakeProvider(responder), "s", "p", ReduceResult)
    assert out.selections and len(calls) == 2
    assert "rejected" in calls[1]


def test_get_provider_names():
    assert get_provider().name == "claude-cli"
    for name in ("ollama", "anthropic", "openai"):
        assert get_provider(name).name == name
    with pytest.raises(ValueError):
        get_provider("nope")
