import json

import pytest

from llm_video_editor.agent import setup


@pytest.fixture(autouse=True)
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMVE_CONFIG", str(tmp_path / "cfg"))


def test_profile_round_trip_and_rules():
    assert setup.load_profile().privacy == "stills"
    p = setup.update_profile(set_fields={"channel": "Solo Rides", "languages": ["en", "hi"]},
                             add_dislikes=["film grain"], add_likes=["warm golden hour"])
    assert p.channel == "Solo Rides" and p.dislikes == ["film grain"]
    p = setup.update_profile(add_dislikes=["film grain"], remove=["warm golden hour"])
    assert p.dislikes == ["film grain"] and p.likes == []
    assert setup.load_profile().languages == ["en", "hi"] and setup.load_profile().updated_at
    with pytest.raises(ValueError, match="cannot set 'likes'"):
        setup.update_profile(set_fields={"likes": ["x"]})


def test_remember_tool_cannot_change_privacy():
    from llm_video_editor.agent.tools import call

    res = call("remember", {"reason": "user said so", "set_fields": {"privacy": "stills"}})
    assert "privacy is the user's setting" in res["error"]
    res = call("remember", {"reason": "user disliked grain", "add_dislikes": ["film grain"]})
    assert res["profile"]["dislikes"] == ["film grain"]
    assert call("profile", {})["profile"]["dislikes"] == ["film grain"]


def test_mcp_config_passes_custom_config(tmp_path):
    conf = setup.mcp_config(tmp_path, "text-only")["mcpServers"]["llmve"]
    assert conf["args"][-2:] == ["--privacy", "text-only"]
    assert conf["env"]["LLMVE_CONFIG"].endswith("cfg")
    json.dumps(conf)
