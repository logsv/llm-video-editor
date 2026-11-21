"""The tool server over MCP: a real client session against the server (in memory)."""

from __future__ import annotations

import json

import anyio
import pytest

mcp = pytest.importorskip("mcp")

from mcp import ClientSession  # noqa: E402
from mcp.shared.memory import create_client_server_memory_streams  # noqa: E402

from llm_video_editor.agent.mcp_server import build_server  # noqa: E402
from llm_video_editor.agent.tools import REGISTRY, Context  # noqa: E402
from tests.conftest import requires_ffmpeg  # noqa: E402
from tests.fakes import FakeTranscriber  # noqa: E402
from tests.fixtures.make_fixture import make_clip  # noqa: E402

pytestmark = [requires_ffmpeg, pytest.mark.integration]


async def session_calls(ctx: Context, calls):
    server = build_server(ctx)
    results = {}
    async with create_client_server_memory_streams() as (client, srv), \
            anyio.create_task_group() as tg:
        tg.start_soon(lambda: server.run(srv[0], srv[1], server.create_initialization_options()))
        async with ClientSession(client[0], client[1]) as session:
            await session.initialize()
            listed = await session.list_tools()
            results["tools"] = [t.name for t in listed.tools]
            results["schemas"] = {t.name: t.input_schema for t in listed.tools}
            state: dict = {}
            for key, name, args in calls:
                args = args(state) if callable(args) else args
                res = await session.call_tool(name, args)
                payload = json.loads(res.content[0].text)
                state[key] = payload
                results[key] = (res.is_error, payload, [c.type for c in res.content[1:]])
        tg.cancel_scope.cancel()
    return results


def plan(raw):
    return [
        ("open", "open_project", {"inputs": [str(raw)]}),
        ("analyze", "analyze", lambda s: {"workspace": s["open"]["workspace"]}),
        ("frames", "frames", lambda s: {"workspace": s["open"]["workspace"],
                                        "clip_id": s["analyze"]["clips"][0]["clip_id"],
                                        "start": 0, "end": 2}),
        ("bad", "trim_clip", lambda s: {"workspace": s["open"]["workspace"],
                                        "clip_id": "c404", "reason": "nope"}),
    ]


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw"
    make_clip(d / "ride.mp4", duration=4, size=(640, 360))
    return d


def ctx(tmp_path, privacy):
    return Context(privacy=privacy, home=tmp_path / privacy,
                   transcriber_factory=lambda m: FakeTranscriber())


def test_tools_listed_and_images_only_in_stills_mode(raw, tmp_path):
    res = anyio.run(session_calls, ctx(tmp_path, "stills"), plan(raw))
    assert sorted(res["tools"]) == sorted(REGISTRY)
    assert "reason" in res["schemas"]["trim_clip"]["required"]
    err, payload, extra = res["frames"]
    assert not err and payload["sheets"] and extra == ["image"]  # sheet attached
    err, payload, _ = res["bad"]
    assert err and "edit" in payload["error"]  # no edit yet -> error flagged


def test_text_only_never_attaches_images(raw, tmp_path):
    res = anyio.run(session_calls, ctx(tmp_path, "text-only"), plan(raw))
    err, payload, extra = res["frames"]
    assert err and "text-only" in payload["error"] and extra == []
    assert all(r[2] == [] for k, r in res.items() if k not in ("tools", "schemas"))
