"""`llmve mcp`: the editor agent's tools served over MCP (stdio).

Claude Code connects to it from `.mcp.json`; the one-command mode runs `claude -p`
with the same server. Each call runs in a worker thread with its own context, so
concurrent calls cannot mix up their images. In ``stills`` privacy mode, contact
sheets, review sheets and thumbnails a tool produced are attached as images; in
``text-only`` mode no image is ever attached. Video files are never sent.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import mimetypes
from pathlib import Path

from .tools import REGISTRY, Context, call

INSTRUCTIONS = (
    "You are the video editor. These tools are your instruments: they execute and "
    "report, you decide. Start with open_project (use `first` to work on a few clips), "
    "then analyze, overview, transcript and frames to understand the footage. Build the "
    "edit with new_edit and batch/add_clip, refine with the other edit tools (always "
    "give a reason), check with audit and review, then render and publish. Every change "
    "is a version; undo restores any earlier one. analyze, review and render can take "
    "minutes: when they return a job_id, call job_status until it is done."
)


def _images(paths: list[str], privacy: str) -> list:
    import mcp.types as t

    if privacy != "stills":
        return []
    out = []
    for p in paths[:8]:  # a call never floods the context with images
        path = Path(p)
        if not path.exists():
            continue
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        if not mime.startswith("image/"):
            continue  # never video or audio
        out.append(t.ImageContent(type="image", mime_type=mime,
                                  data=base64.b64encode(path.read_bytes()).decode()))
    return out


def build_server(ctx: Context):
    import anyio
    import mcp.types as t
    from mcp.server.lowlevel import Server

    async def list_tools(rctx, params) -> t.ListToolsResult:
        return t.ListToolsResult(tools=[
            t.Tool(name=x.name, description=x.description, input_schema=x.schema())
            for x in REGISTRY.values()])

    async def call_tool(rctx, params) -> t.CallToolResult:
        local = dataclasses.replace(ctx, images_out=[])
        res = await anyio.to_thread.run_sync(
            lambda: call(params.name, dict(params.arguments or {}), local))
        content: list = [t.TextContent(type="text",
                                       text=json.dumps(res, indent=1, default=str))]
        content += _images(local.images_out, local.privacy)
        return t.CallToolResult(content=content, is_error="error" in res)

    return Server("llmve", version="0.3", instructions=INSTRUCTIONS,
                  on_list_tools=list_tools, on_call_tool=call_tool)


def serve(ctx: Context) -> None:
    import anyio
    from mcp.server.stdio import stdio_server

    async def main():
        server = build_server(ctx)
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(main)
