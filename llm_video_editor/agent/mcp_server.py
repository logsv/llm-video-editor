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
    "is a version; undo restores any earlier one. Decide the on-screen text too: where "
    "the viewer is (places gives the route and, if the user allowed it, place names), "
    "chapter cards and callouts are graphics (graphics_templates lists them; a "
    "`location` template exists for place titles), and review reports what text the "
    "video has. analyze, review and render can take "
    "minutes: when they return a job_id, call job_status until it is done."
)


MAX_IMAGES = 6  # per call
MAX_SIDE = 1568  # longest side sent to the model; bigger images are scaled down anyway
MAX_TOTAL_BYTES = 3_500_000  # all images of one reply; big replies have closed the connection


def _shrunk(path: Path) -> Path:
    """A JPEG copy at most MAX_SIDE on its longest side (cached next to the image)."""
    from ..ffmpeg.runner import ffmpeg

    out = path.with_name(f".{path.stem}.mcp.jpg")
    if not out.exists() or out.stat().st_mtime < path.stat().st_mtime:
        ffmpeg("-i", str(path), "-vf",
               f"scale='if(gt(iw,ih),min({MAX_SIDE},iw),-2)':'if(gt(iw,ih),-2,min({MAX_SIDE},ih))'",
               "-q:v", "4", "-frames:v", "1", str(out), timeout=60)
    return out


def _images(paths: list[str], privacy: str) -> tuple[list, list[str]]:
    """Image contents for a reply (stills mode only), and the paths left out (sent as
    text instead) once the count or size budget is used up."""
    import mcp.types as t

    if privacy != "stills":
        return [], []
    out, left_out, total = [], [], 0
    for p in paths:
        path = Path(p)
        mime = mimetypes.guess_type(path.name)[0] or ""
        if not path.exists() or not mime.startswith("image/"):
            continue  # never video or audio
        if len(out) >= MAX_IMAGES:
            left_out.append(p)
            continue
        try:
            small = _shrunk(path)
        except Exception:  # noqa: BLE001 - fall back to the original if ffmpeg fails
            small = path
        data = small.read_bytes()
        if total + len(data) > MAX_TOTAL_BYTES:
            left_out.append(p)
            continue
        total += len(data)
        out.append(t.ImageContent(type="image",
                                  mime_type=mimetypes.guess_type(small.name)[0] or "image/png",
                                  data=base64.b64encode(data).decode()))
    return out, left_out


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
        images, left_out = _images(local.images_out, local.privacy)
        if left_out:
            res = {**res, "images_not_attached": left_out,
                   "images_note": "too many or too large for one reply: look at them with "
                                  "the Read tool, or ask for fewer"}
        content: list = [t.TextContent(type="text",
                                       text=json.dumps(res, indent=1, default=str))]
        content += images
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
