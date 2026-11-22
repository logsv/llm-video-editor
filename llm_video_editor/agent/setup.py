"""Machine record and the user's style profile.

* Machine record (`llmve setup`): what this computer can do - ffmpeg and its working
  encoders, the alpha-capable VP9 decoder, Node/HyperFrames, MCP, the Claude CLI,
  CPU count and free disk. The agent reads it before planning heavy work.
* Style profile (`~/.config/llmve/profile.json`, or `$LLMVE_CONFIG/profile.json`):
  channel, audience, languages, default look, likes and dislikes learned from the
  user's accepted notes, and the privacy mode. The agent reads it before every edit
  and appends to it when the user accepts a change ("no film grain"). ``privacy`` and
  ``location_lookup`` are the user's to set; the agent cannot change them.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..schemas import Strict
from ..workspace import atomic_write_text


def config_dir() -> Path:
    d = Path(os.environ.get("LLMVE_CONFIG") or Path.home() / ".config" / "llmve")
    d.mkdir(parents=True, exist_ok=True)
    return d


class Profile(Strict):
    channel: str = ""
    audience: str = ""
    languages: list[str] = Field(default_factory=lambda: ["en"])
    default_look: str = "natural"
    likes: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    privacy: Literal["stills", "text-only"] = "stills"
    # name places from the clips' GPS via OpenStreetMap (sends rounded coordinates)
    location_lookup: bool = False
    footage_dir: str = ""
    output_dir: str = ""
    updated_at: str = ""


def profile_path() -> Path:
    return config_dir() / "profile.json"


def load_profile() -> Profile:
    p = profile_path()
    return Profile.model_validate_json(p.read_text()) if p.exists() else Profile()


def save_profile(profile: Profile) -> Path:
    profile = profile.model_copy(update={"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    p = profile_path()
    atomic_write_text(p, profile.model_dump_json(indent=2))
    return p


def update_profile(*, set_fields: dict | None = None, add_likes: list[str] | None = None,
                   add_dislikes: list[str] | None = None,
                   remove: list[str] | None = None) -> Profile:
    prof = load_profile()
    data = prof.model_dump()
    for k, v in (set_fields or {}).items():
        if k not in Profile.model_fields or k in ("likes", "dislikes", "updated_at"):
            raise ValueError(f"cannot set {k!r}; settable: channel, audience, languages, "
                             "default_look, privacy, location_lookup, footage_dir, output_dir")
        if Profile.model_fields[k].annotation is bool and isinstance(v, str):
            if v.lower() not in ("true", "false", "yes", "no", "1", "0", "on", "off"):
                raise ValueError(f"{k} is true or false, got {v!r}")
            v = v.lower() in ("true", "yes", "1", "on")
        data[k] = v
    for item in add_likes or []:
        if item not in data["likes"]:
            data["likes"].append(item)
    for item in add_dislikes or []:
        if item not in data["dislikes"]:
            data["dislikes"].append(item)
    for item in remove or []:
        data["likes"] = [x for x in data["likes"] if x != item]
        data["dislikes"] = [x for x in data["dislikes"] if x != item]
    prof = Profile.model_validate(data)
    save_profile(prof)
    return prof


def machine_record() -> dict:
    """What this computer can do; written to <config>/machine.json."""
    from ..ffmpeg.encoders import HARDWARE, encoder_works
    from ..ffmpeg.runner import run
    from ..hyperframes import toolchain

    ff = shutil.which("ffmpeg")
    ff_version = run(["ffmpeg", "-version"], check=False, timeout=30).stdout.split("\n")[0] \
        if ff else None
    encoders = [e for e in HARDWARE if ff and encoder_works(e)] + (["libx264"] if ff else [])
    try:
        import mcp  # noqa: F401
        has_mcp = True
    except ImportError:
        has_mcp = False
    free = shutil.disk_usage(Path.home()).free
    rec = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "os": platform.system(), "machine": platform.machine(),
        "python": sys.version.split()[0], "cpus": os.cpu_count(),
        "ffmpeg": ff_version, "encoders": encoders,
        "libvpx_vp9_decoder": toolchain._has_libvpx(),
        "node": toolchain.doctor()["node"], "hyperframes": toolchain.toolchain().installed,
        "mcp": has_mcp, "claude_cli": shutil.which("claude") is not None,
        "free_disk_gb": round(free / 1e9, 1),
        "warnings": [],
    }
    if not ff:
        rec["warnings"].append("ffmpeg not found (macOS: brew install ffmpeg)")
    if not rec["libvpx_vp9_decoder"]:
        rec["warnings"].append("ffmpeg lacks libvpx-vp9: motion graphics would lose alpha")
    if not has_mcp:
        rec["warnings"].append("MCP not installed: uv sync --extra agent")
    if not rec["claude_cli"]:
        rec["warnings"].append("Claude Code CLI not found: needed for one-command agent runs")
    if rec["free_disk_gb"] < 20:
        rec["warnings"].append(f"only {rec['free_disk_gb']} GB free: renders and proxies need space")
    (config_dir() / "machine.json").write_text(json.dumps(rec, indent=2))
    return rec


def mcp_config(repo: Path, privacy: str | None = None) -> dict:
    """`.mcp.json` content that starts this tool server for Claude Code / `claude -p`."""
    # --directory: run from the repo, so `python -m` imports the package from it even if
    # the editable-install .pth is skipped (macOS hidden flag on .venv files)
    args = ["--directory", str(repo), "run", "--extra", "agent",
            "python", "-m", "llm_video_editor", "mcp"]
    if privacy:
        args += ["--privacy", privacy]
    server: dict = {"command": "uv", "args": args}
    if os.environ.get("LLMVE_CONFIG"):  # MCP clients do not pass the parent environment
        server["env"] = {"LLMVE_CONFIG": os.environ["LLMVE_CONFIG"]}
    return {"mcpServers": {"llmve": server}}
