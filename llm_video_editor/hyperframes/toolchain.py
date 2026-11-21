"""Pinned HyperFrames toolchain: install once into a cache folder, run without a shell.

HyperFrames (Apache-2.0, https://github.com/heygen-com/hyperframes) renders HTML +
GSAP compositions to video in headless Chrome. We pin the CLI and GSAP versions,
install them with npm into ``$LLMVE_TOOLS/hyperframes-<version>`` (default
``~/.cache/llmve``), and call ``node <bin>`` directly with argv lists.

Environment set for every call:
  * telemetry and update checks off (``HYPERFRAMES_NO_TELEMETRY``, ``DO_NOT_TRACK``,
    ``HYPERFRAMES_NO_UPDATE_CHECK``) - nothing leaves the machine;
  * ``npm_config_onnxruntime_node_install_cuda=skip`` so ``npm install`` does not try to
    download CUDA binaries (only background removal would use them);
  * ``HYPERFRAMES_BROWSER_PATH`` is passed through when set, so an existing Chrome /
    Chromium is used instead of a download.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..ffmpeg.runner import FFmpegError, run

HYPERFRAMES_VERSION = "0.8.34"
GSAP_VERSION = "3.13.0"
MIN_NODE_MAJOR = 22


class HyperFramesUnavailable(RuntimeError):
    """Node 22+/npm missing, or the pinned toolchain could not be installed."""


def tools_home() -> Path:
    return Path(os.environ.get("LLMVE_TOOLS") or Path.home() / ".cache" / "llmve")


@dataclass(frozen=True)
class Toolchain:
    root: Path  # folder holding node_modules/

    @property
    def cli(self) -> Path:
        return self.root / "node_modules" / "hyperframes" / "bin" / "hyperframes.mjs"

    @property
    def gsap(self) -> Path:
        return self.root / "node_modules" / "gsap" / "dist" / "gsap.min.js"

    @property
    def installed(self) -> bool:
        return self.cli.exists() and self.gsap.exists()


def env() -> dict[str, str]:
    e = dict(os.environ)
    e.update({
        "HYPERFRAMES_NO_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "HYPERFRAMES_NO_UPDATE_CHECK": "1",
        "HYPERFRAMES_SKIP_SKILLS": "1",
        "npm_config_onnxruntime_node_install_cuda": "skip",
        "npm_config_fund": "false",
        "npm_config_audit": "false",
    })
    return e


def node_major() -> int | None:
    if shutil.which("node") is None:
        return None
    out = run(["node", "--version"], check=False, timeout=30).stdout.strip()
    m = re.match(r"v(\d+)", out)
    return int(m.group(1)) if m else None


def toolchain() -> Toolchain:
    return Toolchain(tools_home() / f"hyperframes-{HYPERFRAMES_VERSION}")


def ensure(timeout: float = 900) -> Toolchain:
    """Return the installed toolchain, installing it with npm on first use."""
    major = node_major()
    if major is None or major < MIN_NODE_MAJOR:
        found = f"found Node {major}" if major else "Node not found"
        raise HyperFramesUnavailable(
            f"HyperFrames needs Node.js {MIN_NODE_MAJOR}+ ({found}). "
            "Install it from https://nodejs.org or with your package manager.")
    tc = toolchain()
    if tc.installed:
        return tc
    if shutil.which("npm") is None:
        raise HyperFramesUnavailable("npm not found on PATH (it ships with Node.js)")
    tc.root.mkdir(parents=True, exist_ok=True)
    (tc.root / "package.json").write_text(json.dumps(
        {"name": "llmve-hyperframes", "private": True}), encoding="utf-8")
    try:
        _run(["npm", "install", "--no-save", "--loglevel=error",
              f"hyperframes@{HYPERFRAMES_VERSION}", f"gsap@{GSAP_VERSION}"],
             cwd=tc.root, timeout=timeout)
    except FFmpegError as e:
        raise HyperFramesUnavailable(f"npm install of hyperframes@{HYPERFRAMES_VERSION} "
                                     f"failed:\n{e}") from e
    if not tc.installed:
        raise HyperFramesUnavailable(f"npm finished but {tc.cli} is missing")
    return tc


def _run(argv: list[str | Path], *, cwd: Path, timeout: float | None, check: bool = True):
    return run(argv, cwd=cwd, timeout=timeout, env=env(), check=check)


def hyperframes(*args: str | Path, cwd: Path, timeout: float | None = 600, check: bool = True):
    """``node <hyperframes.mjs> <args>`` in ``cwd`` (a composition folder)."""
    tc = ensure()
    return _run(["node", tc.cli, *args], cwd=cwd, timeout=timeout, check=check)


def doctor() -> dict[str, str]:
    """What is installed, for `llmve graphics doctor`."""
    major = node_major()
    tc = toolchain()
    return {
        "node": f"v{major}" if major else "missing",
        "node_ok": str(bool(major and major >= MIN_NODE_MAJOR)),
        "npm": shutil.which("npm") or "missing",
        "hyperframes": f"{HYPERFRAMES_VERSION} at {tc.root}" if tc.installed else "not installed",
        "browser": os.environ.get("HYPERFRAMES_BROWSER_PATH") or "HyperFrames default",
        "ffmpeg_libvpx_vp9": str(_has_libvpx()),
    }


def _has_libvpx() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    out = run(["ffmpeg", "-hide_banner", "-decoders"], check=False, timeout=30).stdout
    return "libvpx-vp9" in out
