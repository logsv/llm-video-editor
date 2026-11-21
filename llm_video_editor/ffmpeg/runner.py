"""Run ffmpeg/ffprobe without a shell, with timeouts and readable errors."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path


class FFmpegError(RuntimeError):
    def __init__(self, message: str, argv: Sequence[str] = (), stderr: str = "", code: int = -1):
        self.argv = list(argv)
        self.stderr = stderr
        self.code = code
        tail = "\n".join(stderr.strip().splitlines()[-15:])
        super().__init__(f"{message}\n{tail}" if tail else message)


def run(
    argv: Sequence[str | Path],
    *,
    timeout: float | None = None,
    check: bool = True,
    input_text: str | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` (a list, never a shell string) and capture text output."""
    if isinstance(argv, str | bytes):
        raise TypeError("argv must be a sequence of arguments, not a shell string")
    args = [str(a) for a in argv]
    if not args:
        raise ValueError("empty argv")
    if shutil.which(args[0], path=(env or {}).get("PATH")) is None:
        raise FFmpegError(f"Executable not found on PATH: {args[0]}", args)
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, input=input_text, cwd=cwd,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise FFmpegError(f"Timed out after {timeout}s: {args[0]}", args) from e
    if check and proc.returncode != 0:
        raise FFmpegError(
            f"{args[0]} exited with code {proc.returncode}", args, proc.stderr, proc.returncode
        )
    return proc


def ffmpeg(*args: str | Path, timeout: float | None = None, check: bool = True,
           cwd: str | Path | None = None):
    """``ffmpeg -hide_banner -nostdin -y <args>``."""
    return run(["ffmpeg", "-hide_banner", "-nostdin", "-y", *args], timeout=timeout, check=check,
               cwd=cwd)


def ffprobe_json(path: str | Path, *extra: str, timeout: float = 60) -> dict:
    """ffprobe as JSON; ``extra`` adds sections, e.g. ``"-show_chapters"``."""
    proc = run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", *extra, str(path),
        ],
        timeout=timeout,
    )
    return json.loads(proc.stdout)


def ffmpeg_bytes(*args: str | Path, timeout: float | None = 120) -> bytes:
    """Run ffmpeg and return raw stdout bytes (e.g. ``-f rawvideo -``)."""
    argv = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", *map(str, args)]
    if shutil.which("ffmpeg") is None:
        raise FFmpegError("Executable not found on PATH: ffmpeg", argv)
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise FFmpegError(f"Timed out after {timeout}s: ffmpeg", argv) from e
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg exited with code {proc.returncode}", argv,
                          proc.stderr.decode(errors="replace"), proc.returncode)
    return proc.stdout
