"""Thin, shell-free wrappers around ffmpeg/ffprobe."""

from .runner import FFmpegError, ffmpeg, ffprobe_json, run

__all__ = ["FFmpegError", "ffmpeg", "ffprobe_json", "run"]
