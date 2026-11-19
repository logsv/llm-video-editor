"""Planner provider that shells out to ``claude -p`` (no API key needed).

The call runs in a fresh temporary directory so no repository CLAUDE.md,
settings or session history leaks into the planner, with every tool disabled
and session persistence off. The prompt goes in on stdin; the JSON envelope
comes back on stdout and ``structured_output`` is validated with Pydantic.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from typing import TypeVar

from pydantic import BaseModel

from .base import ProviderError, assert_text_only

M = TypeVar("M", bound=BaseModel)


class ClaudeCLIProvider:
    name = "claude-cli"

    def __init__(self, model: str | None = None, executable: str = "claude",
                 timeout: float = 600):
        self.model = model
        self.executable = executable
        self.timeout = timeout

    def build_argv(self, system: str, schema: type[BaseModel]) -> list[str]:
        argv = [
            self.executable, "-p",
            "--output-format", "json",
            "--json-schema", json.dumps(schema.model_json_schema()),
            "--tools", "",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--system-prompt", system,
        ]
        if self.model:
            argv += ["--model", self.model]
        return argv

    @staticmethod
    def parse_envelope(stdout: str) -> dict:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise ProviderError(f"claude-cli returned non-JSON output: {stdout[:300]!r}") from e
        if isinstance(data, list):  # some versions emit the whole message list
            data = next((d for d in reversed(data) if d.get("type") == "result"), {})
        if not isinstance(data, dict) or not data:
            raise ProviderError("claude-cli returned no result envelope")
        return data

    def complete(self, system: str, prompt: str, schema: type[M]) -> M:
        assert_text_only(system, prompt)
        if shutil.which(self.executable) is None:
            raise ProviderError(f"`{self.executable}` not found on PATH; install Claude Code "
                                "or choose another --provider")
        with tempfile.TemporaryDirectory(prefix="llmve-planner-") as cwd:
            try:
                proc = subprocess.run(
                    self.build_argv(system, schema), input=prompt, capture_output=True,
                    text=True, cwd=cwd, timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as e:
                raise ProviderError(f"claude-cli timed out after {self.timeout}s") from e
        if proc.returncode != 0 and not proc.stdout.strip():
            raise ProviderError(f"claude-cli exited {proc.returncode}: {proc.stderr[-500:]}")
        env = self.parse_envelope(proc.stdout)
        if env.get("is_error"):
            raise ProviderError(f"claude-cli reported an error: {str(env.get('result'))[:500]}")
        payload = env.get("structured_output")
        if payload is None:
            raise ProviderError("claude-cli envelope has no structured_output")
        return schema.model_validate(payload)
