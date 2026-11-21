"""Versioned edits and the decision log: the agent's memory of what it did and why.

Every change to the edit is a new numbered version under ``<ws>/edits/vNNNN.json``;
``edits/HEAD`` names the current one and ``edl_raw.json`` mirrors it so every existing
stage keeps working. History is linear: undo creates a new version that restores an
old one, so nothing is ever lost.

* ``agent_log.jsonl``: one JSON line per tool call (edit, look, check), with its
  parameters, the agent's reason, the versions before/after and a result summary.
* ``decisions.md``: the same, as readable lines for the user.

Concurrency: a writer may pass ``expected_version``; if the edit moved on since it
read it (another tool call, or the user editing ``edl_raw.json`` by hand), the write
is rejected with ``EditConflict`` instead of silently overwriting. Hand edits of
``edl_raw.json`` are picked up and recorded as a ``manual edit`` version.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schemas import EDL
from ..workspace import Workspace


class EditConflict(RuntimeError):
    """The edit changed since the caller read it."""


class EditError(ValueError):
    """An edit operation was invalid; the message is written for the agent to act on."""


@dataclass(frozen=True)
class Change:
    version: int
    previous: int
    changed: bool
    summary: str


def _hash(edl: EDL) -> str:
    return hashlib.sha1(edl.model_dump_json().encode()).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class EditStore:
    def __init__(self, ws: Workspace):
        self.ws = ws
        self.dir = ws.root / "edits"
        self.log_path = ws.root / "agent_log.jsonl"
        self.decisions_path = ws.root / "decisions.md"

    # -- versions -------------------------------------------------------------
    @property
    def head_path(self) -> Path:
        return self.dir / "HEAD"

    def head(self) -> int:
        return int(self.head_path.read_text().strip()) if self.head_path.exists() else 0

    def path(self, version: int) -> Path:
        return self.dir / f"v{version:04d}.json"

    def load(self, version: int) -> EDL:
        if not self.path(version).exists():
            raise EditError(f"version {version} does not exist (latest is {self.head()})")
        return Workspace.read(self.path(version), EDL)

    def _write_version(self, edl: EDL) -> int:
        n = self.head() + 1
        Workspace.write(self.path(n), edl)
        tmp = self.head_path.with_suffix(".tmp")
        tmp.write_text(str(n))
        tmp.replace(self.head_path)
        Workspace.write(self.ws.edl_raw, edl)  # existing stages read this
        return n

    def current(self) -> EDL:
        """The current edit, picking up a first import or a hand edit of edl_raw.json."""
        raw = Workspace.read(self.ws.edl_raw, EDL) if self.ws.edl_raw.exists() else None
        n = self.head()
        if n == 0:
            if raw is None:
                raise EditError("no edit yet: plan one (`plan` tool) or create clips first")
            v = self._write_version(raw)
            self._log("edit", "import", {}, "first version of the edit (from edl_raw.json)", 0, v,
                      f"{len(raw.clips)} clips, {raw.total_duration:.1f}s")
            return raw
        head = self.load(n)
        if raw is not None and _hash(raw) != _hash(head):
            v = self._write_version(raw)
            self._log("edit", "manual edit", {}, "edl_raw.json was changed outside the tools",
                      n, v, f"{len(raw.clips)} clips, {raw.total_duration:.1f}s")
            return raw
        return head

    def apply(self, tool: str, params: dict[str, Any], reason: str,
              fn: Callable[[EDL], EDL], *, expected_version: int | None = None,
              summary: Callable[[EDL, EDL], str] | None = None) -> Change:
        """Run ``fn`` on the current edit and save the result as a new version."""
        before = self.current()
        prev = self.head()
        if expected_version is not None and expected_version != prev:
            raise EditConflict(f"the edit is at version {prev}, not {expected_version}: "
                               "read it again (edit_state) and redo the change on top of it")
        after = EDL.model_validate(fn(before).model_dump())  # schema-checked
        text = summary(before, after) if summary else describe_change(before, after)
        if _hash(after) == _hash(before):
            self._log("edit", tool, params, reason, prev, prev, "no change")
            return Change(prev, prev, False, "no change")
        v = self._write_version(after)
        self._log("edit", tool, params, reason, prev, v, text)
        return Change(v, prev, True, text)

    def undo(self, to_version: int, reason: str) -> Change:
        target = self.load(to_version)
        return self.apply("undo", {"to_version": to_version}, reason, lambda _: target,
                          summary=lambda b, a: f"restored version {to_version}")

    # -- log ------------------------------------------------------------------
    def note(self, kind: str, tool: str, params: dict[str, Any], reason: str,
             summary: str) -> None:
        """Record a non-edit tool call (look / check / delivery) in the decision log."""
        v = self.head()
        self._log(kind, tool, params, reason, v, v, summary)

    def _log(self, kind: str, tool: str, params: dict[str, Any], reason: str, before: int,
             after: int, summary: str) -> None:
        entry = {"at": _now(), "kind": kind, "tool": tool, "params": params,
                 "reason": reason, "version_before": before, "version_after": after,
                 "summary": summary}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
        first = not self.decisions_path.exists()
        with open(self.decisions_path, "a", encoding="utf-8") as f:
            if first:
                f.write("# Decisions\n\nEvery change to the edit and every check, with the "
                        "reason given.\n\n")
            ver = f"v{after:04d}" if after != before else f"(v{after:04d})"
            why = f" - {reason}" if reason else ""
            f.write(f"- {entry['at'][:19]} {ver} **{tool}**: {summary}{why}\n")

    def history(self, limit: int = 50, kind: str | None = None) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        rows = [json.loads(line) for line in self.log_path.read_text(encoding="utf-8")
                .splitlines() if line.strip()]
        if kind:
            rows = [r for r in rows if r["kind"] == kind]
        return rows[-limit:]


def describe_change(before: EDL, after: EDL) -> str:
    """Short human summary of what changed between two edits."""
    parts = []
    b_ids = [c.clip_id for c in before.clips]
    a_ids = [c.clip_id for c in after.clips]
    added = [i for i in a_ids if i not in b_ids]
    removed = [i for i in b_ids if i not in a_ids]
    if added:
        parts.append(f"added {', '.join(added)}")
    if removed:
        parts.append(f"removed {', '.join(removed)}")
    if not added and not removed and a_ids != b_ids:
        parts.append("reordered clips")
    b_map = {c.clip_id: c for c in before.clips}
    changed = [c.clip_id for c in after.clips if c.clip_id in b_map and c != b_map[c.clip_id]]
    if changed:
        parts.append(f"changed {', '.join(changed[:6])}" + ("..." if len(changed) > 6 else ""))
    for field in ("pacing", "captions", "mix", "overlays", "graphics", "music",
                  "target_duration", "fit_mode"):
        if getattr(before, field, None) != getattr(after, field, None):
            parts.append(f"{field} updated")
    parts.append(f"now {len(after.clips)} clips, {after.total_duration:.1f}s")
    return "; ".join(parts)
