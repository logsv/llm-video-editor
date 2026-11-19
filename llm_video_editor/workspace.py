"""Workspace layout, input resolution and content-hash caching.

Every batch of inputs (one file, a folder or a glob) maps to
``<home>/<batch_hash>/`` where ``batch_hash`` is derived from the sorted
``(path, mtime, size)`` of every source clip. Stages write their checkpoints
there and record a cache key in ``manifest.json`` so re-runs skip work whose
inputs did not change.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

MEDIA_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mxf"})
DEFAULT_HOME_ENV = "LLMVE_HOME"

M = TypeVar("M", bound=BaseModel)


def _sha1(data: str) -> str:
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def is_media(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_EXTENSIONS


def resolve_inputs(spec: str | os.PathLike[str] | Sequence[str | os.PathLike[str]]) -> list[Path]:
    """Resolve a file, a directory (recursive) or a glob into sorted media paths."""
    specs: Iterable[str | os.PathLike[str]] = (
        [spec] if isinstance(spec, str | os.PathLike) else spec
    )
    found: set[Path] = set()
    for item in specs:
        p = Path(item).expanduser()
        if p.is_file():
            if not is_media(p):
                raise ValueError(f"Not a supported video file: {p}")
            found.add(p.resolve())
        elif p.is_dir():
            found.update(q.resolve() for q in p.rglob("*") if q.is_file() and is_media(q))
        else:
            matches = [Path(m) for m in glob.glob(str(p), recursive=True)]
            found.update(m.resolve() for m in matches if m.is_file() and is_media(m))
    if not found:
        raise FileNotFoundError(f"No video files found for input: {spec}")
    return sorted(found)


@dataclass(frozen=True)
class SourceFingerprint:
    path: str
    mtime_ns: int
    size: int

    @classmethod
    def of(cls, path: Path) -> SourceFingerprint:
        st = path.stat()
        return cls(str(path.resolve()), st.st_mtime_ns, st.st_size)

    @property
    def key(self) -> str:
        return json.dumps([self.path, self.mtime_ns, self.size])

    @property
    def clip_id(self) -> str:
        """Stable, path-free identifier (safe to show to an LLM)."""
        return "v" + _sha1(self.key)[:8]


def batch_hash(files: Iterable[Path]) -> str:
    fps = sorted(SourceFingerprint.of(Path(f)).key for f in files)
    return _sha1("\n".join(fps))[:16]


def default_home() -> Path:
    return Path(os.environ.get(DEFAULT_HOME_ENV, Path.cwd() / ".llmve"))


class Workspace:
    """Directory holding every checkpoint for one batch of source clips."""

    def __init__(self, root: Path, sources: list[Path]):
        self.root = root
        self.sources = sources
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest = self._load_manifest()
        self._manifest["batch_hash"] = root.name
        self._manifest["inputs"] = [str(s) for s in sources]
        self._save_manifest()

    # -- construction -------------------------------------------------
    @classmethod
    def open(
        cls,
        inputs: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        home: Path | None = None,
    ) -> Workspace:
        sources = resolve_inputs(inputs)
        return cls((home or default_home()) / batch_hash(sources), sources)

    @classmethod
    def load(cls, root: Path) -> Workspace:
        """Re-open an existing workspace directory without re-resolving inputs."""
        data = json.loads((root / "manifest.json").read_text())
        return cls(root, [Path(p) for p in data.get("inputs", [])])

    @property
    def batch_hash(self) -> str:
        return self.root.name

    def fingerprints(self) -> list[SourceFingerprint]:
        return [SourceFingerprint.of(s) for s in self.sources]

    # -- well-known paths --------------------------------------------
    @property
    def transcript(self) -> Path:
        return self.root / "transcript.json"

    @property
    def clips_manifest(self) -> Path:
        return self.root / "clips_manifest.json"

    @property
    def sheets_dir(self) -> Path:
        return self.root / "sheets"

    @property
    def sheets_index(self) -> Path:
        return self.sheets_dir / "index.json"

    @property
    def asset_plan(self) -> Path:
        return self.root / "asset_plan.json"

    @property
    def edl_raw(self) -> Path:
        return self.root / "edl_raw.json"

    @property
    def cut_audit(self) -> Path:
        return self.root / "cut_audit.json"

    @property
    def edl_audited(self) -> Path:
        return self.root / "edl_audited.json"

    @property
    def broll_select(self) -> Path:
        return self.root / "broll_select.json"

    @property
    def geometry_map(self) -> Path:
        return self.root / "geometry_map.json"

    @property
    def review_dir(self) -> Path:
        return self.root / "review"

    @property
    def review(self) -> Path:
        return self.root / "review.json"

    @property
    def render_dir(self) -> Path:
        return self.root / "render"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def clip_dir(self, clip_id: str) -> Path:
        d = self.root / "clips" / clip_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def rel(self, path: Path) -> str:
        """Path relative to the workspace root, or absolute if it lives elsewhere."""
        resolved = Path(path).resolve()
        try:
            return str(resolved.relative_to(self.root.resolve()))
        except ValueError:
            return str(resolved)

    # -- checkpoint IO -----------------------------------------------
    @staticmethod
    def write(path: Path, model: BaseModel) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(model.model_dump_json(indent=2))
        tmp.replace(path)
        return path

    @staticmethod
    def read(path: Path, model: type[M]) -> M:
        return model.model_validate_json(Path(path).read_text())

    # -- stage cache -------------------------------------------------
    @staticmethod
    def stage_key(*parts: Any) -> str:
        return _sha1(json.dumps(parts, sort_keys=True, default=str))[:16]

    def is_fresh(self, stage: str, key: str) -> bool:
        rec = self._manifest.get("stages", {}).get(stage)
        if not rec or rec.get("key") != key:
            return False
        return all((self.root / o).exists() for o in rec.get("outputs", []))

    def record(self, stage: str, key: str, outputs: Iterable[Path]) -> None:
        self._manifest.setdefault("stages", {})[stage] = {
            "key": key,
            "outputs": [self.rel(o) for o in outputs],
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self._save_manifest()

    def invalidate(self, *stages: str) -> None:
        for s in stages:
            self._manifest.get("stages", {}).pop(s, None)
        self._save_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {"stages": {}}

    def _save_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._manifest, indent=2))
        tmp.replace(self.manifest_path)
