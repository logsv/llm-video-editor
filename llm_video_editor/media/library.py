"""Search, download and register royalty-free media, and keep its credits.

* ``search``  queries providers, writes the results to ``<ws>/media/search.json``.
* ``fetch``   downloads one result: audio into ``<ws>/media/``, video into a B-roll
  library folder with a ``<clip>.json`` sidecar (description, tags, credit) that
  ``llmve broll index`` reads.
* ``register`` records a file you got elsewhere (YouTube Audio Library / Creator
  Music, a purchased track) with its licence and credit line.

Every fetched or registered file lands in ``<ws>/media/credits.json``; the publish
stage turns it into the description's credits block.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from ..schemas.media import Credit, Credits, MediaKind, MediaResult, MediaSearch
from ..workspace import Workspace
from .providers import DEFAULT_FOR, PROVIDERS, USER_AGENT, ProviderUnavailable

MAX_BYTES = 1_000_000_000
AUDIO_EXT = {".mp3", ".wav", ".ogg", ".oga", ".flac", ".m4a", ".aac", ".opus"}
VIDEO_EXT = {".mp4", ".mov", ".webm", ".ogv", ".mkv", ".m4v"}


def media_dir(ws: Workspace) -> Path:
    d = ws.root / "media"
    d.mkdir(parents=True, exist_ok=True)
    return d


def credits_path(ws: Workspace) -> Path:
    return media_dir(ws) / "credits.json"


def load_credits(ws: Workspace) -> Credits:
    p = credits_path(ws)
    return Workspace.read(p, Credits) if p.exists() else Credits()


def _save_credit(ws: Workspace, credit: Credit) -> None:
    cr = load_credits(ws)
    items = [c for c in cr.items if c.file != credit.file] + [credit]
    Workspace.write(credits_path(ws), Credits(items=items))


def search(query: str, kind: MediaKind, *, providers: tuple[str, ...] | None = None,
           limit: int = 10, length: str | None = None, ws: Workspace | None = None
           ) -> MediaSearch:
    results, errors = [], []
    for name in providers or DEFAULT_FOR[kind]:
        fn = PROVIDERS.get(name)
        if fn is None:
            errors.append(f"{name}: unknown provider (have {', '.join(PROVIDERS)})")
            continue
        try:
            results += fn(query, kind, limit=limit, length=length)[:limit]
        except ProviderUnavailable as e:
            errors.append(f"{name}: {e}")
    res = MediaSearch(query=query, kind=kind, results=results, errors=errors)
    if ws is not None:
        Workspace.write(media_dir(ws) / "search.json", res)
    return res


def last_search(ws: Workspace) -> MediaSearch:
    p = media_dir(ws) / "search.json"
    if not p.exists():
        raise FileNotFoundError("no search yet: run `llmve media search <ws> ...` first")
    return Workspace.read(p, MediaSearch)


def _ext(url: str, content_type: str, kind: MediaKind) -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    allowed = VIDEO_EXT if kind == "video" else AUDIO_EXT
    if ext in allowed:
        return ext
    guess = {"audio/mpeg": ".mp3", "audio/ogg": ".ogg", "audio/wav": ".wav", "audio/x-wav": ".wav",
             "audio/flac": ".flac", "video/mp4": ".mp4", "video/webm": ".webm",
             "video/ogg": ".ogv", "video/quicktime": ".mov"}.get(content_type.split(";")[0])
    return guess or (".mp4" if kind == "video" else ".mp3")


def download(url: str, dest_stem: Path, kind: MediaKind) -> Path:
    """Stream ``url`` to ``dest_stem`` + the right extension; refuses non-media responses."""
    try:
        with requests.get(url, stream=True, timeout=60, headers={"User-Agent": USER_AGENT}) as r:
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "application/octet-stream")
            if not (ctype.startswith(("audio/", "video/", "application/octet-stream",
                                      "application/ogg"))):
                raise ProviderUnavailable(f"{url} is {ctype}, not audio/video")
            out = dest_stem.with_suffix(_ext(url, ctype, kind))
            tmp = out.with_suffix(out.suffix + ".part")
            size = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ProviderUnavailable(f"{url} is larger than {MAX_BYTES >> 20} MB")
                    f.write(chunk)
    except requests.RequestException as e:
        raise ProviderUnavailable(f"download failed: {e}") from e
    tmp.replace(out)
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "clip"


def credit_of(r: MediaResult, path: Path) -> Credit:
    return Credit(file=str(path.resolve()), kind=r.kind, title=r.title, creator=r.creator,
                  provider=r.provider, license=r.license, license_url=r.license_url,
                  landing_url=r.landing_url, attribution=r.attribution,
                  attribution_required=r.attribution_required,
                  added_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))


def fetch(ws: Workspace, result: MediaResult, *, library: Path | None = None,
          description: str | None = None) -> Path:
    """Download one result and record its credit (video also gets a B-roll sidecar)."""
    stem = f"{result.provider}-{result.result_id.split(':', 1)[1]}-{_slug(result.title)}"
    if result.kind == "video":
        lib = library or media_dir(ws) / "broll"
        lib.mkdir(parents=True, exist_ok=True)
        path = download(result.download_url, lib / stem, "video")
        credit = credit_of(result, path)
        path.with_suffix(".json").write_text(json.dumps({
            "description": description or result.title,
            "tags": result.tags[:12],
            "credit": credit.model_dump(),
        }, indent=2), encoding="utf-8")
    else:
        path = download(result.download_url, media_dir(ws) / stem, result.kind)
        credit = credit_of(result, path)
    _save_credit(ws, credit)
    return path


def register(ws: Workspace, file: Path, *, kind: MediaKind, title: str, license: str,
             provider: str = "manual", creator: str = "", attribution: str = "",
             license_url: str = "", landing_url: str = "", copy: bool = True) -> Credit:
    """Record a file obtained outside the search (e.g. YouTube Audio Library)."""
    src = file.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    dest = src
    if copy:
        dest = media_dir(ws) / f"{provider}-{_slug(title)}{src.suffix.lower()}"
        if dest != src:
            shutil.copyfile(src, dest)
    credit = Credit(file=str(dest), kind=kind, title=title, creator=creator, provider=provider,
                    license=license, license_url=license_url, landing_url=landing_url,
                    attribution=attribution, attribution_required=bool(attribution),
                    added_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    _save_credit(ws, credit)
    return credit


def sidecar_credit(path: str | Path) -> Credit | None:
    """Credit stored next to a downloaded B-roll clip, if any."""
    side = Path(path).with_suffix(".json")
    if not side.exists():
        return None
    try:
        data = json.loads(side.read_text(encoding="utf-8")).get("credit")
        return Credit.model_validate(data) if data else None
    except (ValueError, TypeError):
        return None
