"""Search royalty-free media APIs and normalise results to ``MediaResult``.

| provider  | kinds        | key                         | licence kept                          |
|-----------|--------------|-----------------------------|---------------------------------------|
| openverse | music, sfx   | none (anonymous, rate-limited) | CC0, CC BY, CC BY-SA (commercial use and modification allowed) |
| pixabay   | video        | ``PIXABAY_API_KEY``          | Pixabay Content License               |
| pexels    | video        | ``PEXELS_API_KEY``           | Pexels License                        |
| wikimedia | video        | none                         | CC0, public domain, CC BY, CC BY-SA   |

Non-commercial (NC) and no-derivatives (ND) licences are filtered out because an
edited, possibly monetised YouTube video needs both rights. YouTube's own Audio
Library / Creator Music has no public API: pick tracks there and register them
with ``llmve media add``.

Requests carry a generic User-Agent and only the search words; nothing about the
user or their footage is sent.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable

import requests

from ..schemas.media import MediaKind, MediaResult

USER_AGENT = "llmve/0.2 (royalty-free media search)"
TIMEOUT = 20

OPENVERSE = "https://api.openverse.org/v1/audio/"
PIXABAY = "https://pixabay.com/api/videos/"
PEXELS = "https://api.pexels.com/videos/search"
COMMONS = "https://commons.wikimedia.org/w/api.php"


class ProviderUnavailable(RuntimeError):
    """Missing API key, network failure or an unexpected response."""


def _get(url: str, *, params: dict, headers: dict | None = None) -> dict:
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT,
                         headers={"User-Agent": USER_AGENT, **(headers or {})})
    except requests.RequestException as e:
        raise ProviderUnavailable(f"{url}: {e}") from e
    if r.status_code in (401, 403):
        raise ProviderUnavailable(f"{url}: HTTP {r.status_code} (check the API key)")
    if r.status_code == 429:
        raise ProviderUnavailable(f"{url}: rate limited (HTTP 429); try again later")
    if r.status_code >= 400:
        raise ProviderUnavailable(f"{url}: HTTP {r.status_code}")
    try:
        return r.json()
    except ValueError as e:
        raise ProviderUnavailable(f"{url}: response was not JSON") from e


# -- Openverse (CC audio: Jamendo, Freesound, Wikimedia, ...) ------------------------------
def _cc_license(code: str, version: str) -> str:
    return f"{code} {version}".strip() if code else "unknown"


def parse_openverse(data: dict, kind: MediaKind) -> list[MediaResult]:
    out = []
    for r in data.get("results", []):
        url = r.get("url")
        if not url:
            continue
        code = str(r.get("license", "")).lower()
        if "nc" in code.split("-") or "nd" in code.split("-"):
            continue  # belt and braces: the query already asks for commercial+modification
        dur = r.get("duration")
        tags = [t.get("name", "") if isinstance(t, dict) else str(t) for t in r.get("tags") or []]
        out.append(MediaResult(
            result_id=f"openverse:{r.get('id')}", provider="openverse", kind=kind,
            title=str(r.get("title") or "Untitled"), creator=str(r.get("creator") or ""),
            creator_url=str(r.get("creator_url") or ""),
            license=_cc_license(code, str(r.get("license_version") or "")),
            license_url=str(r.get("license_url") or ""),
            attribution=str(r.get("attribution") or ""),
            attribution_required=code not in ("cc0", "pdm"),
            landing_url=str(r.get("foreign_landing_url") or ""), download_url=str(url),
            preview_url=str(url), duration=round(dur / 1000, 2) if isinstance(dur, int | float)
            else None, tags=[t for t in tags if t][:12]))
    return out


def openverse(query: str, kind: MediaKind, *, limit: int = 20,
              length: str | None = None) -> list[MediaResult]:
    if kind == "video":
        raise ProviderUnavailable("openverse has no video search")
    params = {"q": query, "page_size": min(limit, 20),
              "license_type": "commercial,modification",
              "category": "music" if kind == "music" else "sound_effect"}
    if length:
        params["length"] = length  # short | medium | long
    return parse_openverse(_get(OPENVERSE, params=params), kind)


# -- Pixabay (stock video) ------------------------------------------------------------------
def parse_pixabay(data: dict) -> list[MediaResult]:
    out = []
    for h in data.get("hits", []):
        files = h.get("videos") or {}
        best = next((files[k] for k in ("large", "medium", "small", "tiny")
                     if isinstance(files.get(k), dict) and files[k].get("url")), None)
        if best is None:
            continue
        user = str(h.get("user") or "")
        out.append(MediaResult(
            result_id=f"pixabay:{h.get('id')}", provider="pixabay", kind="video",
            title=str(h.get("tags") or "Pixabay video").split(",")[0].strip().title(),
            creator=user, license="pixabay",
            license_url="https://pixabay.com/service/license-summary/",
            attribution=f"Video by {user} from Pixabay" if user else "Video from Pixabay",
            attribution_required=False, landing_url=str(h.get("pageURL") or ""),
            download_url=str(best["url"]), preview_url=str(best.get("thumbnail") or ""),
            duration=float(h["duration"]) if h.get("duration") else None,
            width=best.get("width"), height=best.get("height"),
            tags=[t.strip() for t in str(h.get("tags") or "").split(",") if t.strip()]))
    return out


def pixabay(query: str, kind: MediaKind, *, limit: int = 20, **_) -> list[MediaResult]:
    if kind != "video":
        raise ProviderUnavailable("pixabay's API searches images and videos only")
    key = os.environ.get("PIXABAY_API_KEY")
    if not key:
        raise ProviderUnavailable("set PIXABAY_API_KEY (free at pixabay.com/api/docs)")
    return parse_pixabay(_get(PIXABAY, params={"key": key, "q": query[:100], "safesearch": "true",
                                               "per_page": max(3, min(limit, 200))}))


# -- Pexels (stock video) -------------------------------------------------------------------
def parse_pexels(data: dict) -> list[MediaResult]:
    out = []
    for v in data.get("videos", []):
        files = [f for f in v.get("video_files") or [] if f.get("link")
                 and str(f.get("file_type", "video/mp4")).startswith("video/mp4")]
        if not files:
            continue
        # largest file up to 1920 px wide (4K stock is rarely worth the download),
        # otherwise the smallest one offered
        def width(f: dict) -> int:
            return int(f.get("width") or 0)

        hd = [f for f in files if width(f) <= 1920]
        best = max(hd, key=width) if hd else min(files, key=width)
        user = v.get("user") or {}
        name = str(user.get("name") or "")
        out.append(MediaResult(
            result_id=f"pexels:{v.get('id')}", provider="pexels", kind="video",
            title=_pexels_title(str(v.get("url") or "")), creator=name,
            creator_url=str(user.get("url") or ""), license="pexels",
            license_url="https://www.pexels.com/license/",
            attribution=f"Video by {name} from Pexels" if name else "Video from Pexels",
            attribution_required=False, landing_url=str(v.get("url") or ""),
            download_url=str(best["link"]), preview_url=str(v.get("image") or ""),
            duration=float(v["duration"]) if v.get("duration") else None,
            width=best.get("width"), height=best.get("height")))
    return out


def _pexels_title(url: str) -> str:
    # https://www.pexels.com/video/people-walking-on-a-street-854100/ -> People walking on a street
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    words = [w for w in slug.split("-") if not w.isdigit()]
    return " ".join(words).capitalize() or "Pexels video"


def pexels(query: str, kind: MediaKind, *, limit: int = 20, **_) -> list[MediaResult]:
    if kind != "video":
        raise ProviderUnavailable("pexels has no audio")
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        raise ProviderUnavailable("set PEXELS_API_KEY (free at pexels.com/api)")
    return parse_pexels(_get(PEXELS, params={"query": query, "per_page": min(limit, 80)},
                             headers={"Authorization": key}))


# -- Wikimedia Commons (CC / public-domain video) ------------------------------------------
_TAG = re.compile(r"<[^>]+>")


def _plain(html: str) -> str:
    return _TAG.sub("", html or "").strip()


def commons_license_ok(short: str) -> bool:
    s = short.lower().replace("-", " ")
    if " nc" in f" {s}" or " nd" in f" {s}" or "fair use" in s:
        return False
    return s.startswith(("cc0", "cc by", "public domain", "pd")) or s in ("cc zero",)


def parse_commons(data: dict) -> list[MediaResult]:
    out = []
    pages = (data.get("query") or {}).get("pages") or {}
    for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        short = _plain((meta.get("LicenseShortName") or {}).get("value", ""))
        if not info.get("url") or not commons_license_ok(short):
            continue
        artist = _plain((meta.get("Artist") or {}).get("value", ""))
        required = str((meta.get("AttributionRequired") or {}).get("value", "true")) != "false"
        title = str(page.get("title", "")).removeprefix("File:").rsplit(".", 1)[0]
        out.append(MediaResult(
            result_id=f"wikimedia:{page.get('pageid')}", provider="wikimedia", kind="video",
            title=title.replace("_", " ") or "Wikimedia Commons video", creator=artist,
            license=short, license_url=_plain((meta.get("LicenseUrl") or {}).get("value", "")),
            attribution=(f"{title} by {artist}, {short}, via Wikimedia Commons" if artist
                         else f"{title}, {short}, via Wikimedia Commons"),
            attribution_required=required and not short.lower().startswith(("cc0", "public")),
            landing_url=str(info.get("descriptionurl") or ""), download_url=str(info["url"]),
            preview_url=str(info.get("thumburl") or ""),
            duration=float(info["duration"]) if info.get("duration") else None,
            width=info.get("width"), height=info.get("height")))
    return out


def wikimedia(query: str, kind: MediaKind, *, limit: int = 20, **_) -> list[MediaResult]:
    if kind != "video":
        raise ProviderUnavailable("wikimedia provider searches video only")
    return parse_commons(_get(COMMONS, params={
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f"{query} filetype:video", "gsrlimit": min(limit, 50), "prop": "imageinfo",
        "iiprop": "url|size|extmetadata|mediatype", "iiurlwidth": 480}))


PROVIDERS: dict[str, Callable[..., list[MediaResult]]] = {
    "openverse": openverse, "pixabay": pixabay, "pexels": pexels, "wikimedia": wikimedia}
DEFAULT_FOR: dict[str, tuple[str, ...]] = {
    "music": ("openverse",), "sfx": ("openverse",),
    "video": ("pixabay", "pexels", "wikimedia")}
