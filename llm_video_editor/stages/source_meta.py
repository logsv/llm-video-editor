"""Recording date, device and GPS position from the source files' container tags.

Phones write ISO 6709 coordinates (``+28.6139+077.2090+216.000/``) under
``location``, ``location-eng`` or ``com.apple.quicktime.location.ISO6709``.
Reverse geocoding (coordinates -> "New Delhi, Delhi, India") is opt-in: it sends
the coordinates to OpenStreetMap's Nominatim with a generic User-Agent.
"""

from __future__ import annotations

import re

import requests

from ..ffmpeg.runner import ffprobe_json
from ..schemas.publish import SourceMeta

LOCATION_KEYS = ("com.apple.quicktime.location.ISO6709", "location", "location-eng")
DEVICE_KEYS = (("com.apple.quicktime.make", "com.apple.quicktime.model"),
               ("com.android.manufacturer", "com.android.model"), ("make", "model"))
_ISO6709 = re.compile(r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?")
NOMINATIM = "https://nominatim.openstreetmap.org/reverse"


def parse_iso6709(value: str) -> tuple[float, float, float | None] | None:
    m = _ISO6709.match(value.strip())
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon, float(m.group(3)) if m.group(3) else None


def read_meta(path: str) -> SourceMeta:
    tags = {k.lower(): str(v) for k, v in
            (ffprobe_json(path).get("format", {}).get("tags") or {}).items()}
    lat = lon = alt = None
    for k in LOCATION_KEYS:
        if k.lower() in tags and (pos := parse_iso6709(tags[k.lower()])):
            lat, lon, alt = pos
            break
    device = None
    for make_k, model_k in DEVICE_KEYS:
        make, model = tags.get(make_k.lower(), ""), tags.get(model_k.lower(), "")
        if model:
            device = model if make.lower() in model.lower() else f"{make} {model}".strip()
            break
    return SourceMeta(recorded_at=tags.get("creation_time"), latitude=lat, longitude=lon,
                      altitude=alt, device=device)


def first_meta(paths: list[str]) -> SourceMeta:
    """Metadata of the first source that has GPS (else the first one)."""
    metas = [read_meta(p) for p in paths]
    return next((m for m in metas if m.latitude is not None), metas[0] if metas else SourceMeta())


def reverse_geocode(lat: float, lon: float, timeout: float = 15) -> str | None:
    """City-level place name via OpenStreetMap Nominatim (opt-in; one request)."""
    try:
        r = requests.get(NOMINATIM, timeout=timeout,
                         params={"lat": f"{lat:.4f}", "lon": f"{lon:.4f}", "format": "jsonv2",
                                 "zoom": 10, "accept-language": "en"},
                         headers={"User-Agent": "llmve/0.2 (video editor; reverse geocode)"})
        r.raise_for_status()
        addr = r.json().get("address", {})
    except (requests.RequestException, ValueError):
        return None
    city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("county")
    parts = [p for p in (city, addr.get("state"), addr.get("country")) if p]
    return ", ".join(dict.fromkeys(parts)) or None
