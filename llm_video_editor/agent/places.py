"""Where each clip was filmed: the route from the cameras' GPS and clock.

``route`` works offline: clips in recording order with coordinates, the distance
from the first clip and from the previous one, grouped into *stops* (consecutive
clips within ``stop_radius_km`` of each other). It names nothing by itself.

``name_stops`` turns stops into place names (town, nearest named feature such as a
temple or fort, road, district, state) through OpenStreetMap Nominatim, and
``nearby_stops`` lists named things around each stop (temples, forts, forests,
national parks, rivers, highways such as NH-39) through the Overpass API. Both send
the stops' coordinates (only those, rounded to about 100 m) to OpenStreetMap
services, so they run only when the user has allowed it (``llmve profile --set
location_lookup=true``); the agent cannot switch it on. Answers are cached in
``<ws>/places.json`` and requests are spaced one second apart (Nominatim's policy).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import requests

NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "llmve/0.3 (video editor; place titles)"
MAX_LOOKUPS = 40  # stops named per call (1 request/s, two zoom levels each)


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _key(lat: float, lon: float) -> str:
    return f"{lat:.3f},{lon:.3f}"  # ~100 m: the precision that is sent and cached


def route(footage: dict[str, dict], clip_ids: dict[str, str],
          stop_radius_km: float = 1.0) -> dict[str, Any]:
    """Clips in recording order with distances, grouped into stops."""
    rows = []
    for path, f in footage.items():
        if not f.get("ok", True):
            continue
        rows.append({"clip_id": clip_ids.get(path), "file": f.get("name") or Path(path).name,
                     "recorded_at": f.get("recorded_at"), "duration": f.get("duration"),
                     "lat": f.get("latitude"), "lon": f.get("longitude")})
    rows.sort(key=lambda r: (r["recorded_at"] or "", r["file"]))
    first = prev = None
    total = 0.0
    stops: list[dict] = []
    for r in rows:
        if r["lat"] is None or r["lon"] is None:
            r["km_from_start"] = r["km_from_previous"] = None
            continue
        here = (r["lat"], r["lon"])
        first = first or here
        step = haversine_km(prev, here) if prev else 0.0
        total += step
        r["km_from_previous"] = round(step, 2)
        r["km_from_start"] = round(total, 1)  # along the filmed points, not the road
        prev = here
        last = stops[-1] if stops else None
        if last and haversine_km((last["lat"], last["lon"]), here) <= stop_radius_km:
            last["clips"].append(r["clip_id"] or r["file"])
            last["until"] = r["recorded_at"]
        else:
            stops.append({"stop": len(stops) + 1, "lat": r["lat"], "lon": r["lon"],
                          "key": _key(r["lat"], r["lon"]), "km_from_start": r["km_from_start"],
                          "from": r["recorded_at"], "until": r["recorded_at"],
                          "clips": [r["clip_id"] or r["file"]]})
    with_gps = sum(1 for r in rows if r["lat"] is not None)
    return {"clips": rows, "stops": stops, "clips_with_gps": with_gps,
            "clips_without_gps": len(rows) - with_gps,
            "km_between_filmed_points": round(total, 1)}


def _reverse(lat: float, lon: float, zoom: int, timeout: float = 15) -> dict:
    r = requests.get(NOMINATIM, timeout=timeout,
                     params={"lat": f"{lat:.3f}", "lon": f"{lon:.3f}", "format": "jsonv2",
                             "zoom": zoom, "accept-language": "en", "namedetails": 0},
                     headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.json()


def describe_place(area: dict, nearest: dict) -> dict:
    """Pick the useful names out of two Nominatim answers (town level and nearest object)."""
    a = area.get("address") or {}
    n = nearest.get("address") or {}
    town = (a.get("city") or a.get("town") or a.get("village") or a.get("hamlet")
            or n.get("city") or n.get("town") or n.get("village") or n.get("hamlet"))
    feature = nearest.get("name") or None
    kind = nearest.get("type") or nearest.get("category")
    if feature and feature == town:
        feature = None
    return {"town": town, "nearest_named": feature, "nearest_kind": kind if feature else None,
            "road": n.get("road"), "district": a.get("state_district") or a.get("county"),
            "state": a.get("state"), "country": a.get("country"),
            "label": ", ".join(dict.fromkeys(p for p in (feature, town, a.get("state")) if p))}


def name_stops(stops: list[dict], cache_path: Path, *, max_lookups: int = MAX_LOOKUPS,
               reverse=None, sleep=None) -> tuple[list[dict], list[str]]:
    """Add ``place`` to each stop (cached); returns (stops, errors)."""
    reverse, sleep = reverse or _reverse, sleep or time.sleep
    cache: dict[str, dict] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    errors: list[str] = []
    looked = 0
    for s in stops:
        hit = cache.get(s["key"])
        if hit is None and looked < max_lookups:
            try:
                if looked:
                    sleep(1.1)
                area = reverse(s["lat"], s["lon"], 10)
                sleep(1.1)
                nearest = reverse(s["lat"], s["lon"], 18)
                hit = describe_place(area, nearest)
                cache[s["key"]] = hit
            except (requests.RequestException, ValueError) as e:
                errors.append(f"stop {s['stop']}: {type(e).__name__}: {str(e)[:120]}")
            looked += 1
        if hit is not None:
            s["place"] = hit
    if looked >= max_lookups and any("place" not in s for s in stops):
        errors.append(f"named {max_lookups} new stops this call; call again for the rest")
    cache_path.write_text(json.dumps(cache, indent=1))
    return stops, errors


OVERPASS = "https://overpass-api.de/api/interpreter"
# public mirrors, tried in turn when one is busy or times out
OVERPASS_MIRRORS = (OVERPASS, "https://overpass.kumi.systems/api/interpreter",
                    "https://maps.mail.ru/osm/tools/overpass/api/interpreter")
GIVE_UP_AFTER = 2  # consecutive stops failing on every mirror: stop asking, report it


def _overpass_query(lat: float, lon: float, radius_m: int, road_m: int) -> str:
    a = f"(around:{radius_m},{lat:.3f},{lon:.3f})"
    r = f"(around:{road_m},{lat:.3f},{lon:.3f})"
    return f"""[out:json][timeout:25];
(
  nwr{a}["name"]["tourism"~"attraction|viewpoint|museum|artwork"];
  nwr{a}["name"]["historic"];
  nwr{a}["name"]["amenity"="place_of_worship"];
  nwr{a}["name"]["natural"~"wood|water|peak|waterfall|cliff|cave_entrance"];
  nwr{a}["name"]["landuse"="forest"];
  nwr{a}["name"]["boundary"~"national_park|protected_area"];
  nwr{a}["name"]["leisure"="nature_reserve"];
  nwr{a}["name"]["waterway"~"river|dam"];
  way{r}["highway"~"motorway|trunk|primary|secondary"]["ref"];
);
out tags center 80;"""


def _kind(tags: dict) -> str:
    if tags.get("highway"):
        return "road"
    if tags.get("amenity") == "place_of_worship":
        return f"{tags.get('building') or 'place of worship'} ({tags.get('religion', '?')})" \
            if tags.get("building") in ("temple", "mosque", "church") else \
            f"place of worship ({tags.get('religion', '?')})"
    for key in ("boundary", "leisure", "historic", "tourism", "natural", "waterway", "landuse"):
        if tags.get(key):
            v = tags[key]
            return {"wood": "forest", "forest": "forest", "national_park": "national park",
                    "protected_area": "protected area", "nature_reserve": "nature reserve",
                    "yes": key}.get(v, v.replace("_", " "))
    return "place"


def parse_nearby(data: dict, here: tuple[float, float], limit: int = 15) -> list[dict]:
    out: dict[str, dict] = {}
    for el in data.get("elements", []):
        tags = el.get("tags") or {}
        c = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
        if c.get("lat") is None:
            continue
        name = tags.get("name:en") or tags.get("name")
        if tags.get("highway"):
            name = " / ".join(x for x in (tags.get("ref"), name) if x)
        if not name:
            continue
        item = {"name": name, "kind": _kind(tags),
                "km": round(haversine_km(here, (c["lat"], c["lon"])), 2)}
        if tags.get("wikipedia") or tags.get("wikidata"):
            item["well_known"] = True  # has an encyclopedia entry
        if name not in out or item["km"] < out[name]["km"]:
            out[name] = item
    return sorted(out.values(), key=lambda x: (not x.get("well_known"), x["km"]))[:limit]


def _overpass(lat: float, lon: float, radius_m: int, road_m: int, timeout: float = 30) -> dict:
    """The query on each mirror in turn; raises the last error if all fail."""
    last: Exception | None = None
    for url in OVERPASS_MIRRORS:
        try:
            r = requests.post(url, data={"data": _overpass_query(lat, lon, radius_m, road_m)},
                              timeout=timeout, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last = e
    raise last or requests.RequestException("no Overpass mirror answered")


def nearby_stops(stops: list[dict], cache_path: Path, *, radius_km: float = 2.0,
                 max_lookups: int = 20, fetch=None, sleep=None
                 ) -> tuple[list[dict], list[str]]:
    """Add ``nearby`` (named temples, forts, forests, parks, rivers, highways) to stops."""
    fetch, sleep = fetch or _overpass, sleep or time.sleep
    cache: dict[str, list] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    errors: list[str] = []
    looked = failed_in_a_row = 0
    radius_m, road_m = int(radius_km * 1000), 300
    for s in stops:
        key = f"{s['key']}@{radius_m}"
        hit = cache.get(key)
        if hit is None and looked < max_lookups and failed_in_a_row < GIVE_UP_AFTER:
            try:
                if looked:
                    sleep(1.0)
                hit = parse_nearby(fetch(s["lat"], s["lon"], radius_m, road_m),
                                   (s["lat"], s["lon"]))
                cache[key] = hit
                failed_in_a_row = 0
            except (requests.RequestException, ValueError) as e:
                errors.append(f"stop {s['stop']} nearby: {type(e).__name__}: {str(e)[:120]}")
                failed_in_a_row += 1
                if failed_in_a_row == GIVE_UP_AFTER:
                    errors.append("Overpass did not answer for 2 stops in a row (all "
                                  "mirrors): skipped the rest; try again later, or use "
                                  "a smaller nearby_km")
            looked += 1
        if hit is not None:
            s["nearby"] = hit
    cache_path.write_text(json.dumps(cache, indent=1))
    return stops, errors
