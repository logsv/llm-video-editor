"""On-screen text for travel videos: where clips were filmed (route, place names and
named things nearby, only with the user's permission), objects in the shot (local
detector), the `location` and `label` templates, and review's on-screen text report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_video_editor.agent import places, setup
from llm_video_editor.agent.tools import Context, call
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeTranscriber
from tests.fixtures.make_fixture import make_clip

SCRIPT = [(0.3, "So today I ride from Bhopal."), (2.3, "The road is long and empty.")]
BHOPAL, NEAR_BHOPAL, KHAJURAHO = (23.2599, 77.4126), (23.2610, 77.4130), (24.8318, 79.9199)


@pytest.fixture(autouse=True)
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMVE_CONFIG", str(tmp_path / "cfg"))


def ok(res):
    assert "error" not in res, res
    return res


def _footage(*points):
    out = {}
    for i, (lat, lon) in enumerate(points):
        out[f"/v/clip{i}.mp4"] = {"name": f"clip{i}.mp4", "ok": True, "duration": 10,
                                  "recorded_at": f"2026-03-14T0{i}:00:00Z",
                                  "latitude": lat, "longitude": lon}
    return out


def test_route_orders_by_time_and_groups_stops():
    fx = _footage(BHOPAL, NEAR_BHOPAL, KHAJURAHO)
    fx["/v/nogps.mp4"] = {"name": "nogps.mp4", "ok": True, "recorded_at": None,
                          "latitude": None, "longitude": None}
    r = places.route(fx, {"/v/clip2.mp4": "c-kh"})
    assert [s["clips"] for s in r["stops"]] == [["clip0.mp4", "clip1.mp4"], ["c-kh"]]
    assert 280 < r["km_between_filmed_points"] < 330  # straight line Bhopal-Khajuraho
    assert r["clips_without_gps"] == 1


def test_names_and_nearby_are_cached(tmp_path):
    calls = []

    def reverse(lat, lon, zoom):
        calls.append(zoom)
        if zoom == 10:
            return {"address": {"town": "Khajuraho", "state": "Madhya Pradesh",
                                "country": "India", "county": "Chhatarpur"}}
        return {"name": "Kandariya Mahadeva Temple", "type": "place_of_worship",
                "address": {"road": "Temple Road", "town": "Khajuraho"}}

    stops = places.route(_footage(KHAJURAHO), {})["stops"]
    named, errs = places.name_stops(stops, tmp_path / "p.json", reverse=reverse,
                                    sleep=lambda s: None)
    p = named[0]["place"]
    assert p["label"] == "Kandariya Mahadeva Temple, Khajuraho, Madhya Pradesh" and not errs
    assert p["road"] == "Temple Road" and p["district"] == "Chhatarpur"
    places.name_stops(places.route(_footage(KHAJURAHO), {})["stops"], tmp_path / "p.json",
                      reverse=reverse, sleep=lambda s: None)
    assert calls == [10, 18]  # second call served from the cache

    overpass = {"elements": [
        {"type": "way", "center": {"lat": 24.85, "lon": 79.93},
         "tags": {"name": "Western Group of Temples", "historic": "yes", "wikidata": "Q1"}},
        {"type": "way", "center": {"lat": 24.70, "lon": 80.0},
         "tags": {"name": "Panna National Park", "boundary": "national_park"}},
        {"type": "way", "center": {"lat": 24.832, "lon": 79.92},
         "tags": {"highway": "trunk", "ref": "NH39"}},
        {"type": "node", "lat": 24.833, "lon": 79.92,
         "tags": {"name": "Lakshmana Temple", "amenity": "place_of_worship",
                  "religion": "hindu", "building": "temple"}},
        {"type": "node", "lat": 24.9, "lon": 79.9, "tags": {"natural": "wood"}}]}  # no name
    stops, errs = places.nearby_stops(named, tmp_path / "n.json",
                                      fetch=lambda *a: overpass, sleep=lambda s: None)
    near = stops[0]["nearby"]
    assert near[0]["name"] == "Western Group of Temples" and near[0]["well_known"]
    kinds = {n["name"]: n["kind"] for n in near}
    assert kinds["NH39"] == "road" and kinds["Panna National Park"] == "national park"
    assert kinds["Lakshmana Temple"] == "temple (hindu)" and len(near) == 4


@requires_ffmpeg
def test_places_tool_needs_the_users_permission(tmp_path, monkeypatch):
    raw = tmp_path / "trip"
    make_clip(raw / "a.mp4", duration=2, size=(320, 240))
    c = Context(home=tmp_path / "home")
    ws = ok(call("open_project", {"inputs": [str(raw)]}, c))["workspace"]
    assert "no clip has GPS" in ok(call("places", {"workspace": ws}, c))["note"]

    fp = Path(ws) / "footage.json"
    facts = json.loads(fp.read_text())
    for f in facts.values():
        f["latitude"], f["longitude"] = KHAJURAHO
    fp.write_text(json.dumps(facts))
    monkeypatch.setattr(places, "_reverse", lambda *a, **k: pytest.fail("sent coordinates"))
    res = ok(call("places", {"workspace": ws, "lookup": True}, c))
    assert "permission" in res["note"] and "place" not in res["stops"][0]
    assert "location_lookup is the user's setting" in call(
        "remember", {"reason": "agent wants it", "set_fields": {"location_lookup": True}},
        c)["error"]

    setup.update_profile(set_fields={"location_lookup": "true"})  # the user's command
    monkeypatch.setattr(places, "_reverse", lambda lat, lon, zoom, **k: {
        "name": "Khajuraho", "address": {"town": "Khajuraho", "state": "Madhya Pradesh"}})
    monkeypatch.setattr(places, "_overpass", lambda *a, **k: {"elements": []})
    monkeypatch.setattr(places.time, "sleep", lambda s: None)
    res = ok(call("places", {"workspace": ws, "lookup": True, "nearby_km": 2}, c))
    assert res["stops"][0]["place"]["label"] == "Khajuraho, Madhya Pradesh", res
    assert res["stops"][0]["nearby"] == [] and "key" not in res["stops"][0]
    with pytest.raises(ValueError, match="true or false"):
        setup.update_profile(set_fields={"location_lookup": "maybe"})


class FakeObjects:
    def __init__(self, min_confidence=0.35):
        self.n = 0

    def detect(self, rgb):
        self.n += 1
        boxes = [("cow", (0.6, 0.5, 0.8, 0.8), 0.9)]
        if self.n == 2:
            boxes.append(("dog", (0.1, 0.6, 0.2, 0.7), 0.5))
        return boxes

    def close(self):
        pass


@requires_ffmpeg
def test_objects_templates_and_on_screen_text(tmp_path):
    raw = tmp_path / "trip"
    make_clip(raw / "a.mp4", duration=4, size=(640, 360))
    c = Context(home=tmp_path / "home", object_detector_factory=FakeObjects,
                transcriber_factory=lambda m: FakeTranscriber(SCRIPT))
    ws = ok(call("open_project", {"inputs": [str(raw)]}, c))["workspace"]
    a = ok(call("analyze", {"workspace": ws}, c))["clips"][0]["clip_id"]
    found = ok(call("objects", {"workspace": ws, "clip_id": a, "every": 1}, c))
    cow, dog = found["objects"]
    assert found["frames_checked"] == 4 and cow["label"] == "cow" and cow["frames"] == 4
    assert cow["best"]["x"] == 0.7 and cow["best"]["y"] == 0.65 and dog["first"] == 1.0
    only = ok(call("objects", {"workspace": ws, "clip_id": a, "labels": ["dog"]}, c))
    assert [o["label"] for o in only["objects"]] == ["dog"]

    names = {t["name"]: t for t in ok(call("graphics_templates", {}, c))["templates"]}
    assert {"location", "label", "lower-third", "title-card", "callout"} <= set(names)
    assert set(names["label"]["variables"]) >= {"text", "x", "y"}

    from llm_video_editor.agent.edits import EditStore
    from llm_video_editor.agent.tools import _on_screen_text
    from llm_video_editor.workspace import Workspace

    ok(call("new_edit", {"workspace": ws, "target_platform": "youtube", "reason": "start"}, c))
    ok(call("batch", {"workspace": ws, "reason": "cut", "operations": [
        {"op": "add_clip", "params": {"source_clip_id": a, "start": 0, "end": 3.5,
                                      "chapter": "Leaving Bhopal"}}]}, c))
    edl = EditStore(Workspace.load(Path(ws))).current()
    bare = _on_screen_text(edl)
    assert bare["graphics"] == 0 and "no titles" in bare["note"]
    assert bare["chapters_without_a_title_graphic"] == ["Leaving Bhopal"]
    ok(call("add_item", {"workspace": ws, "kind": "graphics", "reason": "where we are",
                         "fields": {"template": "location", "anchor_clip_id": "c001",
                                    "variables": {"title": "Bhopal"}, "duration": 2.5}}, c))
    titled = _on_screen_text(EditStore(Workspace.load(Path(ws))).current())
    assert titled["graphics"] == 1 and titled["templates"] == ["location"]
    assert titled["chapters_without_a_title_graphic"] == [] and "note" not in titled
    assert titled["longest_without_graphics"]["from"] == 2.5


from llm_video_editor.hyperframes import toolchain  # noqa: E402

real = pytest.mark.skipif(not (toolchain.toolchain().installed
                               and (toolchain.node_major() or 0) >= 22),
                          reason="HyperFrames toolchain not installed")


@real
@pytest.mark.hyperframes
@requires_ffmpeg
def test_review_measures_real_graphics_on_their_shot(tmp_path):
    raw = tmp_path / "trip"
    make_clip(raw / "a.mp4", duration=5, size=(640, 360))
    c = Context(home=tmp_path / "home", transcriber_factory=lambda m: FakeTranscriber(SCRIPT))
    ws = ok(call("open_project", {"inputs": [str(raw)]}, c))["workspace"]
    a = ok(call("analyze", {"workspace": ws}, c))["clips"][0]["clip_id"]
    ok(call("new_edit", {"workspace": ws, "target_platform": "youtube", "reason": "start"}, c))
    ok(call("batch", {"workspace": ws, "reason": "cut and titles", "operations": [
        {"op": "add_clip", "params": {"source_clip_id": a, "start": 0.2, "end": 4.6}},
        {"op": "add_item", "params": {"kind": "graphics", "fields": {
            "template": "location", "anchor_clip_id": "c001", "duration": 3,
            "variables": {"title": "Khajuraho", "subtitle": "Madhya Pradesh",
                          "accent": "auto"}}}},
        {"op": "add_item", "params": {"kind": "graphics", "fields": {
            "template": "label", "anchor_clip_id": "c001", "offset": 0.5, "duration": 2.5,
            "variables": {"text": "Temple", "x": 0.7, "y": 0.5}}}}]}, c))
    rv = ok(call("review", {"workspace": ws, "detector": "none"}, c))
    checks = {g["graphic_id"]: g for g in rv["graphic_checks"]}
    assert set(checks) == {"g001", "g002"}
    loc = checks["g001"]
    assert loc["bbox"][0] < 0.2 and loc["bbox"][1] < 0.2  # top-left, inside safe margins
    assert loc["contrast"] > 1 and loc["palette"] and set(loc["suggested"]) == {
        "match", "contrast", "text"}
    assert not any("outside the safe area" in i for i in loc["issues"])
