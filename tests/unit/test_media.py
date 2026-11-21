"""Royalty-free media providers, parsed from responses in each API's documented shape."""

import json

import pytest

from llm_video_editor.media import library, providers
from llm_video_editor.schemas import MediaResult

OPENVERSE = {"result_count": 3, "results": [
    {"id": "a1", "title": "Sunny Day", "creator": "Ann", "creator_url": "https://x/ann",
     "license": "by", "license_version": "4.0",
     "license_url": "https://creativecommons.org/licenses/by/4.0/",
     "url": "https://cdn.example/sunny.mp3", "foreign_landing_url": "https://jamendo/1",
     "duration": 125000, "attribution": '"Sunny Day" by Ann is licensed under CC BY 4.0.',
     "tags": [{"name": "happy"}, {"name": "acoustic"}]},
    {"id": "a2", "title": "Free Loop", "creator": "Bo", "license": "cc0", "license_version": "1.0",
     "url": "https://cdn.example/loop.mp3", "duration": 30000},
    {"id": "a3", "title": "No Commerce", "license": "by-nc", "license_version": "4.0",
     "url": "https://cdn.example/nc.mp3"}]}

PIXABAY = {"total": 1, "hits": [{
    "id": 125, "pageURL": "https://pixabay.com/videos/id-125/", "tags": "cat, funny, pet",
    "duration": 12, "user": "Jo", "user_id": 9,
    "videos": {"large": {"url": "", "width": 0, "height": 0},
               "medium": {"url": "https://cdn.pixabay/125_m.mp4", "width": 1920, "height": 1080,
                          "size": 1, "thumbnail": "https://cdn.pixabay/125.jpg"}}}]}

PEXELS = {"videos": [{
    "id": 854100, "width": 3840, "height": 2160,
    "url": "https://www.pexels.com/video/people-walking-on-a-street-854100/",
    "image": "https://images.pexels/854100.jpg", "duration": 9,
    "user": {"name": "Kim", "url": "https://www.pexels.com/@kim"},
    "video_files": [
        {"id": 1, "quality": "uhd", "file_type": "video/mp4", "width": 3840, "height": 2160,
         "link": "https://cdn.pexels/uhd.mp4"},
        {"id": 2, "quality": "hd", "file_type": "video/mp4", "width": 1920, "height": 1080,
         "link": "https://cdn.pexels/hd.mp4"},
        {"id": 3, "quality": "sd", "file_type": "video/mp4", "width": 640, "height": 360,
         "link": "https://cdn.pexels/sd.mp4"}]}]}

COMMONS = {"query": {"pages": {
    "11": {"pageid": 11, "index": 1, "title": "File:Funny_goat_jump.webm", "imageinfo": [{
        "url": "https://upload.wikimedia.org/goat.webm", "descriptionurl": "https://commons/goat",
        "width": 1280, "height": 720, "duration": 6.5, "extmetadata": {
            "LicenseShortName": {"value": "CC BY-SA 4.0"},
            "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0"},
            "Artist": {"value": "<a href='x'>Farmer Lee</a>"},
            "AttributionRequired": {"value": "true"}}}]},
    "12": {"pageid": 12, "index": 2, "title": "File:Nope.webm", "imageinfo": [{
        "url": "https://upload.wikimedia.org/nope.webm",
        "extmetadata": {"LicenseShortName": {"value": "CC BY-NC 2.0"}}}]}}}}


def test_openverse_keeps_only_commercial_and_editable_licences():
    rs = providers.parse_openverse(OPENVERSE, "music")
    assert [r.result_id for r in rs] == ["openverse:a1", "openverse:a2"]
    a, b = rs
    assert a.duration == 125.0 and a.license == "by 4.0" and a.attribution_required
    assert a.attribution.startswith('"Sunny Day" by Ann') and a.tags == ["happy", "acoustic"]
    assert b.license == "cc0 1.0" and not b.attribution_required


def test_pixabay_skips_empty_renditions():
    (r,) = providers.parse_pixabay(PIXABAY)
    assert r.download_url.endswith("125_m.mp4") and (r.width, r.height) == (1920, 1080)
    assert r.tags == ["cat", "funny", "pet"] and r.attribution == "Video by Jo from Pixabay"


def test_pexels_prefers_hd_over_4k_and_names_the_clip():
    (r,) = providers.parse_pexels(PEXELS)
    assert r.download_url.endswith("hd.mp4") and r.title == "People walking on a street"
    assert r.creator == "Kim" and not r.attribution_required


def test_commons_filters_nc_and_strips_html():
    (r,) = providers.parse_commons(COMMONS)
    assert r.title == "Funny goat jump" and r.creator == "Farmer Lee"
    assert r.license == "CC BY-SA 4.0" and r.attribution_required
    assert "via Wikimedia Commons" in r.attribution
    assert not providers.commons_license_ok("CC BY-ND 3.0")
    assert providers.commons_license_ok("Public domain")


def test_missing_keys_are_reported_not_raised(monkeypatch):
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.setattr(providers, "_get", lambda url, **k: COMMONS)
    res = library.search("funny goat", "video")
    assert [r.provider for r in res.results] == ["wikimedia"]
    assert any("PIXABAY_API_KEY" in e for e in res.errors)
    assert any("PEXELS_API_KEY" in e for e in res.errors)


def test_openverse_request_asks_for_commercial_and_modification(monkeypatch):
    seen = {}

    def fake(url, *, params, headers=None):
        seen.update(params)
        return OPENVERSE

    monkeypatch.setattr(providers, "_get", fake)
    providers.openverse("calm piano", "music", length="short")
    assert seen["license_type"] == "commercial,modification" and seen["category"] == "music"
    assert seen["length"] == "short"
    with pytest.raises(providers.ProviderUnavailable):
        providers.openverse("x", "video")


class _Resp:
    def __init__(self, body: bytes, ctype: str):
        self.body, self.headers = body, {"Content-Type": ctype}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, n):
        yield self.body


def test_download_refuses_web_pages(monkeypatch, tmp_path):
    monkeypatch.setattr(library.requests, "get", lambda *a, **k: _Resp(b"<html>", "text/html"))
    with pytest.raises(providers.ProviderUnavailable, match="not audio/video"):
        library.download("https://x/page", tmp_path / "f", "music")


def test_fetch_records_credit_and_video_sidecar(monkeypatch, tmp_path):
    from llm_video_editor.workspace import Workspace

    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    ws = Workspace(tmp_path / "ws", [src])
    monkeypatch.setattr(library.requests, "get",
                        lambda *a, **k: _Resp(b"\x00\x01", "video/mp4"))
    (r,) = providers.parse_pixabay(PIXABAY)
    path = library.fetch(ws, r, description="cat jumps off a sofa")
    side = json.loads(path.with_suffix(".json").read_text())
    assert path.parent.name == "broll" and side["description"] == "cat jumps off a sofa"
    assert side["credit"]["attribution"] == "Video by Jo from Pixabay"
    assert library.sidecar_credit(path).provider == "pixabay"
    music = MediaResult(result_id="openverse:a1", provider="openverse", kind="music",
                        title="Sunny Day", license="by 4.0", download_url="https://x/s.mp3",
                        attribution="Sunny Day by Ann", attribution_required=True)
    monkeypatch.setattr(library.requests, "get", lambda *a, **k: _Resp(b"ID3", "audio/mpeg"))
    mp3 = library.fetch(ws, music)
    assert mp3.suffix == ".mp3"
    credits = library.load_credits(ws).items
    assert {c.provider for c in credits} == {"pixabay", "openverse"}
    reg = library.register(ws, src, kind="music", title="Night Drive",
                           license="YouTube Audio Library", provider="youtube-audio-library",
                           attribution="Night Drive - Artist")
    assert reg.attribution_required and len(library.load_credits(ws).items) == 3
