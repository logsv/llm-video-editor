import pytest

from llm_video_editor.schemas import Credit
from llm_video_editor.stages import publish as pub
from llm_video_editor.stages.source_meta import parse_iso6709


@pytest.mark.parametrize("value, expected", [
    ("+28.6139+077.2090+216.000/", (28.6139, 77.209, 216.0)),
    ("-33.8688+151.2093/", (-33.8688, 151.2093, None)),
    ("+40.7128-074.0060+010.5/", (40.7128, -74.006, 10.5)),
    ("not a place", None),
    ("+95.0+010.0/", None),
])
def test_iso6709(value, expected):
    assert parse_iso6709(value) == expected


def test_titles_are_cleaned_capped_and_flagged():
    warnings = []
    long = "word " * 30
    out = pub.fix_titles(["<b>Great</b> video", "Great video", long, ""], warnings)
    assert out[0] == "bGreat/b video" and len(out[-1]) <= 100
    assert len(out) == 3 and any("characters" in w for w in warnings)
    assert pub.fix_titles([], []) == ["Untitled video"]


def test_tags_fit_youtubes_500_characters():
    tags = pub.fix_tags([f"tag phrase number {i}" for i in range(60)] + ["a,b", "TAG PHRASE NUMBER 1"])
    cost = sum(len(t) + (2 if " " in t else 0) for t in tags) + len(tags) - 1
    assert cost <= 500 and len(set(t.lower() for t in tags)) == len(tags)


def test_hashtags_normalised_and_capped():
    assert pub.fix_hashtags(["budget tech", "#Mic", "#mic", "x!", "four", "five"]) == [
        "#budgettech", "#Mic", "#x"]


def _copy(body="Body."):
    return pub._Copy(titles=["t"], hook="Hook line.", body=body, tags=[], hashtags=[],
                     category="Education", pinned_comment="", thumbnails=[])


def test_description_keeps_chapters_and_credits_when_trimmed():
    c = Credit(file="/m.mp3", kind="music", title="Song", provider="openverse", license="by 4.0",
               attribution='"Song" by Ann is licensed under CC BY 4.0.',
               landing_url="https://jamendo/1")
    text = pub.build_description(_copy("x" * 6000), [(0.0, "Intro"), (65.0, "Setup")], [c],
                                 ["#tag"])
    assert len(text) <= pub.DESCRIPTION_MAX
    assert "Chapters\n0:00 Intro\n1:05 Setup" in text
    assert '"Song" by Ann is licensed under CC BY 4.0. https://jamendo/1' in text
    assert text.endswith("#tag") and text.startswith("Hook line.")


def test_credit_line_without_provider_text():
    c = Credit(file="/v.mp4", kind="video", title="Goat", creator="Lee", provider="wikimedia",
               license="CC BY-SA 4.0")
    assert pub.credit_line(c) == "Goat by Lee (CC BY-SA 4.0)"


def test_material_is_sampled_not_cut_off():
    lines = [f"[clip c1 #{i} @ 0:00] " + "word " * 40 for i in range(2000)]
    text = pub._fit(lines, 20_000)
    assert len(text) < 21_000 and "#0 " in text and "#1999" not in text.split("\n")[0]
    assert text.endswith("(sampled evenly)")
