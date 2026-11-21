import pytest

from llm_video_editor.stages.durations import duration_from_prompt, parse_duration


@pytest.mark.parametrize("text,expected", [
    ("10m", 600), ("90s", 90), ("1:30", 90), ("1:02:03", 3723), ("1m30s", 90),
    ("2 minutes", 120), ("45", 45), ("1.5min", 90), ("1h", 3600),
])
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected


@pytest.mark.parametrize("bad", ["soon", "10 apples", "", "m"])
def test_parse_duration_rejects(bad):
    with pytest.raises(ValueError):
        parse_duration(bad)


@pytest.mark.parametrize("prompt,expected", [
    ("60s reel: top 3 takeaways, captions, upbeat", 60),
    ("30s reel: minimal edits", 30),  # the old parser returned 1800 here
    ("10 min compilation: best moments", 600),
    ("a 2-minute recap", 120),
    ("top 3 takeaways", None),
    ("make it 4k and punchy", None),
])
def test_duration_from_prompt(prompt, expected):
    assert duration_from_prompt(prompt) == expected
