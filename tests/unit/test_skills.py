"""The Claude Code skills must stay in sync with the CLI they drive."""

import re
from pathlib import Path

import pytest

from llm_video_editor.cli.app import app

SKILLS = Path(__file__).resolve().parents[2] / ".claude" / "skills"
EXPECTED = {"edit-video", "watch", "cut-audit", "take-geometry", "b-roll-select", "review-gate",
            "youtube-edit", "youtube-shorts", "motion-graphics", "royalty-free-media",
            "youtube-publish", "refine-edit", "voiceover"}


def _commands() -> set[str]:
    names = {c.name or c.callback.__name__ for c in app.registered_commands}
    for group in app.registered_groups:
        sub = group.typer_instance
        names.add(group.name)
        names |= {f"{group.name} {c.name or c.callback.__name__.split('_')[-1]}"
                  for c in sub.registered_commands}
    return names


def test_all_skills_present_with_frontmatter():
    found = {p.parent.name for p in SKILLS.glob("*/SKILL.md")}
    assert found == EXPECTED
    for p in SKILLS.glob("*/SKILL.md"):
        text = p.read_text()
        m = re.match(r"---\nname: ([\w-]+)\ndescription: (.+?)\n---\n", text, re.S)
        assert m, f"{p} lacks name/description frontmatter"
        assert m.group(1) == p.parent.name
        assert 40 < len(m.group(2)) < 1024


@pytest.mark.parametrize("skill", sorted(EXPECTED))
def test_skill_commands_exist(skill):
    known = _commands()
    text = (SKILLS / skill / "SKILL.md").read_text()
    for cmd in re.findall(r"uv run llmve ([a-z-]+(?: [a-z]+)?)", text):
        first = cmd.split()[0]
        assert cmd in known or first in known, f"{skill}: unknown command `llmve {cmd}`"
