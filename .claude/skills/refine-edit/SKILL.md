---
name: refine-edit
description: Apply the user's notes to an existing edit and re-render - "make it shorter", "cut the part about X", "music is too loud", "add my name", "more energy", "add a funny clip", "different thumbnail", "start with the result". Maps each note to the exact change in edl_raw.json or an llmve command, then runs `llmve finish`. Use whenever the user reacts to a rendered video.
---

# refine-edit

Work on the workspace the last `llmve auto` printed (`.llmve/<batch_hash>`). Read
`<ws>/transcript.md` (run `uv run llmve transcript <ws>` if missing) and
`<ws>/edl_raw.json` before changing anything. Batch all notes of one message
into one round, then run `uv run llmve finish <ws>` once.

## Notes -> changes
| Note | Change |
|---|---|
| "shorter" / "longer" | set `target_duration` in `edl_raw.json`, or remove/re-add clips (lowest `priority` first) |
| "cut the part about X" | find X's segment ids in `transcript.md`; remove that clip, or split it by setting `end`/`start` to those segments' times |
| "start with the result" / "hook first" | copy the clip holding that moment to the top of `clips` (cut-audit keeps the replay) |
| "too slow" / "more energy" | `"pacing": {"tighten": true, "max_gap": 0.45, "punch_in": "alternate"}` |
| "too choppy" | `"pacing": {"max_gap": 1.0, "punch_in": "off"}` |
| "music too loud / quiet" | `volume_db` of the track in `music`, relative to the speech (default -10 long-form, -8 vertical; steps of 2-3 dB) or `"mix": {"duck_depth_db": 16}` for a deeper dip under speech |
| "different music" | remove the track; `uv run llmve media search <ws> "<mood>" --kind music`, `media get`, `music add <ws> <file> --at c001 --loop` |
| "no music" | `"music": []` |
| "add a sound effect" | `uv run llmve media search <ws> "whoosh" --kind sfx`, then `music add ... --role sfx --at <clip> --offset <s>` |
| "add my voice-over" | `uv run llmve music add <ws> narration.wav --at c001 --role voiceover` (music ducks under it) |
| "add a funny clip / B-roll here" | exact moment: `uv run llmve media search <ws> "<what>" --kind video`, `media get <ws> <id>`, `uv run llmve media place <ws> <file> --at <clip> --offset <s> --duration 2.5`; let it choose: `media suggest` + `media fetch` |
| "fewer cutaways" | delete entries from `overlays` (ids `x001`...) |
| "add my name" / "title card" | `uv run llmve graphics add <ws> --at <clip> -t lower-third --var title=... --var subtitle=...`; `graphics chapters <ws>` |
| "captions on / off / bigger words" | `"captions": {"enabled": true, "style": "word_highlight"}`; misheard names: `captions.replacements` |
| "fix the chapter names" | edit `chapter` on the clips (first one starts the video, each >= 10 s, at least 3) |
| "face cut off" (vertical) | install face tracking (`uv sync --extra vision`) so geometry follows the face; otherwise `"fit_mode": "contain"` (letterbox) |
| "better title / thumbnail" | re-run `uv run llmve publish <ws> --brief "<what to stress>"` (no re-render needed) |

## Then
```bash
uv run llmve finish <ws>      # add --draft for a quick look, --review to check the sheet first
```
Report what changed (one line per note), the new length and QC line, and show
the new review sheet or thumbnails if they changed. If a note is ambiguous
("make it better"), propose 2-3 specific changes and do the one the user picks.
