---
name: youtube-shorts
description: "Finds the best 20-60 second moments in a long video and renders them as vertical 9:16 YouTube Shorts / Reels with a hook-first cut and captions file. Trigger with /youtube-shorts or 'make shorts from this video', 'clip this podcast', 'find viral moments'."
---

# /youtube-shorts

Long video in, 3-5 vertical clips out. Uses the same tools as `/youtube-edit`; read
`.claude/skills/youtube-edit/SKILL.md` (Commands, Hard rules) and `reference.md` first.

## Workflow
1. **Ingest** (skip if `projects/<slug>/words.json` exists from a `/youtube-edit` run):
   `LF ingest "<source>" projects/<slug> --keywords "<names>"`
2. **Pick moments** from `transcript.md`. A good Short:
   - works with zero context: the first sentence is the hook (claim, question, result, conflict)
   - one idea, with a payoff in the last 3 s; 20-45 s is the sweet spot, 60 s max
   - can start mid-passage: a later punchy line moved to the front is a valid hook
   Rank candidates and pick the top 3-5 (ask how many if the user cares; default 3).
3. **One EDL per short**: `projects/<slug>/shorts/<n>-<name>.json`, same schema as `edl.json`, with
   `"tighten": {"enabled": true, "max_gap": 0.4}` and no chapters. Use `cx` (0-1) per segment to keep
   the speaker in the 9:16 crop: look at `work/sheet.jpg` to see where they sit in the frame
   (left third ~0.3, centre 0.5, right third ~0.7). Screen-share segments rarely survive a 9:16 crop:
   avoid them or say so.
4. **Render** each:
   `LF render projects/<slug> --edl projects/<slug>/shorts/<file>.json --vertical --preview`
   check the `.cuts.jpg` (speaker centred? face never cut off?), then without `--preview`.
5. **Deliver**: paths, length, the hook line, a title (under 40 characters) per short, and the `.srt`
   files. Burned-in animated captions are not built in; suggest CapCut/Premiere or say so if asked.
