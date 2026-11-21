---
name: youtube-shorts
description: Find the best 20-60 second moments in a long video and render each as a vertical 9:16 YouTube Short / Reel with a hook-first cut, face-tracked crop and burned-in captions. Trigger with /youtube-shorts or "make shorts from this video", "clip this podcast", "find the best moments". Builds on the edit-video pipeline.
---

# /youtube-shorts

Same stages and gates as `edit-video` (read it first). One workspace holds one
cut at a time, so shorts are made one after another.

## 1. Find the moments
```bash
uv run llmve watch <recording> --chunk-asr-over 10   # instant if already watched
uv run llmve transcript <ws>
```
From `transcript.md`, rank candidates. A good Short works with zero context,
opens on its hook (claim, question, result), carries one idea and lands a
payoff; 20-45 s is the sweet spot. Pick the top 3 unless told otherwise and
show the list (segment ids, hook line, length).

## 2. For each short
```bash
uv run llmve plan <ws> -t shorts --target-duration 40s -p "<this moment: segments N-M, hook first>"
```
Check `edl_raw.json` holds that moment (fix it from `transcript.json` times if
not; a later punchy line moved to the front is a valid hook) and set
`"pacing": {"tighten": true, "max_gap": 0.4, "drop_fillers": true}` and
`"captions": {"style": "word_highlight"}`.
```bash
uv run llmve cut-audit <ws>
uv run llmve geometry <ws>        # face-tracked 9:16 crop and caption band
uv run llmve review <ws>          # stop: show the sheet to the user
uv run llmve render <ws> -o shorts/<n>-<name>.mp4
```
Screen-share segments rarely survive a 9:16 crop: avoid them or say so.

## 3. Deliver
Per short: path, length, hook line, a title under 40 characters, QC line.
