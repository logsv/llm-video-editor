---
name: edit-video
description: Turn raw footage (a file, a folder of clips, or a glob) into a finished, platform-ready video from a natural-language brief, e.g. "10-minute YouTube edit from ./raw_clips" or "60s reel of the top 3 takeaways". Runs the whole edit with one command (auto-approved), shows the result, then improves it from the user's notes. Use this as the entry point for any video-editing request in this repo.
---

# Edit a video

One command does the whole edit; then you and the user improve it together.
You never open video or audio files (`.claude/settings.json` blocks them): you
work from JSON checkpoints and the PNG sheets and thumbnails the stages write.

## 1. Brief (ask only what you cannot infer, in one message)
Platform (`youtube`, `shorts`, `reels`, `tiktok`, `twitter`, `linkedin`), target
length, and the speaker's name/role if a lower third would help. Everything else
has a sensible automatic default.

## 2. One command
```bash
uv run llmve auto <inputs> -p "<brief>" -t <platform> --target-duration 10m -o <out.mp4>
```
It runs watch -> plan -> pacing, captions, chapters, ducked music, licensed
cutaways, graphics -> cut-audit -> geometry -> review (auto-approved; the user
chose this workflow) -> render -> publish, and prints one line per step
(`done` / `skip` with the reason / `fail`). Useful options:
`--name "Jane Doe|Founder"`, `--music-query "lofi chill"`, `--music off|<file>`,
`--no-cutaways`, `--no-humor`, `--pacing natural`, `--captions burn`,
`--broll-library <dir>`, `--look warm|golden-hour|teal-orange|cinematic|natural|...`
(`--look-strength 0.5`, `--match-color` to make clips match), `--draft` (fast preview), `--review` (stop at the
sheet instead of approving). Long recordings: it chunks ASR automatically.
YouTube lengths from 2 to 50 minutes work (`--target-duration 2m` ... `50m`): speech
comes first and, when it can't fill the length, shots from clips without speech fill
the rest in recording order (travel footage). If the footage can't cover the target,
plan says so and the cut comes out shorter.

If a core step fails (watch, plan, cut-audit, render) read the message and
`<ws>/auto_report.json`; fix the cause (brief, missing tool) and run again: it
reuses finished work.

## 3. Show the result
Tell the user, briefly: output path, length, QC line (LUFS, A/V offset), what each
automatic choice did (from the step lines: chapters, music track and licence,
cutaways placed, pauses removed) and every warning. Show
`<ws>/review/sheet_000.png` and the three `<out>.thumbN.jpg` (read the images),
then the recommended title from `<out>.upload.md`. Offer two or three concrete
improvements you noticed (e.g. "the intro runs 25 s before the point").

## 4. Improve from notes (the loop)
For each note, use the `refine-edit` skill: it maps notes to the exact change
(edit `edl_raw.json`, `llmve music/graphics/media ...`), then
```bash
uv run llmve finish <ws>          # audit -> review (approved) -> render -> publish
```
No new transcription or planning, so each round is quick. Re-plan with
`llmve auto ... -p "<sharper brief>"` only when the user wants a different
story, not a tweak.

## Rules
- Times come from `transcript.json`/`transcript.md`, never invented. Never hand-edit
  `edl_audited.json` (cut-audit writes it).
- The planner only receives text (clip ids, transcript lines); never put media
  paths in a prompt.
- Licensed media only (see `royalty-free-media`); every third-party file needs a
  credit before upload.
- Step-by-step stages (`watch`, `plan`, `cut-audit`, `geometry`, `review`,
  `render`, `publish`) are still there for debugging one stage; prefer `auto` and
  `finish`.
