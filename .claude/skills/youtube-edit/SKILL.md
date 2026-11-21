---
name: youtube-edit
description: Edit one long raw recording (talking head, tutorial, podcast, screen share; 5 min to hours) into a finished 16:9 YouTube video - last clean take of every line, cold open, jump-cut tightening with punch-ins, YouTube chapters, sidecar SRT captions and -14 LUFS. Trigger with /youtube-edit or "edit this recording for YouTube", "clean up my long video", "add chapters". Builds on the edit-video pipeline.
---

# /youtube-edit

Long-form specialisation of `edit-video`: same stages, same gates, same rules
(never open media files, only the user approves the review). Read
`.claude/skills/edit-video/SKILL.md` first; this skill adds the editorial
judgement and the long-form options. See `reference.md` for the EDL fields.

## 1. Intake
Ask in one message only what you cannot infer: target length (default: what
the content earns, typically 40-70% of the raw), pacing (default `tight`), and
names/terms the ASR may mishear.

## 2. Watch and read the whole transcript
```bash
uv run llmve watch <recording> --chunk-asr-over 10     # parallel ASR for long files
uv run llmve transcript <ws>                            # writes <ws>/transcript.md
```
Read `transcript.md` end to end: `[id] start-end text`, `(retake?)` where a
line was restarted, `(unclear N)` low-confidence words, `-- pause Xs --` dead
air. Skim the overview sheets for setups and screen shares.

## 3. Plan, then refine edl_raw.json
```bash
uv run llmve plan <ws> -t youtube --target-duration 15m -p "<brief>"
```
Put the editorial intent in the brief: "keep the last clean take of every
line, cut setup chatter and tangents, open on <the strongest moment>". Then
edit `edl_raw.json` yourself (times only from `transcript.json`):
- **Cold open**: copy the clip holding the most compelling 5-20 s moment (a
  result, a bold claim) to the top of `clips`, trimmed to that moment.
- **Chapters**: set `"chapter"` on the clip that starts each topic. At least 3,
  each 10 s or longer; short, specific names ("Install in 2 minutes").
- **Pacing**: `"pacing": {"tighten": true, "max_gap": 0.6, "drop_fillers": true,
  "punch_in": "alternate"}` for a single talking-head camera. Use
  `"punch_in": "off"` for screen recordings (zoom crops UI text) and keep
  live demos untightened by giving them `"role": "broll"`.
- **Caption fixes**: `"captions": {"enabled": false, "replacements": {"cloud": "Claude"}}`
  (long-form uploads the SRT instead of burning text in).
Post a short table (clip, source range, length, chapter) and the runtime.

## 4. Audit, geometry, review
```bash
uv run llmve cut-audit <ws>        # PASSED, plus "Xs of pauses tightened"
uv run llmve geometry <ws>
uv run llmve review <ws>           # stop: show the sheets to the user
```
Tightening runs inside cut-audit after the sentence fixes, so splits always
land on word edges. Any edit to `edl_raw.json` means cut-audit, geometry and
review again.

## 5. Render
```bash
uv run llmve render <ws> --draft --no-captions -o <ws>/render/preview.mp4   # optional quick look
uv run llmve render <ws> --no-captions -o <out.mp4>
```
Check `QC PASSED`, then fix every chapter warning (YouTube silently hides
chapters that break its rules) and re-render.

## 6. Deliver
- MP4 path, raw -> final runtime, the 3-5 biggest editorial calls
- `<out>.chapters.txt` contents, ready to paste into the description
- `<out>.srt` (upload in YouTube Studio > Subtitles)
- 3 title options (under 60 characters, payoff first) and 2-3 thumbnail moments
Notes ("cut the part about X") -> edit `edl_raw.json`, re-run from cut-audit.
