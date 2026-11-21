---
name: youtube-edit
description: "Edits a long raw recording (talking head, tutorial, podcast, screen share; 5 min to 3 h) into a finished 16:9 YouTube video: reads the full transcript, keeps the last clean take of every line, cuts tangents and dead air, adds a cold open, tightens pauses with jump cuts and punch-ins, names YouTube chapters, normalises loudness to -14 LUFS, and exports captions, chapters and a frame-by-frame cut check. Trigger with /youtube-edit or phrases like 'edit this video for YouTube', 'cut my recording', 'clean up this long video', 'make chapters for this'."
---

# /youtube-edit

Raw recording in, YouTube-ready MP4 + captions + chapters out. Claude is the editor: it reads the
WHOLE transcript and writes the edit decision list. The repo's scripts do the mechanical work and
are frame-accurate (tested to the exact frame), so never hand-roll ffmpeg cuts.

Read `reference.md` (next to this file) before writing the first EDL: schema, pacing values, and the
traps that cost a render.

## Commands

Run everything from the repo root. `LF` below means:

```
uv run --no-project --with faster-whisper python -m llm_video_editor.longform
```

(`--no-project` skips the repo's heavy torch/langgraph install; the long-form tools only need
faster-whisper and ffmpeg.) If `uv` is missing: `pip install faster-whisper` then use
`python -m llm_video_editor.longform`. If ffmpeg is missing, give the install line (macOS
`brew install ffmpeg`, Ubuntu `sudo apt install ffmpeg`, Windows `winget install --id Gyan.FFmpeg -e`)
and stop.

## Hard rules
1. Never modify or delete the user's source files. Everything goes in `projects/<slug>/`.
2. Every cut is decided from `transcript.md` times, never guessed. Cut in silence: start ~0.05-0.10s
   before the first word, end ~0.15-0.25s after the last word.
3. Keep the speaker's meaning. Removing a retake, a stumble or a tangent is fine; splicing words
   from different sentences into a claim they did not make is not.
4. Captions say what was said: fix ASR mishearings of names and terms with `caption_fixes`.
5. Prove the result from the final render (`*.cuts.jpg`, the A/V drift line), not from the plan.
6. Keep every version in `renders/`; never overwrite a render the user has seen.

## Workflow

### 1. Intake
- Source: the file the user names (ask once if none). Slug = short kebab-case of the topic.
- Ask in ONE message only what you cannot infer, with defaults: target length (default: as long as
  the content earns, typically 40-70% of raw), style (default "tight": jump cuts, alternate punch-in),
  names/terms for ASR (default: take them from the title/description they gave).

### 2. Ingest
```
LF ingest "<source>" projects/<slug> --model small --keywords "<names, products, jargon>"
```
- `--model medium` (or `large-v3` with a GPU) for accents, non-English, or heavy jargon.
  `--language hi` etc. for non-English. A 1 h recording takes ~5-15 min on CPU with `small`; run it in
  the background and tell the user.
- Re-running reuses `words.json`; delete it to re-transcribe.
- Then read `projects/<slug>/transcript.md` fully and look at `work/sheet.jpg` (+ `sheet.txt` for the
  timestamps of each frame) to see setups, screen shares, and where the camera changed.

### 3. Plan the edit -> `projects/<slug>/edl.json`
Work through the transcript in order and decide, for each passage: keep, cut, or move.
- **Retakes**: lines marked `(retake?)` restart a sentence. Keep the LAST complete clean take unless an
  earlier one is clearly better; cut everything from the first false start to the start of that take.
- **Dead air / setup chatter**: "is this recording", "let me share my screen", long `-- pause --`s.
- **Tangents**: cut anything that does not serve the video's promise. Be decisive; mention big cuts.
- **Cold open (hook)**: pick the single most compelling 5-20 s moment (a result, a bold claim, a
  surprising line) and place it as the first segment with `"tighten": false`, then the intro.
  The first 30 s decide retention: after the hook, get to the point fast (trim long intros).
- **Chapters**: put `"chapter"` on the segment that starts each topic. First chapter starts the video,
  at least 3 chapters, each at least 10 s. Names: short, specific, scannable ("Install in 2 minutes",
  not "Part 2").
- **Pacing**: `"tighten": {"enabled": true}` for talking head; set `"tighten": false` on segments with a
  live demo, music, or deliberate pauses. `"punch_in": {"mode": "alternate", "zoom": 1.12}` hides jump
  cuts on a single static camera; turn it off for screen recordings (it crops UI text).
- `caption_fixes`: map each ASR mishearing you spotted to the right spelling (`"cloud": "Claude"`).
- Post a short table to the user (section, source time range, kept length, chapter) plus the total
  runtime and what you cut, then keep going; do not wait for approval unless they asked to review.

### 4. Preview render
```
LF render projects/<slug> --dry-run      # instant: ranges, runtime, chapter warnings
LF render projects/<slug> --preview      # fast 540p review copy
```
Fix every WARNING (chapter rules, clipped segments). Look at `renders/*-preview.cuts.jpg` (before |
after frame for each cut, listed in `.cuts.txt`): the speaker should not visibly teleport (big
position jumps want a punch-in or a longer pause), and screen-share cuts should not jump between
unrelated windows.

### 5. Final render and QA
```
LF render projects/<slug>                         # source resolution, x264 crf 18, -14 LUFS
LF render projects/<slug> --encoder nvenc         # NVIDIA GPU; --encoder videotoolbox on a Mac
```
Check the printed A/V drift (should be ~0 ms; over 100 ms means a variable-frame-rate source: tell
the user), the `.report.json`, and spot-check `.cuts.jpg`.

### 6. Deliver
Give the user, in plain terms:
- the MP4 path, runtime (raw -> final), and the 3-5 biggest editorial calls you made
- `*.chapters.txt` contents, ready to paste into the description
- `*.srt` path (upload it in YouTube Studio > Subtitles; better than burned-in captions for long-form)
- 3 title options (under 60 characters, front-load the payoff), a 2-3 line description opening, and
  2-3 timestamps that would make strong thumbnails (look at those frames first)
Notes from the user ("cut the part about X", "the intro is too long") -> edit `edl.json`, re-render;
the version number bumps automatically.

## Defaults that came from real edits (do not re-litigate)
- Upload sidecar SRT captions for long-form; burned-in captions are for Shorts.
- Keep source resolution: a 4K upload gets a better YouTube encode even when watched at 1080p.
- -14 LUFS integrated, -1 dBTP is the long-form target; do not push louder.
- No zoom punch-in on screen recordings; alternate 1.0 / 1.12 on a single talking-head camera.
