---
name: watch
description: Ingest raw video (file, folder or glob) for editing - word-level Whisper transcript, scene/motion scores, talking-head vs B-roll classification, and low-token contact sheets. Use before planning any edit, or to look at footage (request dense 2 FPS sheets for a specific window).
---

# watch

```bash
uv run llmve watch ./raw_clips/            # or a single file, or "shoot/*.mov"
```
Prints a table and the workspace path (`.llmve/<batch_hash>`). Re-running on
unchanged inputs is instant, because results are cached by each file's path,
mtime and size.

## Outputs (read these, never the media)
- `transcript.json`: per clip, segments with ids, times, text and words.
- `clips_manifest.json`: per clip, `clip_id`, duration, size, fps,
  `kind` (`talking_head`, or `broll` when there is no speech), `speech_ratio`,
  `motion_score`, `scenes` and `overview_sheets`.
- `sheets/index.json`: maps every tile of every sheet to a source timestamp.
- `sheets/<clip_id>_overview_NNN.png`: a 4x4 grid at 0.5 FPS with timestamps burned in.

## Looking at footage
1. Start with the overview sheets of the clips you care about. Each is about
   1.8k vision tokens for 16 frames.
2. Only for candidate moments, request dense sheets:
   ```bash
   uv run llmve watch-window <ws> <clip_id> <start_s> <end_s> --fps 2
   ```
   Keep windows short (30 s or less, which is 4 sheets). Don't request 2 FPS
   for whole clips: a flat 2 FPS over 15 minutes costs about 200k tokens.
3. Use `sheets/index.json` to turn a tile into a timestamp.

See `reference.md` for token budgets.

## Notes
- The default ASR model is `distil-large-v3`, which is English-only. For other
  languages, pass `--asr-model large-v3-turbo --language <code>`.
- A clip that has audio but no recognised speech is tagged `broll`.
