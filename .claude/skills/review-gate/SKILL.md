---
name: review-gate
description: Build the pre-render preview - one composited frame per cut and per B-roll overlay showing the exact crop, burned-in caption, overlay placement and detected face box - then verify it yourself and auto-approve it when every check passes (fix and re-review when one fails) before the full encode. Use after take-geometry, and whenever the cut changes.
---

# review-gate

```bash
uv run llmve review <ws>
```
This takes seconds. It writes `review/sheet_NNN.png` (6x3 tiles) and
`review/review.json`, which has `status: pending` and one entry per tile
(`t_out`, caption text, caption position, subject box).

## What to check on the sheet (look at every tile)
1. The red box (the detected face) sits fully inside the frame, not cut off.
2. Captions never sit on the red box, are readable, and match the speech.
3. Overlay tiles (labelled `oNNN over cNNN`) show relevant B-roll, and
   picture-in-picture insets cover neither the face nor the captions.
4. There are no black or blank frames, and no letterboxing you didn't intend.

Also read the `warnings` in `review.json`.

## Decision (auto-verify, auto-approve)
You make the call; do not wait for the user. Look at every tile and read
`review.json` yourself, then:

- **Every check passes and no warning matters** -> approve it and carry on to render:
  ```bash
  uv run llmve review approve <ws> --notes "auto-verified: <N> tiles, face/captions/overlays OK, warnings: <none|list>"
  ```
- **A check fails** -> fix the cause, not the sheet: edit `edl_raw.json`
  (times come from `transcript.json`) or the graphics/B-roll choice, re-run
  `cut-audit` -> `geometry` -> `llmve review`, and verify again. Allow up to
  3 fix rounds.
- **Still failing after 3 rounds, or the problem can't be fixed from the
  checkpoints** (e.g. the face is out of frame in the source) -> run
  `uv run llmve review reject <ws> --notes "<what failed>"`, stop, and tell the
  user what you saw.
- If the user asks to review the sheet themselves ("let me check", "don't
  approve"), show them the sheet and wait instead.

Always report what you checked and which warnings you accepted in the final
summary, with the sheet paths.

`llmve render` won't run unless the current cut and geometry were approved.
Any change to the cut needs a fresh `llmve review` and a fresh verification.
