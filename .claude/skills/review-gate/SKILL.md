---
name: review-gate
description: Build the pre-render preview - one composited frame per cut and per B-roll overlay showing the exact crop, burned-in caption, overlay placement and detected face box - and get explicit approval before the full encode. Use after take-geometry, and whenever the cut changes.
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

## Decision
- Show the sheet(s) to the user and summarise what you checked. Only the user
  approves, unless they've explicitly handed that to you:
  ```bash
  uv run llmve review approve <ws> --notes "..."
  uv run llmve review reject  <ws> --notes "what to change"
  ```
- `llmve render` won't run unless the current cut and geometry were
  approved. Any change to the cut needs a fresh `llmve review`.
