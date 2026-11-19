---
name: take-geometry
description: Detect the speaker's face/subject per used scene and compute safe zones - crop centre for vertical reframing, a caption band that does not cover the face, and a picture-in-picture corner. Use after cut-audit passes and before review.
---

# take-geometry

```bash
uv run llmve geometry <ws>                  # --detector auto|mediapipe|yolo|none
```
For every scene the cut uses, this decodes 3 keyframes with seeks (never a
full decode) and runs the MediaPipe face detector. With the `pro` extra
installed, YOLO person boxes are the fallback. It writes `geometry_map.json`.

## Output
- `clips[].scenes[]`: detections as normalised `[x_min, y_min, x_max, y_max]`
  boxes, plus `focus` and `subject_box`.
- `placements[]`, one per EDL clip: `crop_center`, `subject_box_out` (in
  output-frame coordinates), `caption_position` (`bottom`, `top` or `center`),
  `caption_overlap` and `pip_corner`.

## Rules
- Placements are tied to the cut by `edl_hash`. If you re-run cut-audit,
  re-run geometry too; render and review ignore stale maps.
- `caption_overlap > 0` means no caption band is free of the subject. Mention
  it at review, or switch that EDL to `fit_mode: contain` and re-audit.
- `detector: none` means no detector was available, so crops stay centred.
  Tell the user to install the `vision` extra, plus `libegl1` on Linux.
