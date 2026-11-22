---
name: color-grade
description: Make footage look consistent and premium - match clips to each other (exposure, white balance, saturation), tone-map HDR phone clips, and give the whole video one restrained look (natural, warm, golden-hour, teal-orange, cinematic, vivid, matte, bw, or a .cube LUT) that suits the content and mood. Use when the user asks for colour grading, a look, a "cinematic" or "premium" feel, or when clips from different cameras don't match.
---

# color-grade

Goal: the viewer should not notice the grade, only that the video looks good and
like one piece. Two layers, in this order:

## 1. Match (per clip, corrections only)
- `color_report` measures every clip of the edit: `mid` (exposure), black/white
  points, saturation, `warm` (colour cast), clipped highlights/shadows, the spread
  across clips, HDR clips that need tone mapping, and a `suggested` correction per
  clip towards the median clip.
- Apply suggestions with `set_color` (or `batch` ops `set_color`): exposure (EV),
  contrast, saturation, warmth, tint, `tonemap: true` for HDR (HLG/PQ) phone clips.
- Small is right: exposure within +/-0.7 EV, saturation 0.85-1.2, warmth/tint within
  +/-0.4. Don't lift a dark clip far (noise appears); cover it or cut it instead.
- Never use per-clip corrections as creative looks: every clip gets the same look below.

## 2. One look for the video (`set_grade`)
Choose from the brief, the content and the music's mood; one look per video.
| Look | When |
|---|---|
| `natural` | talking heads, tutorials, anything where truth matters |
| `warm` | travel, food, family, friendly vlogs |
| `golden-hour` | sunsets, roads, nostalgic travel |
| `teal-orange` | adventure, action, bike rides with sky and skin |
| `cinematic` | slower, story-led edits (soft contrast, slightly desaturated) |
| `vivid` | nature and landscapes (adds colour, careful with skin) |
| `matte` | moody, film-like, quiet edits |
| `bw` | a deliberate style choice only |
- `strength` 0.4-0.6 reads premium; above 0.8 looks filtered.
- `lut`: a `.cube` file the user provides replaces the preset.
- No grain, sharpening or heavy filters: they add noise to phone and action-cam footage.

## 3. Check
- `audit` lists colour warnings (big pushes, clips that disagree, a strong look).
- `review`: the tiles show the graded picture, and graphic checks measure text against it.
  Look for skin that turned orange or grey, skies that went green or cyan, crushed shadows.
- One-command mode: `llmve auto ... --look warm --look-strength 0.5 --match-color`.
