# /youtube-edit reference

## Long-form EDL fields (`edl_raw.json`)
| Field | Where | Meaning |
|---|---|---|
| `chapter` | clip | a YouTube chapter starts at this clip (moved to 0:00 if it is the first) |
| `zoom` | clip | fixed punch-in, 1.0-2.0, around `crop_center`; overrides `punch_in` |
| `pacing.tighten` | EDL | split A-roll clips at pauses longer than `max_gap` (cut-audit applies it) |
| `pacing.max_gap` / `lead` / `tail` | EDL | pause threshold, air before / after each speech island |
| `pacing.min_island` | EDL | drop islands shorter than this (stray breaths, one-word mumbles) |
| `pacing.drop_fillers` | EDL | treat standalone um/uh as pauses |
| `pacing.punch_in` / `punch_zoom` | EDL | `alternate`: every second island of a clip zooms by `punch_zoom` |
| `captions.replacements` | EDL | ASR fixes for burned-in and SRT captions |

Tightened pieces are named `c003`, `c003.2`, `c003.3`...; the first keeps the
chapter, all keep the priority, and B-roll overlays move to the piece that
plays their moment. `llmve schema edl` prints the full schema.

## Pacing presets
| Style | pacing | Feels like |
|---|---|---|
| tight (default) | tighten, max_gap 0.45-0.6, punch_in alternate | tutorial, commentary |
| natural | tighten, max_gap 0.9-1.2, punch_in off | interview, podcast, story |
| raw | tighten off | lecture, stream; remove only big blocks |

Typical keep ratio: tutorials 50-70 %, podcasts 80-95 %, rambling drafts 30-50 %.

## Outputs next to the render
- `<out>.mp4`: chapters embedded, +faststart, -14 LUFS / -1 dBTP
- `<out>.srt`: sidecar captions for the edited timeline (fillers removed)
- `<out>.chapters.txt`: description block
- `render/qc_report.json`: QC, `warnings` (chapter rules), `chapters`

## Traps
- Whisper can start a phrase 0.1-0.3 s late; if a cut sounds clipped, lower
  that clip's `start` slightly and let cut-audit re-snap it.
- `(retake?)` is a hint from repeated opening words; read the lines.
- A cold open replays material that plays again later. cut-audit keeps the
  later clip whole and reports `overlap ... (teaser or cold open)` as a warning;
  only partial overlaps are trimmed.
- A target duration set before tightening is a ceiling: tightening only shortens.
