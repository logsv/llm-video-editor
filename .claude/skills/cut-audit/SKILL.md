---
name: cut-audit
description: Verify and repair a rough cut before rendering - snaps in/out points to word edges, extends or trims cuts that end mid-sentence, clamps each cut to its own source clip, merges overlaps, drops slivers and enforces the duration budget. Use after planning or after any edit to edl_raw.json.
---

# cut-audit

```bash
uv run llmve cut-audit <ws>
```
This is deterministic: there's no LLM involved. It reads `edl_raw.json`,
`transcript.json` and `clips_manifest.json`, and writes `edl_audited.json`
and `cut_audit.json`.

## Reading the result
Each violation has `kind`, `severity`, `clip_id`, `source_file`, `at`,
`message`, `fixed` and `fix`:

| kind | Auto-fix |
|---|---|
| `mid_word_in` / `mid_word_out` | moved to the word edge |
| `mid_sentence_out` | extended to the sentence end (up to 2.5 s), or trimmed back to the previous one |
| `out_of_bounds` | clamped to that clip's own duration (or dropped) |
| `overlap` | merged (adjacent) or trimmed (repeated material) |
| `too_short` | dropped (under 0.6 s) |
| `duration_budget` | lowest-priority cuts dropped until within 5% |
| `overlay_out_of_bounds` | overlay re-bounded or dropped |
| `unknown_source` | dropped |

## Pass/fail loop
- `PASSED` (exit 0): continue to take-geometry.
- `FAILED` (exit 1): only unfixed `error`s block. Usually the cut is still over
  budget after dropping clips, or a sentence end is out of reach. Fix
  `edl_raw.json` (times must come from the segment and word times in
  `transcript.json`), or re-run `llmve plan` with a tighter brief. Then run
  cut-audit again, and repeat until it passes. Never edit `edl_audited.json`
  by hand.
