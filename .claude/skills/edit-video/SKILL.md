---
name: edit-video
description: Turn raw footage (a file, a folder of clips, or a glob) into a platform-ready cut from a natural-language brief, e.g. "10-minute YouTube edit from ./raw_clips" or "60s reel of the top 3 takeaways". Runs the checkpointed llmve pipeline stage by stage and owns the approval points. Use this as the entry point for any video-editing request in this repo.
---

# Edit a video with the llmve pipeline

Specialised entry points that follow this pipeline: `youtube-edit` for one
long recording -> 16:9 YouTube video (cold open, jump-cut pacing, chapters,
SRT) and `youtube-shorts` for clipping vertical Shorts out of a long video.

You orchestrate deterministic, tested stages. You never open video or audio
files (`.claude/settings.json` blocks them); you work only from the JSON
checkpoints and PNG contact sheets the stages write under
`.llmve/<batch_hash>/`.

## 0. Clarify the brief (only what you cannot infer)
Ask only for what you can't infer, at most these three: the target platform
(`youtube`, `reels`, `tiktok`, `twitter`, `linkedin`), the target duration,
and whether a local B-roll library exists.

## 1. Run the stages in order
| Step | Skill | Command | Gate |
|---|---|---|---|
| 1 | `watch` | `uv run llmve watch <inputs>` | prints the workspace path; check `clips_manifest.json` |
| 2 | (plan) | `uv run llmve plan <ws> -p "<brief>" -t <platform> --target-duration 10m` | `edl_raw.json` exists |
| 3 | `b-roll-select` | `uv run llmve broll match <ws> --library <dir>` | optional |
| 3b | `motion-graphics` | `uv run llmve graphics add <ws> --at <clip> -t <template> ...` | optional |
| 3c | `royalty-free-media` | `uv run llmve music add ...`, `uv run llmve media suggest <ws>` | optional |
| 4 | `cut-audit` | `uv run llmve cut-audit <ws>` | must print `PASSED` |
| 5 | `take-geometry` | `uv run llmve geometry <ws>` | every cut has a placement |
| 6 | `review-gate` | `uv run llmve review <ws>` | **stop and show the sheet to the user** |
| 7 | render | `uv run llmve render <ws> -o <out.mp4>` | `QC PASSED` |
| 8 | `youtube-publish` | `uv run llmve publish <ws>` | `<out>.upload.md` + 3 thumbnails |

`uv run llmve run <inputs> -p ... -t ... [--broll-library <dir>]` chains steps
1-6 and stops at the review sheet (exit code 2). Prefer the step-by-step form
when something needs fixing between stages.

## 2. Rules
- Never approve the review for the user unless they told you to. The full
  render is the slow, expensive step; the review sheet is there so a person
  catches problems before it runs.
- If cut-audit fails, read `cut_audit.json`. Then either re-plan with a
  sharper brief or edit `edl_raw.json` (segment times must come from
  `transcript.json`), and run cut-audit again. Never hand-edit `edl_audited.json`.
- The planner (`--provider claude-cli`, the default) only ever receives text:
  clip ids, transcript segments and B-roll descriptions. Never put media file
  paths in a prompt; the provider layer rejects them.
- At the end, report the QC line (duration, A/V offset, LUFS) and any
  `review.json` warnings to the user.
