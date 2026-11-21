---
name: b-roll-select
description: Index a local B-roll folder once (persistent broll_index.md + JSON, incremental rescans) and match B-roll cutaways to what is said in the edit. Use when the user has a B-roll library or the shoot contains silent clips, after planning and before cut-audit.
---

# b-roll-select

## 1. Index the library (incremental, safe to re-run)
```bash
uv run llmve broll index <library_dir>
```
This writes `<library>/.llmve_broll_index.json`, `<library>/broll_index.md`
and labelled thumbnail sheets at `<library>/.llmve_broll/sheet_NNN.png`. Only
new or changed files are probed.

## 2. Improve descriptions (the biggest win)
Matching is text-only, so descriptions are what matter, and a filename like
`C0042.MP4` produces no tags at all. Look at the thumbnail sheets (each tile
shows its item id) and describe what each clip shows:
```bash
uv run llmve broll describe <library_dir> <item_id> "hands pouring coffee, close-up, warm light" --tags coffee,morning,closeup
```
Manual descriptions survive rescans. Sidecar files (`clip.txt`, or
`clip.json` with `description` and `tags`) also work.

## 3. Match
```bash
uv run llmve broll match <ws> --library <library_dir> [--provider claude-cli] [--generate-missing]
```
This fills the planner's B-roll slots. If the planner made none, it creates
about one slot per 20 s, on the longest segments. Candidates are ranked with
BM25 keyword search over the descriptions, preferring the output's
orientation and clips not already used. `--provider` re-ranks the top 5 with
an LLM, using ids and text only. Slots with no good match stay empty. It
writes `broll_select.json`, updates `asset_plan.json` and rebuilds
`edl_raw.json`; run cut-audit next.

See `matching-rules.md` for the scoring details.
