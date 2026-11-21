# LLM Video Editor

Prompt-driven video editor: raw clips in, platform-ready cuts out (YouTube
16:9, Reels/TikTok 9:16), through checkpointed stages.

## Pipeline (each stage = one tested module + one CLI command)
`watch` → `plan` → `b-roll-select` → `cut-audit` → `take-geometry` → `review-gate` → `render`

- Code: `llm_video_editor/stages/*.py`; ffmpeg wrappers in `llm_video_editor/ffmpeg/`;
  checkpoint schemas in `llm_video_editor/schemas/`; LLM providers in `llm_video_editor/llm/`.
- Checkpoints live in `.llmve/<batch_hash>/` (see README for the file list).
- CLI: `uv run llmve --help`. The entry skill for editing requests is
  `.claude/skills/edit-video/SKILL.md`; `youtube-edit` (long-form 16:9) and
  `youtube-shorts` (vertical clips) specialise it.
- Long-form options live on the EDL: `pacing` (jump-cut tightening/punch-ins,
  applied in `stages/pacing.py` at the end of cut-audit), `EDLClip.zoom`,
  `EDLClip.chapter` (`stages/chapters.py`), `captions.replacements`.
- Motion graphics: `EDL.graphics` (HyperFrames compositions). Toolchain/rendering in
  `llm_video_editor/hyperframes/` (pinned versions, cached WebM, telemetry off);
  placement in `stages/graphics.py` and `ffmpeg/graph.add_graphics`. Always decode the
  WebM with `-c:v libvpx-vp9` (the native decoder drops alpha). Never commit GSAP.

## Rules for working in this repo
- Never read raw media (`*.mp4`, `*.mov`, `*.wav`, ...); use JSON checkpoints and
  contact sheets (`sheets/*.png`, `review/sheet_*.png`).
- Planner prompts carry text only (clip ids, transcript segments, B-roll
  descriptions). `llm.base.assert_text_only` enforces it; keep it that way.
- The LLM never invents timestamps: it selects segment ids, resolved locally.
- Run ffmpeg through `llm_video_editor.ffmpeg.runner` (argv lists, never `shell=True`).
- Review gate: Claude verifies every review tile itself and approves with
  `llmve review approve --notes "auto-verified: ..."` when all checks pass (fix and
  re-review otherwise; see `review-gate`). Never pass `--skip-review` / `--approve`,
  and stop for the user only if they ask to see the sheet.
- Checkpoint schema changes: bump nothing silently; update the model, its tests
  and `llmve schema` output together.

## Development
```bash
uv sync --extra dev --extra vision   # MediaPipe needs libegl1 on Linux
uv run ruff check .
uv run pytest                        # synthetic media; ASR/LLM/detector are faked
uv run llmve graphics doctor --install   # optional: enables the real HyperFrames tests
```
Python 3.11+, FFmpeg with libass/libfreetype. Optional extras: `vision`, `pro`,
`gen`, `graph`, `anthropic`, `openai`.
