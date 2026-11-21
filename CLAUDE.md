# LLM Video Editor

Prompt-driven video editor: raw clips in, platform-ready cuts out (YouTube
16:9, Reels/TikTok 9:16), through checkpointed stages.

## Working with Claude (the primary way to use this repo)
- Default flow: `uv run llmve auto <inputs> -p "<brief>" -t <platform> -o <out.mp4>` does
  the whole edit in one command and auto-approves its review; then the user gives
  notes and Claude applies them (`refine-edit` skill) and runs `uv run llmve finish <ws>`
  (audit -> review -> render -> publish, no re-transcription or re-planning).
- Skill routing: any edit request -> `edit-video`; one long recording for YouTube ->
  `youtube-edit`; clips/Shorts -> `youtube-shorts`; reactions to a render -> `refine-edit`;
  music/SFX/voice-over/B-roll/funny clips -> `royalty-free-media`; titles, thumbnails,
  description -> `youtube-publish`; titles/lower thirds/cards -> `motion-graphics`.
  Single-stage skills (`watch`, `cut-audit`, `take-geometry`, `review-gate`,
  `b-roll-select`) are for debugging one stage.
- After each run, show the user the review sheet and thumbnails (read the PNG/JPG),
  the QC line and every warning, and offer concrete next improvements.
- Local Python API / a future frontend: build on `stages.auto.auto_edit` and
  `finish_edit` (same options as the CLI); keep them free of CLI concerns.

## Agent mode (in progress: docs/agent-mode-plan.md)
- The agent is the editor; `llm_video_editor/agent/` holds its instruments. Tools execute,
  validate and report; they never make editorial choices. Do not add rules that decide
  for the agent: expose parameters and put advice in skills instead.
- `agent/tools.py`: the tool registry (open_project with `first` N clips, analyze,
  overview, transcript, speech_map, frames, edit_state, history, edit ops, batch, undo,
  audit (report only), review, render, publish, media, profile, remember). Every edit
  tool needs a `reason`; every edit is a version (`agent/edits.py`, `<ws>/edits/`) and is
  logged in `agent_log.jsonl` + `decisions.md`.
- `agent/mcp_server.py` serves the registry over MCP (`python -m llm_video_editor mcp`,
  project `.mcp.json`); `llmve tool <name> '<json>'` calls one tool from the terminal.
  Over MCP, `analyze`/`review`/`render` run as jobs in their own process
  (`agent/jobs.py`, `<ws>/jobs/`) and return a `job_id` for `job_status` when slow.
- Renders never go inside a footage folder (`workspace.check_output_path`); folder
  scans skip `.llmve` workspaces and earlier renders (`<video>.llmve.json`).
- Privacy: `stills` (requested frames/sheets/thumbnails may go to the model) or
  `text-only`; video files are never sent. Set by the user (`llmve profile --set
  privacy=...`), never by the agent.
- Cut-audit has an agent mode (`AuditConfig(editorial_fixes=False)`): editorial issues are
  reported as suggestions; only guardrails (unknown source, out of bounds) are applied.

## Pipeline (each stage = one tested module + one CLI command)
`watch` → `plan` → `b-roll-select` → (graphics, media) → `cut-audit` → `take-geometry` →
`review-gate` → `render` → `publish`; `auto` runs all of them, `finish` the part from cut-audit on.

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
- Media: `llm_video_editor/media/` (providers: Openverse, Pixabay, Pexels, Wikimedia;
  credits in `<ws>/media/credits.json`). Only licensed sources that allow commercial
  use and editing; never scrape social platforms or unlicensed web clips.
  `EDL.music` (music/sfx/voiceover) is mixed in render with ducking keyed to the
  speech bus, levelled to measured speech loudness (`ffmpeg/graph.add_music`).
  Cutaways: `stages/cutaways.py` (budgeted, may suggest none).
- Publish: `stages/publish.py` writes `<video>.upload.md/.json` and thumbnails
  (`stages/thumbnails.py`); location is opt-in (`--geocode`, `--use-location`).

## Rules for working in this repo
- Never read raw media (`*.mp4`, `*.mov`, `*.wav`, ...); use JSON checkpoints and
  contact sheets (`sheets/*.png`, `review/sheet_*.png`).
- Planner prompts carry text only (clip ids, transcript segments, B-roll
  descriptions). `llm.base.assert_text_only` enforces it; keep it that way.
- The LLM never invents timestamps: it selects segment ids, resolved locally.
- Run ffmpeg through `llm_video_editor.ffmpeg.runner` (argv lists, never `shell=True`).
- `llmve auto` / `llmve finish` approve the review automatically: the user chose this
  workflow, and the sheet is still written for reference. In step-by-step mode, do not
  approve a review or pass `--skip-review` / `--approve` unless the user asked.
- Checkpoint schema changes: bump nothing silently; update the model, its tests
  and `llmve schema` output together.

## Development
```bash
uv sync --extra dev --extra vision --extra agent   # MediaPipe needs libegl1 on Linux
uv run ruff check .
uv run pytest                        # synthetic media; ASR/LLM/detector are faked
uv run llmve graphics doctor --install   # optional: enables the real HyperFrames tests
```
Python 3.11+, FFmpeg with libass/libfreetype. Optional extras: `vision`, `pro`,
`gen`, `graph`, `anthropic`, `openai`.
