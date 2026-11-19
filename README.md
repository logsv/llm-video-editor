# LLM Video Editor

Prompt-driven video editing for YouTube (16:9) and Reels/TikTok (9:16). You
point it at a clip or a folder of raw footage and describe the cut you want.
It transcribes, plans and audits the cut, keeps captions and overlays off the
speaker's face, shows you a preview sheet to approve, and then renders once,
normalised to -14 LUFS.

![LLM Video Editor Workflow](assets/video_editor_workflow.gif)

```bash
uv run llmve run ./raw_clips/ -p "10 minute YouTube edit: the three key ideas" -t youtube
# ... look at the review sheet it prints, then:
uv run llmve review approve .llmve/<batch_hash>
uv run llmve render .llmve/<batch_hash> -o final.mp4
```

## How it works

Each stage reads the previous stage's JSON checkpoint and writes its own
under `.llmve/<batch_hash>/`. The batch hash comes from the sorted
`(path, mtime, size)` of the inputs, so re-runs reuse finished work.

| Stage | Command | Writes | Notes |
|---|---|---|---|
| watch | `llmve watch <inputs>` | `transcript.json`, `clips_manifest.json`, `sheets/` | word-level faster-whisper (`--chunk-asr-over 10` splits clips over 10 min into ~5 min silence-aligned chunks, merged by timestamp; `--asr-workers N` runs them in parallel); scene/motion scores; clips with no speech tagged as B-roll; 4x4 contact sheets (0.5 FPS overview, 2 FPS on request via `watch-window`) |
| plan | `llmve plan <ws> -p ...` | `asset_plan.json`, `edl_raw.json` | map/reduce over the whole transcript; the LLM picks **segment ids**, never timestamps; only text leaves the machine |
| b-roll-select | `llmve broll index/describe/match` | `broll_select.json` | persistent `broll_index.md` in your B-roll folder, incremental rescans, BM25 + optional LLM re-rank |
| cut-audit | `llmve cut-audit <ws>` | `cut_audit.json`, `edl_audited.json` | deterministic: word-edge snapping, sentence completion, per-clip clamping, overlaps, duration budget |
| take-geometry | `llmve geometry <ws>` | `geometry_map.json` | MediaPipe face boxes per used scene; crop centre, caption band and PiP corner that avoid the face |
| review-gate | `llmve review <ws>` / `review approve` | `review/sheet_*.png`, `review/review.json` | one composited frame per cut/overlay in seconds; render refuses until the current cut is approved |
| render | `llmve render <ws>` | `render/<platform>.mp4`, `render/qc_report.json` | one encode; every segment normalised to 30 fps, square pixels, cover-crop/letterbox (never stretched), 48 kHz stereo; frame-exact concat; two-pass `loudnorm=I=-14:TP=-1.5:LRA=11`; NVENC/VideoToolbox when they actually work |

`llmve run` chains every stage and stops at the review sheet unless you pass
`--approve`. `llmve schema <name>` prints the JSON Schema of any checkpoint.

### Multi-clip shoots (e.g. 30 minutes of clips into a 10-minute edit)
Pass a folder or glob. Mixed cameras (30/60 fps, different resolutions and
codecs, phone rotation) are normalised per segment before the concat. Clips
without speech become B-roll candidates the planner can cut away to. Edits
with more than 24 cuts render in groups to near-lossless intermediate files
first, so memory stays bounded.

## Install

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/) and FFmpeg with libass
and libfreetype (`apt install ffmpeg` or `brew install ffmpeg`).

```bash
uv sync --extra dev                       # core pipeline + tests
uv sync --extra dev --extra vision        # + MediaPipe face detection (Linux: apt install libegl1)
```

| Extra | Adds |
|---|---|
| `vision` | MediaPipe face detector for take-geometry |
| `pro` | YOLO person fallback, music ducking (Demucs) |
| `gen` | generative B-roll fallback (Open-Sora / Stable Video Diffusion) |
| `graph` | LangGraph orchestration (`llm_video_editor.graph`) with a review interrupt |
| `anthropic`, `openai` | API planner providers |

## Planner providers

`--provider claude-cli` (the default) runs
`claude -p --output-format json --json-schema <schema> --tools "" --no-session-persistence`
in a clean temporary directory, with the prompt on stdin. It needs Claude Code
installed but no API key. The alternatives are `ollama` (local; `/api/chat`
with a JSON-schema `format`), `anthropic` and `openai`. Every provider returns
a validated Pydantic model. Prompts that mention a media file path are
rejected before they're sent.

## Claude Code skills

`.claude/skills/` exposes the pipeline to Claude Code: `edit-video` (entry
point), `watch`, `cut-audit`, `take-geometry`, `b-roll-select` and
`review-gate`. Each skill tells Claude which tested `llmve` command to run and
how to read its checkpoint. `.claude/settings.json` denies reads of raw
video/audio, so agents work only from JSON checkpoints and contact sheets.

## Programmatic use

See `examples/pipeline.py` for the stage functions, or use the LangGraph
graph, which pauses for review:

```python
from langgraph.types import Command
from llm_video_editor.graph import build_graph

app = build_graph()
cfg = {"configurable": {"thread_id": "edit-1"}}
state = app.invoke({"inputs": ["./raw_clips"], "prompt": "60s reel", "target": "reels"}, cfg)
print(state["__interrupt__"][0].value["review_sheets"])
final = app.invoke(Command(resume={"approve": True}), cfg)
print(final["output"], final["qc_passed"])
```

## Development

```bash
uv run ruff check .
uv run pytest
```
The integration tests generate synthetic multi-camera media with ffmpeg and
check A/V sync with flash/beep markers, -14 LUFS output, the review gate and
the full CLI run. ASR, the LLM and the face detector are replaced by fakes.
