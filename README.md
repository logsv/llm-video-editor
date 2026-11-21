# LLM Video Editor

Prompt-driven video editing for YouTube (16:9) and Reels/TikTok (9:16). You
point it at a clip or a folder of raw footage and describe the cut you want.
It transcribes, plans and audits the cut, keeps captions and overlays off the
speaker's face, shows you a preview sheet to approve, and then renders once,
normalised to -14 LUFS.

![LLM Video Editor Workflow](assets/video_editor_workflow.gif)

## Quick start

**In Claude Code** (open this repo, then type a slash command):

```
/edit-video 10 minute YouTube edit of ./raw_clips: the three key ideas
/youtube-edit edit ~/Videos/raw-tutorial.mp4 for YouTube, about 15 minutes
/youtube-shorts make 3 shorts from ~/Videos/raw-tutorial.mp4
```

Claude runs the stages below, stops at the review sheet for your approval, and
then renders.

**From the terminal:**

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
| watch | `llmve watch <inputs>` | `transcript.json`, `clips_manifest.json`, `sheets/` | word-level faster-whisper (optionally chunked and parallel for long clips, see below); scene/motion scores; clips with no speech tagged as B-roll; 4x4 contact sheets (0.5 FPS overview, 2 FPS on request via `watch-window`) |
| plan | `llmve plan <ws> -p ...` | `asset_plan.json`, `edl_raw.json` | map/reduce over the whole transcript; the LLM picks **segment ids**, never timestamps; only text leaves the machine |
| b-roll-select | `llmve broll index/describe/match` | `broll_select.json` | persistent `broll_index.md` in your B-roll folder, incremental rescans, BM25 + optional LLM re-rank |
| cut-audit | `llmve cut-audit <ws>` | `cut_audit.json`, `edl_audited.json` | deterministic: word-edge snapping, sentence completion, per-clip clamping, overlaps, duration budget |
| take-geometry | `llmve geometry <ws>` | `geometry_map.json` | MediaPipe face boxes per used scene; crop centre, caption band and PiP corner that avoid the face |
| review-gate | `llmve review <ws>` / `review approve` | `review/sheet_*.png`, `review/review.json` | one composited frame per cut/overlay in seconds; render refuses until the current cut is approved |
| render | `llmve render <ws>` | `render/<platform>.mp4` (+ `.srt`, `.chapters.txt`), `render/qc_report.json` | one encode; every segment normalised to 30 fps, square pixels, cover-crop/letterbox (never stretched), 48 kHz stereo; frame-exact concat; two-pass `loudnorm=I=-14:TP=-1.5:LRA=11`; NVENC/VideoToolbox when they actually work |

`llmve transcript <ws>` writes a compact `transcript.md` (segment ids, times,
`(retake?)` markers, long pauses) for reading or planning by hand.

`llmve run` chains every stage and stops at the review sheet unless you pass
`--approve`. `llmve schema <name>` prints the JSON Schema of any checkpoint.

### Multi-clip shoots (e.g. 30 minutes of clips into a 10-minute edit)
Pass a folder or glob. Mixed cameras (30/60 fps, different resolutions and
codecs, phone rotation) are normalised per segment before the concat. Clips
without speech become B-roll candidates the planner can cut away to. Edits
with more than 24 cuts render in groups to near-lossless intermediate files
first, so memory stays bounded.

### Long recordings (chunked, parallel ASR)
By default every clip is transcribed in one pass. For long clips, turn on
chunking:

```bash
uv run llmve watch ./raw_clips/ --chunk-asr-over 10 --asr-workers 4
uv run llmve run ./raw_clips/ -p "..." --chunk-asr-over 10   # same flags on `run`
```

Clips longer than `--chunk-asr-over` minutes have their 16 kHz audio split
into chunks of about 5 minutes with 2 s of overlap. Each cut goes in the middle
of a pause found by ffmpeg `silencedetect` (within ±30 s of the nominal
boundary), or at the fixed point when there is no pause. Chunks run in a
process pool with one model per worker. `--asr-workers 0` (the default) means
1 worker on GPU and up to 4 on CPU.

Chunks are merged by timestamp, not by text: each word belongs to the chunk
whose interior holds its midpoint, and words that both chunks transcribed in
the overlap are kept once. Segment ids are re-numbered per clip, so
`transcript.json` looks the same as an unchunked run. Chunk settings (but not
the worker count) are part of the per-clip ASR cache key; clips below the
threshold keep their existing cache.

From Python, pass `chunking=ChunkConfig(...)` (`llm_video_editor.stages.chunked_asr`)
to `watch()`, or set `Deps(asr_chunking=ChunkConfig(...))` for the LangGraph
graph. `ChunkConfig` also exposes the chunk length, overlap, search window and
silence threshold.

### Long-form YouTube: pacing, punch-ins, chapters
One long recording into a finished 16:9 video (this is what `/youtube-edit`
runs):

```bash
uv run llmve watch ./raw/recording.mp4 --chunk-asr-over 10
uv run llmve transcript .llmve/<batch_hash>        # read transcript.md: retakes, pauses
uv run llmve plan .llmve/<batch_hash> -t youtube --target-duration 15m \
    -p "keep the last clean take of every line, cut setup chatter, open on the result"
# optional: edit edl_raw.json (cold open, chapters, pacing, caption fixes; see below)
uv run llmve cut-audit .llmve/<batch_hash>         # also applies pacing.tighten
uv run llmve geometry .llmve/<batch_hash>
uv run llmve review .llmve/<batch_hash>            # look at review/sheet_*.png
uv run llmve review approve .llmve/<batch_hash> --notes "looks good"
uv run llmve render .llmve/<batch_hash> --no-captions -o final.mp4
# -> final.mp4 (chapters embedded), final.srt, final.chapters.txt
```

Optional EDL fields in `edl_raw.json`:

```json
{"pacing": {"tighten": true, "max_gap": 0.6, "drop_fillers": true, "punch_in": "alternate"},
 "captions": {"enabled": false, "replacements": {"cloud": "Claude"}},
 "clips": [{"clip_id": "c001", "chapter": "Intro", "zoom": 1.0, "...": "..."}]}
```

- **Jump-cut tightening** runs as the last step of cut-audit: A-roll clips are
  split into speech islands at pauses longer than `max_gap` (standalone um/uh
  count as pauses with `drop_fillers`). Splits sit on word edges inside clips
  that already passed the sentence checks; overlays follow the piece that plays
  their moment.
- **Punch-ins**: `zoom` per clip, or `punch_in: alternate` to zoom every second
  island so jump cuts read as camera changes. Geometry and review use the same
  zoomed crop.
- **Chapters**: `chapter` on the clip that starts a topic. Render embeds them in
  the MP4, writes `<out>.chapters.txt` for the description and warns when
  YouTube's rules (first at 0:00, at least 3, each 10 s or more) are broken.
- **Cold opens**: a hook replayed from later in the recording is kept (cut-audit
  reports it as a warning instead of trimming the later clip).
- **Captions**: every render also writes `<out>.srt` (fillers removed,
  `captions.replacements` applied) for upload; `--no-captions` skips burning in.
- The `youtube` preset allows up to 12 hours; `shorts` is a 9:16 preset (180 s max).

### Shorts from a long video
One workspace holds one cut at a time, so make shorts one after another (this
is what `/youtube-shorts` runs):

```bash
uv run llmve transcript .llmve/<batch_hash>        # pick the moments (segment ids)
uv run llmve plan .llmve/<batch_hash> -t shorts --target-duration 40s \
    -p "segments 41-47 only, hook first"
uv run llmve cut-audit .llmve/<batch_hash>
uv run llmve geometry .llmve/<batch_hash>          # face-tracked 9:16 crop
uv run llmve review .llmve/<batch_hash>
uv run llmve review approve .llmve/<batch_hash>
uv run llmve render .llmve/<batch_hash> -o shorts/01-hook.mp4
```

### Motion graphics (HyperFrames)
Lower thirds, chapter title cards, keyword callouts or any custom animated
HTML/GSAP composition, rendered by [HyperFrames](https://github.com/heygen-com/hyperframes)
(Apache-2.0) to transparent VP9 WebM and composited in the render pass:

```bash
uv run llmve graphics doctor --install           # once: Node 22+, pinned HyperFrames + GSAP
uv run llmve graphics templates                  # lower-third, title-card, callout
uv run llmve graphics add .llmve/<batch_hash> --at c003 -t lower-third --offset 0.5 \
    --duration 4 --var title="Jane Doe" --var subtitle="Founder"
uv run llmve graphics chapters .llmve/<batch_hash>    # a title card per chapter
uv run llmve graphics add .llmve/<batch_hash> --at c004 -c ./my-composition --box 0.55,0.1,0.4,0.5
uv run llmve cut-audit .llmve/<batch_hash>       # then geometry, review, render as usual
```

- Graphics are `EDL.graphics` entries anchored to a clip (like B-roll overlays):
  cut-audit drops ones with a missing anchor or template, keeps them inside the
  cut and moves them with jump-cut tightening.
- Templates size themselves to the output frame (16:9 or 9:16); a custom
  composition is any HyperFrames project folder with an `index.html`.
- The review sheet gets a `gNNN over cNNN` tile per graphic; render draws them
  above B-roll and below burned-in captions. Renders are cached by content hash.
- The toolchain is pinned (`hyperframes@0.8.34`, `gsap@3.13.0`) and installed into
  `$LLMVE_TOOLS` (default `~/.cache/llmve`); telemetry and update checks are off.
  `HYPERFRAMES_BROWSER_PATH` points it at an existing Chrome/Chromium. GSAP is
  copied in at render time and never committed (it is not open source).
- Graphics are decoded with `libvpx-vp9`, because ffmpeg's built-in VP9 decoder
  drops the alpha channel and turns the transparent area black.

## Install

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/) and FFmpeg with libass
and libfreetype (`apt install ffmpeg` or `brew install ffmpeg`). Motion graphics
also need Node.js 22+ and an FFmpeg built with libvpx (standard builds are).

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

## Command reference

`<ws>` is the workspace directory `llmve watch` prints (`.llmve/<batch_hash>`).
Every command has `--help`.

| Command | What it does | Useful options |
|---|---|---|
| `llmve watch <inputs...>` | transcribe, score scenes, classify A-roll/B-roll, overview sheets | `--asr-model` (default `distil-large-v3`, English), `--language`, `--chunk-asr-over <min>`, `--asr-workers`, `--home` |
| `llmve watch-window <ws> <clip_id> <start> <end>` | dense contact sheets for one window | `--fps` (default 2) |
| `llmve transcript <ws>` | write `transcript.md` with ids, times, pauses, `(retake?)` | `--clip <clip_id>` |
| `llmve plan <ws> -p "<brief>"` | plan the cut -> `asset_plan.json`, `edl_raw.json` | `-t youtube\|shorts\|reels\|tiktok\|twitter\|linkedin`, `--target-duration 10m`, `--provider claude-cli\|ollama\|anthropic\|openai`, `--model` |
| `llmve broll index <library>` | index a B-roll folder (incremental) | `--rescan`, `--no-sheets` |
| `llmve broll describe <library> <item_id> "<text>"` | set a manual description | `--tags a,b,c` |
| `llmve broll match <ws>` | fill B-roll slots, rebuild `edl_raw.json` | `-l <library>`, `--provider`, `--model`, `--generate-missing` |
| `llmve cut-audit <ws>` | snap, fix, budget, apply pacing -> `edl_audited.json` | exits 1 if it fails |
| `llmve geometry <ws>` | face boxes, crop centres, caption/PiP zones | `--detector auto\|mediapipe\|yolo\|none`, `--samples` |
| `llmve graphics templates` | list built-in HyperFrames templates and their variables | |
| `llmve graphics add <ws> --at <clip_id>` | add a motion graphic to `edl_raw.json` | `-t <template>` or `-c <folder>`, `--var k=v`, `--offset`, `--duration`, `--box x,y,w,h` |
| `llmve graphics chapters <ws>` | a title card per chapter marker | `-t`, `--duration`, `--var`, `--no-numbered` |
| `llmve graphics render <ws>` | pre-render every graphic (cached) | `--no-draft` |
| `llmve graphics doctor` | check Node, npm, HyperFrames, libvpx | `--install` |
| `llmve review <ws>` | build the review sheet (status: pending) | |
| `llmve review approve <ws>` / `reject <ws>` | record the decision | `--notes "..."` |
| `llmve render <ws>` | one encode, -14 LUFS, QC, SRT, chapters | `-o out.mp4`, `--encoder auto\|h264_nvenc\|h264_videotoolbox\|libx264`, `--draft`, `--no-captions`, `--skip-review` |
| `llmve run <inputs...> -p "<brief>"` | watch -> plan -> b-roll -> cut-audit -> geometry -> review | all of the above plus `--broll-library`, `--approve` (approve and render) |
| `llmve schema <name>` | JSON Schema of a checkpoint (`edl`, `transcript`, `qc_report`, ...) | |

## Claude Code skills

`.claude/skills/` exposes the pipeline to Claude Code: `edit-video` (entry
point), `watch`, `cut-audit`, `take-geometry`, `b-roll-select`,
`review-gate`, `motion-graphics`, and two specialised entry points: `youtube-edit` (one long
recording into a 16:9 YouTube video with cold open, pacing, chapters and SRT)
and `youtube-shorts` (vertical clips from a long video). Each skill tells
Claude which tested `llmve` command to run and how to read its checkpoint.

| Slash command | Use it for |
|---|---|
| `/edit-video` | any edit from a brief (entry point; runs the whole pipeline) |
| `/youtube-edit` | one long recording -> 16:9 YouTube video with chapters and SRT |
| `/youtube-shorts` | the best 20-60 s moments of a long video -> vertical clips |
| `/motion-graphics` | lower thirds, chapter cards, callouts, custom HyperFrames compositions |
| `/watch`, `/cut-audit`, `/take-geometry`, `/b-roll-select`, `/review-gate` | one stage on its own |

Claude never approves the review sheet for you unless you tell it to. `.claude/settings.json` denies reads of raw
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
uv sync --extra dev --extra vision   # MediaPipe needs libegl1 on Linux
uv run ruff check .
uv run pytest                        # or: uv run pytest tests/unit -q
```
The integration tests generate synthetic multi-camera media with ffmpeg and
check A/V sync with flash/beep markers, -14 LUFS output, the review gate and
the full CLI run. ASR, the LLM and the face detector are replaced by fakes.
