# Agent mode: plan

Status: P0 (foundation) is done and merged. On `claude/agent-mode-fixes`: fixes from the
first real run, place/route and object tools, place and label graphics, text safety
checks, and long (2-50 min) cuts from mostly silent footage. Everything else is listed in
[section 11, Remaining work](#11-remaining-work), the to-do list for this plan.

## Goal

The agent is the editor. It watches the footage through its tools, decides every
editorial and visual choice, writes code when no tool fits, reviews its own render
and delivers. Python, ffmpeg, HyperFrames and other libraries are its instruments,
not decision makers. The current rule-based `llmve auto` stays only as a fallback
when no agent is available.

## Decisions (agreed)

| Topic | Decision |
|---|---|
| Who decides | The agent, for everything that shapes the video |
| Plan checkpoint | Auto-approved: the agent writes its plan to the log and carries on to the final render |
| Code | The agent may install Python libraries and write its own scripts |
| What reaches the model | Text plus small still frames / contact sheets the agent asks for. No video files or streams are ever uploaded; no Files API, no Managed Agents. `text-only` stays available as a setting |
| Where it runs | On the user's machine: Claude Code (interactive) or `claude -p` (one command), tools local |
| First test | The Bhopal -> Khajuraho solo bike trip footage, a 10-minute YouTube video, run on the user's Mac |

## Reference studied: tenfoldmarc/video-edit-skill

A Claude Code skill that turns phone clips into Instagram reels. It has no licence
file, so we reuse its ideas, not its code, templates or assets. What it gets right,
and what we take from each point:

| In video-edit-skill | What we adopt |
|---|---|
| Claude picks takes, writes the EDL, designs every look; scripts only ingest, transcribe, assemble, cut out, measure head position and check cuts | Same split. Our tools become small, single-purpose instruments; no taste lives in code |
| `build.py` is a finished real project the agent copies and rewrites ("keep the machinery, rewrite the content") | Project build scripts the agent writes from templates: shared machinery (timing helpers, render, audio, grades) plus per-project content it authors |
| `references/` with exact values: looks (filter values, fonts, sizes, motion), layout (safe zone, placement formulas), gotchas ("each one cost a render") | An editor's handbook skill with the same three kinds of file, written for long-form YouTube and travel, plus a gotchas file the agent appends to after every project (memory that grows) |
| "Defaults that came out of real feedback (do not re-litigate)" | A user style profile the agent updates from accepted notes (e.g. "no film grain", "captions centred"), read at the start of every edit |
| Seven hard rules: safe zone, never cover the face, captions say what was said, subtle SFX, credit creators, prove cuts from the final render, never delete the user's files | Our minimal guardrails (section 4) take the same shape: few, absolute, about correctness and respect, not taste |
| Take selection: people repeat a line until it lands, keep the last clean take; in/out from a speech map with silence detection; re-transcribe unsure words with a bigger model and the names in the prompt | Perception tools: speech map (words + silences), targeted re-transcribe; the selection itself is the agent's call |
| Punch-ins cropped from 4K; wide/close alternation like a studio interview | Visual-variety knowledge in the handbook; crop/zoom as tool parameters |
| Animated inserts built by parallel sub-agents from a strict brief (fixed size, duration, beats locked to word times, transparent WebM, report what looks off) | Specialist sub-agents with brief templates: graphics/insert builder, colourist pass, sound designer, thumbnail designer; the lead editor merges and fixes from their notes |
| Proof only from the final render: before/after frame strip at every cut, safe-zone guide in snapshots only, audio peak check | Self-review tools that read the rendered file: cut strips, continuity, loudness/peaks, safe areas, plus a 720p review copy |
| Every round is a fast re-render from notes; versions kept in `renders/`, assets backed up before re-assembly | Versioned edits and renders, undo per decision, fast draft renders for review |
| First-run setup detects the machine and writes the exact commands; config with name, handle, default look, folders | A `llmve setup` the agent runs first: machine record (ffmpeg, Node, GPU, encoders), user profile (channel, style, folders) |
| The whole reel is a HyperFrames composition (CSS grades, cutout layers, kinetic type) | Not for the whole long-form render: Chrome screenshot rendering is fine for 25 s but too slow for 10+ minutes at 1080p/4K. We keep the ffmpeg renderer for picture and sound, and HyperFrames for graphic layers (already built). Subject cut-out (HyperFrames remove-background, local) is a later optional tool for text behind the subject |

## 1. Architecture

```
user brief / notes
      |
  Lead editor agent (Claude Opus 5.5, Claude Code or claude -p, on the user's machine)
      |-- reads: handbook skill, user style profile, gotchas memory
      |-- calls: typed tools (local tool server, `llmve mcp`)
      |-- writes: project scripts in the workshop (Python, may install libraries)
      |-- delegates: specialist sub-agents with briefs (graphics, colour, sound, thumbnails)
      v
  edit state (versioned edl.json + assets + decision log)
      |
  renderer (ffmpeg picture/sound, HyperFrames graphic layers) -> self-review tools -> deliver
```

One tool server serves both hosts, so interactive and one-command runs behave the
same. Everything runs locally; only text and requested stills go to the model.

- Built by wrapping the existing tested stages (watch, plan, cut-audit, geometry,
  review, render, publish, media, graphics, music): no parallel implementation.
  Rules move out of them into parameters; their correctness code stays.
- A future local API / frontend calls the same tool server and the same agent entry
  point (`llmve agent` as a library function), with no CLI-only logic.
- The existing skills (`edit-video`, `refine-edit`, `youtube-edit`, `youtube-shorts`,
  `royalty-free-media`, `motion-graphics`, `youtube-publish`) are rewritten to drive the
  agent tools; notes in chat keep working through `refine-edit`.

## 2. The toolbox (instruments, no decisions)

Looking (read-only):
- project overview: clips, durations, speech/no speech, devices, GPS, dates
- transcript with word times in the spoken language (Hindi, English, Hinglish; detected or forced); speech map (words and silences, voice activity, not just loudness); re-transcribe a range with a bigger model and a name list
- location track: GPS per clip (and per point where the file has a track), place names via opt-in reverse geocoding, route order; recording times across cameras with clock-offset correction
- frames and contact sheets for any clip and time window (images, logged)
- measurements per scene: exposure, white balance, saturation, noise, shake, motion, faces, scene-change scores
- audio: loudness, peaks, noise floor, music/speech detection, beat times
- current edit, version history, decision log

Editing (validated, versioned, undoable; every call carries a reason):
- clips: add, remove, trim, split, reorder, speed (slow motion, ramps), stabilise
- framing: crop, zoom/punch-in, vertical reframing
- colour: correction per clip (from measurements), match clips, looks (parameters, curves, LUT)
- transitions per cut: hard cut, crossfade, dip to black, whip/slide, J-cut, L-cut
- audio: noise reduction (wind, engine), EQ, levels; music, sound effects, voice-over (recorded by the user, or local TTS with disclosure); ducking that follows speech from both the clips' own voice and the voice-over (speech detected by voice activity, so wind or engine noise does not hold the music down), depth and timing set by the agent
- deliverables: several outputs per project (the 10-minute video plus Shorts from the same footage), each with its own edit, render and publish package
- cutaways (licensed search or exact placement), graphics (HyperFrames), captions, chapters

Checking and delivery:
- audit: reports problems (mid-word cuts, out-of-range times, missing credits, safe-area breaks) and never fixes silently
- draft render of a range; cut strips and continuity frames from the render; loudness/peak report; compare versions
- final render; review copy (720p); publish package (titles, description, tags, thumbnails)

Workshop:
- a scratch folder per project where the agent writes and runs Python (ffmpeg, OpenCV, librosa, PIL, anything it installs)
- outputs come back as registered assets (a clip, a LUT, a cut list, a mask) the edit can use
- each script and installed package (with version) is recorded in the decision log so a run can be repeated

## 3. Knowledge (what guides the agent's taste)

- Editor's handbook skill: storytelling for long-form and travel, pacing, shot variety,
  colour looks with exact values, transitions and when they help, sound (music levels,
  ducking, SFX restraint), YouTube specifics (chapters, end-screen area, thumbnails).
- User style profile: channel, audience, default look, likes and dislikes learned from notes.
- Gotchas memory: technical lessons from real renders, appended by the agent after each project.

These are advice the agent weighs, not rules code enforces.

## 4. Guardrails (the only things code enforces)

1. Frame-accurate timing; nothing outside a clip's real duration; valid output files.
2. Never modify or delete the user's footage; outputs go to the project folder.
3. No video files or streams to the model; stills only through the frame tool (or none in `text-only`).
4. Licensed media only (commercial use and editing allowed); every third-party asset has a credit; copyrighted music detected in the original audio is flagged before publish.
5. Disclosure: AI-generated narration, B-roll or images are marked for YouTube's altered/synthetic content question.
6. Captions say what was said, in the language it was said.
7. Budgets: tokens, time, iterations per run; agent-written code runs only inside the project folder and its own virtual environment.

Everything the current code decides by rule (pause thresholds, cutaway budgets, music
volume, caption style, crop centre, chapter rules, cut-audit auto-fixes) moves to tool
parameters or handbook advice.

## 5. How an edit runs

1. Setup: machine record, user profile, handbook, gotchas.
2. Understand: transcript, speech map, scene measurements, contact sheets; the agent's own notes on story and quality.
3. Plan: story, structure, length, look, sound, inserts; written to the log, auto-approved.
4. Edit: tool calls, workshop scripts, sub-agent briefs, each producing a new version.
5. Self-review: draft render, cut strips, continuity, loudness, safe areas; fix and repeat until satisfied or out of budget.
6. Deliver: final render, review copy, publish package (upload.md next to the video: SEO titles, description with chapters and credits, tags, location details, three honest click-worthy thumbnails), decision log (`decisions.md`), cost report, gotchas update. Uploading to YouTube itself stays manual.
7. Notes from the user restart at step 4.

## 6. Model, cost, caching

- Lead editor and sub-agents: `claude-opus-5-5`, adaptive thinking, effort `high`.
- Stills only on request (a 16-frame sheet is about 1.8k input tokens); text-only mode uses none.
- Stable tool list and handbook first in the prompt so they cache; notes arrive as mid-conversation messages.
- Task budgets cap a run; long sessions use compaction.
- Rough estimate: $1-5 of model usage per 10-minute video, to be measured in P5.

## 7. Phases

| Phase | Delivers | Done when | Status |
|---|---|---|---|
| P0 Foundation | Stages split into small tools with parameters instead of rules; audit reports only; versioned edits, undo, decision log; `llmve mcp` tool server; `llmve setup` (machine record, user profile); privacy setting | Claude Code reproduces today's edit quality by calling tools only and choosing every parameter | Done (headless permissions for `claude -p` still open) |
| P1 Perception | Speech map, re-transcribe, frame/sheet viewer, scene measurements, audio analysis and beats | The agent can describe every clip of the trip footage and justify its selects | Partly: frames, speech map, GPS route and place names, local object detection |
| P2 Craft | Colour, transitions (incl. J/L-cuts), speed, stabilise, audio cleanup and mix; workshop with registered outputs | Each tool's render matches the request (frame and audio checks) | Partly: graphics safety checks, place/label templates; transitions, colour, speed, audio cleanup and voice-over to do |
| P3 Editor agent | `llmve agent <clips> -p "<brief>"`; handbook, style profile, gotchas memory; plan -> edit -> self-review loop; specialist sub-agents with briefs | One command produces a finished edit with a readable decision log and no rules in the decision path | To do |
| P4 Visual story | Story building from scenes as well as speech (travel, B-roll-heavy footage) | The Bhopal -> Khajuraho footage becomes a coherent 10-minute video | Partly: `plan` fills long cuts with silent footage (rule-based); agent-side scene story to do |
| P5 Evaluation | Test shoots (talking head, travel, multi-cam, screen recording), quality and cost metrics, recorded runs replayed as regression tests | The agent beats the rule-based pipeline on the metrics, with measured cost per video | To do |

## 8. First test: Bhopal -> Khajuraho

- Run on the user's Mac (the footage never leaves it): `llmve agent "$HOME/Documents/Video/Bhopal to Khajuraho" -p "10-minute solo bike trip vlog" -t youtube`.
- Expect mostly riding and scenery with some talking, so it exercises P1 (scene understanding) and P4 (visual story) most.
- Check: a 10-minute cut with a route story, consistent colour across cameras, transitions that fit location changes, music under riding, the user's voice clear where they talk, chapters by place, credits complete, decision log readable.
- The user's notes from this run seed the style profile and the first gotchas.


## 9. Requirements traceability

Every requirement given during planning, and where the plan covers it.

| # | Requirement | Where in the plan |
|---|---|---|
| R1 | Long-form YouTube editing from raw footage | Goal, sections 2, 5; first test |
| R2 | Use through Claude Code skills/commands; Claude workspace is the primary way | Section 1 (hosts, skills rewritten) |
| R3 | One command end to end, auto-approved | `llmve agent`, decisions table, section 5 |
| R4 | Interactive improvement from notes, with the right skills and tool calls | Section 1 (`refine-edit`), section 5 step 7 |
| R5 | The agent decides everything (scenes, colour, transitions, clip creation, ...); code is only its instruments | Goal, section 4 (rules moved to parameters/advice) |
| R6 | The agent may install libraries and write code | Workshop (section 2), guardrail 7 |
| R7 | No video uploaded to the cloud; some stills to the model are fine | Decisions table, guardrail 3 |
| R8 | Royalty-free music and SFX; YouTube Audio Library / Creator Music (no API: manual registration) | Section 2 (audio, cutaways), guardrail 4 |
| R9 | Royalty-free B-roll and funny clips, only when needed, licensed sources only | Section 2 (cutaways), handbook advice on restraint |
| R10 | Music ducking under both the voice-over and the video's own voice | Section 2 (audio) |
| R11 | Upload markdown next to the video: SEO title/description, music credits, location, details, three thumbnails | Section 5 step 6 |
| R12 | HyperFrames for graphics | Section 1 (renderer), reference table |
| R13 | Local API / frontend later, kept in scope | Section 1 |
| R14 | Enhance existing code, do not duplicate it | Section 1 (wrap existing stages) |
| R15 | Shorts / vertical as well as long-form | Section 2 (deliverables, vertical reframing) |
| R16 | First test: Bhopal -> Khajuraho, 10 minutes, on the user's Mac | Section 8 |
| R17 | Captions (.srt), chapters, -14 LUFS loudness kept from the current pipeline | Section 2 (checking and delivery), handbook |

## 10. Edge cases

| Area | Case | Handling | Phase |
|---|---|---|---|
| Footage | Mixed cameras: phone (variable frame rate, HDR HLG/Dolby Vision), action cam (fisheye, 60/120 fps, 10-bit HEVC), drone (log profile), helmet/dash cam | Measurement tools report colour space, transfer, frame rate and lens; colour tool tone-maps HDR/log to SDR before grading; high-fps clips offered as slow motion | P1, P2 |
| Footage | Action-cam recordings split across files (e.g. GoPro chapters every ~4 GB) | Overview detects chapter sequences and treats them as one continuous recording | P0 |
| Footage | Rotation metadata, odd sizes, corrupt or partial files, clips without audio, mono/5.1, 44.1/48 kHz | Probe reports them; corrupt files are skipped with a warning; audio normalised in the renderer (already done) | P0 |
| Footage | Hours of riding (very large total) | Proxies and sampled analysis first, detail only where the agent looks; caching; progress reporting; disk-space check before proxies and renders | P1 |
| Footage | Clock differences between cameras, time zones | Offset correction from GPS/time tags or a shared moment; order by corrected time for the route | P1 |
| Footage | Duplicate or near-duplicate clips | Detected and reported; the agent chooses | P1 |
| Footage | Too little usable footage for the target length, or far too much | The agent reports and decides (shorter cut, slower pacing, more scenery); never pads with filler silently | P3 |
| Audio | Wind and engine noise on a bike; speech under a helmet | Noise reduction tool; ASR on cleaned audio; voice activity detection for the speech map and ducking | P1, P2 |
| Audio | Copyrighted music in the background (radio, temple loudspeakers, a cafe) | Music detection in original audio; flagged in the decision log and publish warnings; the agent mutes, ducks or replaces | P1 |
| Language | Hindi or mixed Hindi-English speech | Language-aware ASR model; captions and upload copy in the chosen language(s); Devanagari-capable fonts for captions, graphics and thumbnails | P1, P2 |
| Privacy | Faces of strangers, number plates | Optional blur tool the agent can use; handbook advice | P2 |
| Privacy | Location in public metadata | GPS read locally; reverse geocoding and location in copy stay opt-in (as today) | P0 |
| Agent | Model refusal, rate or usage limits (long runs on a Claude subscription), network loss mid-run | Every step versioned; `llmve agent --resume` continues from the last version; the rule-based fallback can finish the edit | P3 |
| Agent | Invalid tool calls, endless loops, budget running out mid-edit | Tools return errors the agent can fix; iteration and token caps; finish with the best version so far and say what is missing | P3 |
| Agent | Runs differ between attempts | Decision log, installed packages and scripts recorded; recorded runs replayable in tests | P3, P5 |
| Agent | Headless `claude -p` needs tool permissions | Define the allowed tools and working folder for agent runs explicitly; nothing outside the project folder | P0 |
| Code | Agent-written scripts and installed packages | Project virtual environment only (pinned versions), never global installs; scripts cannot touch the footage folder except to read | P2 |
| Code | User edits the edit while the agent runs | Version check before each write; conflicting writes rejected and reported | P0 |
| Render | 10-minute 4K render times, transitions forcing re-encodes | Draft renders at low resolution for self-review; hardware encoders; final render once | P2 |
| Render | Self-review on edits with hundreds of cuts | Sampled cut strips plus every cut the audit flags | P3 |
| Data | Existing checkpoints and edit files | Schema changes versioned with tests and `llmve schema` output, as CLAUDE.md requires | P0 |
| Delivery | Several deliverables from one trip (long video plus Shorts) | Separate edits per deliverable in one project, sharing analysis | P3 |
| Delivery | AI narration or generated B-roll | Disclosure flag in the publish package | P3 |
| Cost | Visibility of model spend | Per-run token and cost report next to the decision log | P3 |

## 11. Remaining work

The to-do list, in suggested order. Each item keeps the agent-mode rule: tools execute,
check and report, and the agent decides.

### Done so far (for reference)
- **P0:**
  - versioned edits with undo and the decision log;
  - atomic edit tools plus `batch`;
  - report-only audit;
  - review, render and publish tools;
  - MCP server plus `llmve tool`;
  - profile, machine record and privacy setting;
  - `first` N clips;
  - footage inspection (HDR, VFR, rotation, audio, split recordings, GPS).
- **Fixes from the first Mac run:**
  - background jobs plus `job_status` for analyze, review and render;
  - `batch` takes the same params as the standalone tools;
  - media results kept from every search;
  - `python -m llm_video_editor` launcher;
  - renders refused inside footage folders, and earlier renders skipped by scans;
  - absolute workspace paths.
- **On-screen text:**
  - `places` (route, stops, distances; place names and nearby temples, forts, forests, parks, rivers and highways, with the user's `location_lookup` permission);
  - `objects` (local YOLO);
  - `location` and `label` templates;
  - `graphics_templates`;
  - review reports on-screen text coverage.
- **Text safety:**
  - per graphic, review measures contrast, busy background, size, title-safe area (16:9 and 9:16), and face/caption overlap;
  - it also gives the shot's palette and suggested colours;
  - `accent: "auto"` picks the colour from the shot.
- **Length:**
  - `plan` fills targets that speech can't reach with shots from clips without speech, in recording order;
  - YouTube 2-50 min cuts work when the footage covers them.

- **Second Mac run fixes:**
  - images in MCP replies shrunk (JPEG, 1568 px) and capped per reply;
  - `text_color` on every template, and `style: card` on `location`;
  - music, SFX and voice-over levelled to the measured speech;
  - words ASR invented (low confidence, wrong script) flagged as doubtful in `transcript`, `speech_map` and audit.

### 11.1 Next: problems seen in the Bhopal -> Khajuraho run
| Item | Why | Notes |
|---|---|---|
| Transitions | Only hard cuts exist; the trip edit asks for crossfades between places | Per-cut `transition` on the EDL: cut, crossfade, dip to black/white, whip/slide (duration set by the agent); J-cut and L-cut (audio leads or trails the picture). Render with `xfade`/`acrossfade`; review tiles show mid-transition frames; audit reports transitions longer than either clip |
| Flip / rotate clips | Helmet camera mounted upside down, no rotation tag | `set_clip` fields `rotate` (90/180/270) and `flip` (h/v), applied in the fit filter, review and geometry |
| Whisper hallucinations on wind and engine noise | Invented speech in silent riding clips ("鱼", "I'm" over wind) | Partly done: low-confidence and wrong-script words are flagged doubtful. Still to do: voice-activity speech map (Silero VAD or similar) marking words outside voice activity; `analyze` option to drop them |
| Copyrighted audio in the footage | Sound & Light show narration, temple loudspeakers, radio | Music/speech classification of clip audio; flagged in audit and publish; the agent mutes, ducks or replaces |
| Shot inventory and balance report | The 7-minute brief asks for balanced road / animal / river / people / temple shares | `inventory` tool: categories per clip from objects, places, speech and the agent's own labels (stored); `balance` report on the current edit (seconds and share per category, longest run of one category). The agent sets the target shares |
| Place titles in `auto` mode | `auto` adds no place titles by itself | With `location_lookup` on, one `location` title per new stop (`--place-titles`), passed through the text safety check |

### 11.2 AI voice-over (P2)
- **Already there:**
  - `MusicTrack(role="voiceover")`;
  - music ducks under the voice-over as well as the clips' own voice.
- **To build:**
  - `voices` tool: engines and voices available, language, licence, whether each runs locally or in the cloud.
  - `voiceover` tool: lines in (text, voice, speed, style), audio out; reports each line's exact length; cached by text and voice; short audition sample.
  - `pronounce` tool: a list of place names (Khajuraho, Orchha, Chhatarpur) and how to say them.
  - Placement: lines anchored to clips, in gaps found by the speech map.
  - Audit reports:
    - a line longer than its gap;
    - a line over the clips' speech;
    - a line running past its shot;
    - a line whose anchor clip was removed.
  - Mix: the clip's own sound dips slightly under the voice-over (new), and lines are levelled to the measured speech loudness.
  - Captions: voice-over lines go into the burned-in captions and the `.srt` file.
  - Publish: AI-voice disclosure in `upload.md`, with a reminder of YouTube's altered/synthetic content checkbox when the voice is realistic (always when it is a copy of the user's voice).
- **Engines:**
  - local Kokoro by default (licence allows commercial use, Hindi voices);
  - Piper as a light alternative;
  - macOS `say` as a fallback;
  - cloud (ElevenLabs, OpenAI, Azure, Google) only when the user sets `tts=cloud`, which sends the script text only.
  - Voice-copy models with non-commercial licences (XTTS, F5-TTS) are not used for monetised channels.
  - Check every licence again at implementation.
- **Consent:**
  - copying a voice only from the user's own recorded sample, with explicit consent, never anyone else's;
  - `tts` is the user's setting, never the agent's.
- **Phases:**
  - V1: local engine plus tools, audit checks, mix, captions, disclosure;
  - V2: cloud engines and a copy of the user's own voice;
  - V3: `auto --voiceover auto` and narration advice in the skills.
- **Waiting on the user:**
  - language (English, Hindi, Hinglish);
  - voice (generic narrator or a copy of their own voice);
  - engine (local first, or a cloud engine from the start);
  - amount (a few lines between scenes, or narration over most of the riding).

### 11.3 P1 Perception (rest)
- Re-transcribe a range with a bigger model and a name list; Hindi/Hinglish models; Devanagari fonts for captions, graphics and thumbnails.
- Scene measurements: exposure, white balance, saturation, noise, shake, motion.
- Audio: loudness, peaks, noise floor, beat times (cutting to music).
- Clock-offset correction between cameras; near-duplicate detection; proxies and sampled analysis for many hours of footage; disk-space checks.

### 11.4 P2 Craft (rest)
- Colour: per-clip correction from measurements, matching across cameras, looks (curves, LUT), HDR/log to SDR tone mapping.
- Speed (slow motion from 60/120 fps, ramps) and stabilisation.
- Audio cleanup: wind and engine noise reduction, EQ, levels.
- Blur tool for strangers' faces and number plates.
- Workshop: scratch folder for agent-written Python, outputs registered as assets, scripts and package versions logged; project virtual environment only.
- Self-review from the render: cut strips, continuity frames, loudness/peak report, 720p review copy, version compare.

### 11.5 P3 Editor agent
- `llmve agent <clips> -p "<brief>"`: one command running `claude -p` with the MCP server, the handbook and an explicit allowed-tool list (headless permissions).
- Handbook and skills rewritten for agent mode (story, pacing, balance, text, music, transitions), style profile and gotchas memory fed from accepted notes.
- Plan -> edit -> self-review loop; `--resume` from the last version; iteration and token budgets; cost report per run.
- Several deliverables per project (the long video plus Shorts), sharing analysis.
- Disclosure flag for AI narration or generated B-roll.

### 11.6 P4 Visual story and P5 Evaluation
- P4: the agent builds the story from scenes as well as speech (route order, balance targets, establishing -> detail -> reaction patterns), replacing the rule-based fill for agent runs.
- P5: test shoots (talking head, travel, multi-camera, screen recording), quality and cost metrics, recorded runs replayed as regression tests; decide whether `llmve auto` stays as a fallback.

## Open

- Whether `llmve auto` is removed once agent mode beats it (default: keep as fallback).
- Voice-over decisions listed in 11.2.
- Order of 11.1 items (suggested: transitions, flip/rotate, VAD, inventory/balance, copyrighted audio, place titles in auto).
