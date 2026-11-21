# Agent mode: plan

Status: plan only, nothing implemented. Branch `claude/agent-mode-plan`.

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

## 2. The toolbox (instruments, no decisions)

Looking (read-only):
- project overview: clips, durations, speech/no speech, devices, GPS, dates
- transcript with word times; speech map (words and silences); re-transcribe a range with a bigger model and a name list
- frames and contact sheets for any clip and time window (images, logged)
- measurements per scene: exposure, white balance, saturation, noise, shake, motion, faces, scene-change scores
- audio: loudness, peaks, noise floor, music/speech detection, beat times
- current edit, version history, decision log

Editing (validated, versioned, undoable; every call carries a reason):
- clips: add, remove, trim, split, reorder, speed (slow motion, ramps), stabilise
- framing: crop, zoom/punch-in, vertical reframing
- colour: correction per clip (from measurements), match clips, looks (parameters, curves, LUT)
- transitions per cut: hard cut, crossfade, dip to black, whip/slide, J-cut, L-cut
- audio: noise reduction, EQ, levels; music, sound effects, voice-over; ducking depth
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
4. Licensed media only (commercial use and editing allowed); every third-party asset has a credit.
5. Captions say what was said.
6. Budgets: tokens, time, iterations per run.

Everything the current code decides by rule (pause thresholds, cutaway budgets, music
volume, caption style, crop centre, chapter rules, cut-audit auto-fixes) moves to tool
parameters or handbook advice.

## 5. How an edit runs

1. Setup: machine record, user profile, handbook, gotchas.
2. Understand: transcript, speech map, scene measurements, contact sheets; the agent's own notes on story and quality.
3. Plan: story, structure, length, look, sound, inserts; written to the log, auto-approved.
4. Edit: tool calls, workshop scripts, sub-agent briefs, each producing a new version.
5. Self-review: draft render, cut strips, continuity, loudness, safe areas; fix and repeat until satisfied or out of budget.
6. Deliver: final render, review copy, publish package, decision log (`decisions.md`), gotchas update.
7. Notes from the user restart at step 4.

## 6. Model, cost, caching

- Lead editor and sub-agents: `claude-opus-5-5`, adaptive thinking, effort `high`.
- Stills only on request (a 16-frame sheet is about 1.8k input tokens); text-only mode uses none.
- Stable tool list and handbook first in the prompt so they cache; notes arrive as mid-conversation messages.
- Task budgets cap a run; long sessions use compaction.
- Rough estimate: $1-5 of model usage per 10-minute video, to be measured in P5.

## 7. Phases

| Phase | Delivers | Done when |
|---|---|---|
| P0 Foundation | Stages split into small tools with parameters instead of rules; audit reports only; versioned edits, undo, decision log; `llmve mcp` tool server; `llmve setup` (machine record, user profile); privacy setting | Claude Code reproduces today's edit quality by calling tools only and choosing every parameter |
| P1 Perception | Speech map, re-transcribe, frame/sheet viewer, scene measurements, audio analysis and beats | The agent can describe every clip of the trip footage and justify its selects |
| P2 Craft | Colour, transitions (incl. J/L-cuts), speed, stabilise, audio cleanup and mix; workshop with registered outputs | Each tool's render matches the request (frame and audio checks) |
| P3 Editor agent | `llmve agent <clips> -p "<brief>"`; handbook, style profile, gotchas memory; plan -> edit -> self-review loop; specialist sub-agents with briefs | One command produces a finished edit with a readable decision log and no rules in the decision path |
| P4 Visual story | Story building from scenes as well as speech (travel, B-roll-heavy footage) | The Bhopal -> Khajuraho footage becomes a coherent 10-minute video |
| P5 Evaluation | Test shoots (talking head, travel, multi-cam, screen recording), quality and cost metrics, recorded runs replayed as regression tests | The agent beats the rule-based pipeline on the metrics, with measured cost per video |

## 8. First test: Bhopal -> Khajuraho

- Run on the user's Mac (the footage never leaves it): `llmve agent "$HOME/Documents/Video/Bhopal to Khajuraho" -p "10-minute solo bike trip vlog" -t youtube`.
- Expect mostly riding and scenery with some talking, so it exercises P1 (scene understanding) and P4 (visual story) most.
- Check: a 10-minute cut with a route story, consistent colour across cameras, transitions that fit location changes, music under riding, the user's voice clear where they talk, chapters by place, credits complete, decision log readable.
- The user's notes from this run seed the style profile and the first gotchas.

## Open

- Whether `llmve auto` is removed once agent mode beats it (default: keep as fallback).
