---
name: motion-graphics
description: Add HyperFrames motion graphics to a cut - lower thirds, chapter title cards, keyword callouts, or a custom animated HTML/GSAP composition - rendered to transparent video and composited at render time. Use when the user asks for titles, name tags, text pop-ups, animated inserts or chapter cards, after planning and before cut-audit.
---

# motion-graphics

Graphics live in `EDL.graphics`, anchored to a clip like B-roll overlays.
HyperFrames (pinned CLI, Node 22+) renders each one to a transparent VP9 WebM;
review shows every graphic on its own tile and render composites them under
the burned-in captions.

## 0. Toolchain (once)
```bash
uv run llmve graphics doctor --install     # Node 22+, npm, pinned HyperFrames + GSAP
```
Telemetry and update checks are off. If no Chrome can be downloaded, point
`HYPERFRAMES_BROWSER_PATH` at an existing Chrome/Chromium.

## 1. Built-in templates
```bash
uv run llmve graphics templates            # names and variables
uv run llmve graphics add <ws> --at c003 -t lower-third --offset 0.5 --duration 4 \
    --var title="Jane Doe" --var subtitle="Founder, Acme"
uv run llmve graphics add <ws> --at c007 -t callout --duration 2 --var text="-14 LUFS"
uv run llmve graphics chapters <ws>        # one title card per clip with a chapter
```
- `lower-third`: name/role, `position` bottom-left | top-left | bottom-center
  (in 9:16 it sits above the caption band automatically).
- `title-card`: full-frame dimmed card with `title`, optional `kicker`, `dim`.
- `callout`: one short phrase that pops in, `position` top | center.
- All take `accent` (hex colour). Keep text short: 2-6 words per line.
Clip ids come from `edl_raw.json`. Time a graphic to the words: offset = word
start in `transcript.json` minus the clip's `start`.

## 2. Custom compositions
For anything the templates don't cover (an app UI recreation, a stat counter, a
diagram), build a HyperFrames project folder and add it with
`uv run llmve graphics add <ws> --at c004 -c <folder> --duration 3 --box 0.55,0.1,0.4,0.5`.
Rules for `index.html` (lint enforces most):
- root `<div data-composition-id="main" data-start="0" data-duration="D"
  data-width="W" data-height="H">`; size it to the box it will fill
- `<script src="assets/gsap.min.js">` (copied in at render; no CDN needed)
- one paused timeline, `gsap.timeline({ paused: true })`, registered as
  `window.__timelines["main"]`; initial states with `gsap.set` outside it
- transparent `html, body` background; only what should show is painted
- text from `data-composition-variables` + `window.__hyperframes.getVariables()`
  when it should be reusable
Never use real people's names, logos or likenesses the user didn't provide.

## 3. Then
`uv run llmve cut-audit <ws>` (drops graphics with a missing anchor or template,
keeps them inside the cut, follows jump-cut tightening), `uv run llmve review <ws>`,
check the `gNNN over cNNN` tiles: nothing covers the face box or the
captions, text is readable at tile size; then approve as in `review-gate`. `uv run llmve graphics render <ws>`
pre-renders them if you want to check the files first.
