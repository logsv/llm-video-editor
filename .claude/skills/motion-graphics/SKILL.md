---
name: motion-graphics
description: Add HyperFrames motion graphics to a cut - lower thirds, chapter title cards, keyword callouts, place titles (where we are), labels pointing at things in the shot (animal, temple, road), or a custom animated HTML/GSAP composition - rendered to transparent video and composited at render time. Use when the user asks for titles, name tags, text pop-ups, animated inserts or chapter cards, after planning and before cut-audit.
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
- `location`: place title with a map pin, `title` (place) + `subtitle` (region,
  road, distance), `position` top-left | bottom-left | top-right | center.
- `label`: points at something in the shot, `text`, optional `detail`, `x`/`y` (0-1
  of the frame). It does not track movement: use it on a steady shot, 2-3 s.
- All take `accent` (hex colour). Keep text short: 2-6 words per line.
Clip ids come from `edl_raw.json`. Time a graphic to the words: offset = word
start in `transcript.json` minus the clip's `start`.

## 1b. Travel and vlog footage: say where we are and what we see
Viewers of a trip video want to know where they are. Decide with the user's brief
(nothing here is mandatory):
- **Places**: `places` (agent tool) gives the route from the cameras' GPS and clock.
  With the user's permission (`location_lookup`) it names each stop (town, nearest
  temple/fort, road, state) and, with `nearby_km`, named temples, forts, forests,
  national parks, rivers and highways (e.g. "NH39"). Without GPS or permission, ask
  the user or read signs in frames and the transcript. Never guess a place name.
- **Titles**: a `location` title when a new stop starts (2.5-4 s, on a steady shot),
  `title-card` for each leg or day (chapters), `callout` for a fact ("380 km",
  "Day 2", "11th century"), `label` for something worth pointing at.
- **Objects**: `objects` lists animals, vehicles and people in a clip (local detector)
  with their frame position for a `label`. Temples, forests, rivers and roads are not
  detector classes: look at `frames` (stills privacy) and name them from `places`.
- Keep it sparse: one text element on screen at a time, none over faces or captions,
  and about one place title per new location, not per clip. `review` reports the
  longest stretch without titles and chapters that have none.

## 1c. Readable, safe and matched to the shot
`review` measures every graphic on its shot (`review.json` -> `graphic_checks`; the
agent's `review` tool returns them): contrast (3:1 minimum), busy background, size,
title-safe area (5% in 16:9; in 9:16 also the right 15% and bottom 20%), face and
caption overlap, the shot's palette and suggested colours. Act on each issue:
- low contrast or busy background: a template with a card (`lower-third`), the
  suggested `text`/`contrast` colour, another `position`, or a calmer moment (offset);
- colours: `"accent": "auto"` takes a colour from the shot (or a contrasting one when
  the match would be faint); or use `suggested.match` for a calm look,
  `suggested.contrast` to stand out. Keep one accent per video unless the brief asks;
- size: keep titles short so the template can stay large; never shrink text to fit;
- overlap: move it (`position`, `box`) or time it to a shot without a face.

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
captions, text is readable at tile size. `uv run llmve graphics render <ws>`
pre-renders them if you want to check the files first.
