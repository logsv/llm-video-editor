---
name: royalty-free-media
description: Find and add royalty-free music, sound effects, voice-over and stock or funny cutaway clips to an edit - only where they help - with licences and credits tracked. Searches Openverse (CC music/SFX), Pixabay, Pexels and Wikimedia Commons; registers tracks from YouTube's Audio Library / Creator Music. Music ducks under the clips' voice and any voice-over. Use when the user wants background music, sound effects, B-roll, a funny clip or a voice-over.
---

# royalty-free-media

Everything lands in `<ws>/media/` with its licence in `media/credits.json`; the
`youtube-publish` skill turns that into the description's credits block.

## Rules (do not bend them)
- Only licensed sources: the providers below, or files the user has rights to.
  Never pull clips from YouTube, TikTok, Instagram, films, TV or memes, and never
  scrape them (Apify or otherwise): they are copyrighted and get Content ID claims.
- Licences must allow commercial use and editing: no NC, no ND (the search
  already filters them).
- Add media only where it helps. Fewer, better cutaways beat constant ones; "none"
  is a fine answer. Never cover the opening hook or a step the viewer must see.
- Search words describe generic footage: no real people, brands or characters.

## 1. Music, sound effects
```bash
uv run llmve media search <ws> "calm lofi beat" --kind music --length medium
uv run llmve media search <ws> "whoosh" --kind sfx
uv run llmve media get <ws> openverse:<id>
uv run llmve music add <ws> <ws>/media/<file>.mp3 --at c001            # bed, ducked
uv run llmve music add <ws> <ws>/media/<sfx>.mp3 --at c004 --offset 1.2 --role sfx
```
- Music beds: `--volume -10` (default; relative to the speech, files are levelled first) under talking, `--loop` for short loops,
  `--duration` to stop early. Ducking follows the speech level automatically.
- **YouTube Audio Library / Creator Music** (no API): ask the user to open their
  Studio > Audio Library (or Creator Music filtered to "No cost"), download the
  track and copy its licence/credit text, then
  ```bash
  uv run llmve media add <ws> ~/Downloads/track.mp3 --title "Track" \
      --license "YouTube Audio Library" --provider youtube-audio-library \
      --creator "Artist" --attribution "<exact credit text from YouTube>"
  ```

## 2. Voice-over
Record or receive a narration file, then
`uv run llmve music add <ws> narration.wav --at c001 --role voiceover`.
Voice-over plays at full level, is never ducked, and music ducks under it as
well as under the clips' own voice (`"mix": {"duck_depth_db": 12}` in the EDL
sets how far).

## 3. Cutaways: B-roll and funny clips, only when needed
```bash
uv run llmve media suggest <ws> --brief "playful"     # may suggest none
uv run llmve media fetch <ws>                         # licensed clip per slot -> overlays
```
`suggest` holds the planner to a budget: one cutaway per 45 s at most (`--every`),
humour at most one per 2 minutes (`--no-humor` turns it off), 20 s apart, never in
the first 5 s. Read `media/cutaways.json` (slots and what was dropped) and show
the user the list before fetching if they care about tone.

## 4. Keys and checks
- `PIXABAY_API_KEY`, `PEXELS_API_KEY` (free accounts) enable those providers;
  Openverse and Wikimedia need none. Missing keys are reported, not fatal.
- `uv run llmve media credits <ws>` lists what needs credit. Then
  `uv run llmve cut-audit <ws>`, review, render as usual.
