---
name: voiceover
description: Add AI narration to an edit only where it helps - a place or fact the footage can't say, a joke, context for a silent stretch - spoken by a local voice (Kokoro, nothing uploaded), levelled to the speech, with music ducking under it and the scene's own sound dipping slightly. Use when the user asks for a voice-over or narration, or when an edit (often silent travel footage) needs a few lines of context. Not every video needs one.
---

# voiceover

## 0. Decide whether it helps (most important)
Narration is optional. Add it only when a line gives the viewer something the
picture and the user's own voice don't:
- **information**: where we are, how far, what this place is ("Khajuraho's temples
  are a thousand years old");
- **a turn in the story**: "Then the road ran out.";
- **humour** that fits the vibe: a dry aside over a cow blocking the highway;
- **details** a viewer would ask about: the bike, the cost, the time of day.

Do not narrate what is plainly visible, do not talk over the user's own speech,
and leave long stretches with music and natural sound only. A 7-minute trip video
usually needs 4-10 short lines, sometimes none. If the user's brief says no
narration, add none.

## 1. Write the lines
- Short: 4-14 words, one idea per line, spoken English (or Hindi/Hinglish if the
  brief asks). Write as it will be captioned ("380 km"); numbers and units are
  spoken out automatically.
- Match the tone: calm and warm for travel, punchy for a hook, deadpan for comedy.
- Names: add respellings once with `pronounce` (`{"Khajuraho": "Kha-joo-raa-ho"}`,
  scope `all` keeps them for every project).

## 2. Place them
- Find room with `speech_map` / `transcript`: gaps without the user's speech,
  riding shots, establishing shots, the moment before a reveal.
- Start a line 0.3-0.8 s into a shot, not on the cut; leave at least 2 s between lines.

## 3. Make them
```
voiceover {"workspace": ..., "reason": "...", "voice": "am_michael",
           "lines": [{"text": "...", "anchor_clip_id": "c014", "offset": 0.6}],
           "audition": true}     # first: lengths only, the edit is unchanged
```
- `voices` lists voices (`language: "Hindi"`); good defaults: `af_heart`,
  `am_michael`, `bf_emma`, `bm_george`, Hindi `hf_alpha` / `hm_omega`. Keep one
  voice per video.
- `speed` 0.9-1.1; slower for calm travel, never rushed.
- Then call it without `audition`. Read `fit`: fix every issue (over the user's
  speech, overlapping lines, past the end) by moving the line, shortening the text
  or lengthening the shot.

## 4. Mix (automatic, adjustable)
- Lines are levelled to the measured speech (`volume_db` 0 = as loud as the user's
  voice); the music ducks under them like under speech (`mix.duck_depth_db`), and
  the scene's own sound dips by `mix.clip_duck_db` (default 6 dB) so the narration
  is clean without muting the place.
- Too loud or quiet: the line's `volume_db` in steps of 2 dB. Music fighting it:
  lower the music bed or raise `duck_depth_db`.

## 5. Captions and publish
- Captions and the `.srt` use the line text. Publish adds "Narration: AI-generated
  voice" to the description and a reminder about YouTube's altered/synthetic
  content setting. Never imitate a real person's voice.
