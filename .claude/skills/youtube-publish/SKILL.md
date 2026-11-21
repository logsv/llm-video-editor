---
name: youtube-publish
description: Prepare a rendered video for upload - SEO titles, description with chapters, credits and hashtags, tags, category, pinned comment, recording date/device/location, and three honest click-worthy thumbnails - written as <video>.upload.md next to the video. Use after render, or when the user asks for a YouTube title, description, tags or thumbnails.
---

# youtube-publish

```bash
uv run llmve publish <ws> --brief "channel about budget audio gear, beginners"
```
Writes next to the rendered video: `<video>.upload.md` (copy-paste ready),
`<video>.upload.json`, `<video>.thumb1.jpg` .. `thumb3.jpg` (1280x720, or 1080x1920
for vertical cuts; HyperFrames typography when installed, ffmpeg otherwise).

## What to do with it
1. Read `<video>.upload.md`. Fix every item under "Check before uploading":
   missing credits (`llmve media add`), chapter rules, long titles.
2. Look at the three thumbnails. They must be true to the video: curiosity,
   contrast or payoff the video delivers; no fake reactions, no misleading claims,
   no brands or faces the user did not provide. Re-run publish for new ideas.
3. Location: the GPS position is read from the footage. Only with `--geocode`
   are the coordinates sent to OpenStreetMap for a place name, and only with
   `--use-location` may the copy mention the place. Ask the user before either:
   a public location reveals where they filmed.
4. Give the user the recommended title, the description block, and the three
   thumbnail paths; mention YouTube's Test & Compare for the thumbnails.

## Copy rules the stage enforces
Titles <= 100 characters (60 shown in search; longer ones are flagged), no `<` or
`>`, description <= 5000 characters (chapters and credits are never trimmed),
tags <= 500 characters, 3 hashtags. The planner sees text only (transcript lines
of the edit, chapters, brief); thumbnail moments are segment ids, resolved locally.
