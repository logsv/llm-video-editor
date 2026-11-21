# /youtube-edit reference

## Project layout (`projects/<slug>/`)
| File | Written by | What |
|---|---|---|
| `source.json` | ingest | path, duration, size, fps, VFR flag |
| `words.json` | ingest | `[{"w","s","e","p"}]` word timestamps in source seconds |
| `silences.json` | ingest | `[[start, end]]` from silencedetect (-35 dB, 0.4 s) |
| `transcript.md` | ingest | timestamped sentences, `(retake?)`, `(unclear N)`, `-- pause --` |
| `work/sheet.jpg` + `.txt` | ingest | 24 frames across the recording and their times |
| `edl.json` | **Claude** | the edit |
| `renders/<slug>-vN.mp4` | render | final file (chapters embedded, +faststart) |
| `renders/<slug>-vN.srt` / `.chapters.txt` | render | captions and description chapters for the edited timeline |
| `renders/<slug>-vN.cuts.jpg` / `.txt` | render | before/after frame for each cut (up to 30, sampled) |
| `renders/<slug>-vN.report.json` | render | durations, A/V drift, warnings |

## edl.json schema
```json
{
  "title": "I automated my editing with Claude",
  "tighten": {"enabled": true, "max_gap": 0.6, "pad_before": 0.08, "pad_after": 0.18,
              "min_island": 0.25, "drop_fillers": true},
  "punch_in": {"mode": "alternate", "zoom": 1.12},
  "caption_fixes": {"cloud": "Claude", "langraph": "LangGraph"},
  "segments": [
    {"in": 1834.20, "out": 1851.70, "chapter": "The result", "tighten": false},
    {"in": 12.40, "out": 95.20, "chapter": "Why I built this"},
    {"in": 130.00, "out": 410.50},
    {"in": 410.50, "out": 900.00, "chapter": "Setup in 5 minutes", "zoom": 1.0},
    {"in": 905.00, "out": 1210.00, "chapter": "Live demo", "tighten": false}
  ]
}
```
- `in`/`out`: source seconds (copy from `transcript.md`). Segments play in the listed order; going
  backwards (cold open) is fine and just adds a render pass.
- Per segment: `chapter`, `tighten` (overrides the global), `zoom` (fixed punch-in, 1.0-1.5; overrides
  alternate), `cx`/`cy` (0-1 crop centre for zoom and `--vertical`, default 0.5), `note` (ignored, for you).
- `tighten` turns each segment into speech islands: pauses longer than `max_gap` are removed, keeping
  `pad_before`/`pad_after` of air. Segments with no words (demo, b-roll) stay whole.
- `drop_fillers` removes standalone um/uh when there is a pause around them; fillers are always left
  out of captions.
- Optional `"source": "/path"` overrides `source.json`.

## Pacing guide
| Style | max_gap | punch_in | Feels like |
|---|---|---|---|
| tight (default) | 0.45-0.6 | alternate 1.12 | modern tutorial / commentary |
| natural | 0.9-1.2 | off | interview, podcast, storytelling |
| raw | tighten off | off | lectures, live streams; only remove big blocks |

Typical keep ratio: tutorials 50-70 %, podcasts 80-95 %, rambling first drafts 30-50 %.

## Traps
- **Whisper timestamps** can be 0.1-0.3 s off at the start of a phrase and it sometimes invents a word
  right after silence. If a cut sounds clipped, extend that `in` by 0.1 s; check `silences.json`.
- **Retake markers are hints**, based on repeated opening words; read the lines, do not trust blindly.
- **Variable frame rate** (phone, OBS with VFR, screen recorders): ingest warns. Render re-encodes to
  constant frame rate, but if the report shows drift > 100 ms, transcode the source first:
  `ffmpeg -i in.mp4 -fps_mode cfr -r 30 -c:v libx264 -crf 16 -c:a copy in_cfr.mp4` and re-ingest.
- **Long sources**: render seeks to each forward run, so a cold open from minute 50 costs only its
  own short pass. Preview first; a 1 h 4K final on CPU x264 can take a while: suggest `--encoder nvenc`
  / `videotoolbox` when available, `--height 1080` to downscale.
- **Chapters** silently disappear on YouTube if any rule fails: render prints a WARNING; fix them.
- **Multiple files** (A-cam + screen recording, several takes): this pipeline edits one source. Join
  sequential takes first (`ffmpeg -f concat`), or edit each separately; multicam sync is out of scope.
