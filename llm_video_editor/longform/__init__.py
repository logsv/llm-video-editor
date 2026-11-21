"""
Long-form (YouTube 16:9) editing toolkit designed to be driven by Claude Code.

The split of work:
  * ``ingest``  - probe the source, transcribe with word timestamps, map silences,
                  write a compact ``transcript.md`` and a contact sheet.
  * Claude      - reads the whole transcript and writes ``edl.json`` (what to keep,
                  chapter names). No local LLM, no truncated context.
  * ``render``  - frame-accurate cut in a single ffmpeg pass, optional jump-cut
                  tightening, loudness normalisation, chapters, captions, QA sheet.

Run ``python -m llm_video_editor.longform --help``.
"""
