# Matching rules

- Query: the planner's B-roll `query`, or the transcript text of the segment.
- Document: description plus tags (tags count twice). B-roll clips from the
  same shoot use the words in their filenames.
- Score: BM25 (k1 = 1.4, b = 0.75), then
  - +0.3 when the clip's orientation matches the output (portrait for reels and tiktok),
  - -0.5 when the clip is shorter than the slot,
  - -1.0 when the clip is already used in this edit.
- The top candidate is accepted only if it scores 0.5 or more; otherwise the
  slot stays empty.
- Optional LLM re-rank: one text-only call covering every slot and its top 5
  candidates. An id that isn't among that slot's candidates is ignored.
- The source window starts at the clip's longest scene, which is usually the
  steadiest shot.
