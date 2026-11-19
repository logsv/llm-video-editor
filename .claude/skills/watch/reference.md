# Contact-sheet token budget

Claude bills an image at about `width * height / 750` tokens; images over
1568 px on the long edge are downscaled first. Tiling saves tokens only
because each tile is small and there are fewer images.

| Sheet | Tile | Sheet size | ~Tokens | Frames | ~Tokens/frame |
|---|---|---|---|---|---|
| overview, 16:9 source | 384x216 | 1556x884 | 1,835 | 16 | 115 |
| overview, 9:16 source | 216x384 | 884x1556 | 1,835 | 16 | 115 |

For a 15-minute source:
- overview at 0.5 FPS: 450 frames, 29 sheets, about 53k tokens
- dense 2 FPS for three 30 s windows: 180 frames, 12 sheets, about 22k tokens
- flat 2 FPS everywhere (don't): 1,800 frames, 113 sheets, about 207k tokens

`sheets/index.json` records `est_tokens` per sheet, and the `watch` command
prints the total.
