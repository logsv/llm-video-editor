"""Motion graphics stage helpers: add graphics to the cut and render them for the encode.

Graphics live in ``EDL.graphics`` (see schemas/edl.py). ``llmve graphics add`` and
``llmve graphics chapters`` write them into ``edl_raw.json``; cut-audit validates and
re-anchors them; render turns each into a cached transparent WebM with HyperFrames
(``llm_video_editor.hyperframes``) and composites it in the final pass.
"""

from __future__ import annotations

from pathlib import Path

from ..ffmpeg.graph import GraphicSpec, quantize
from ..schemas import EDL, Graphic
from ..workspace import Workspace


def graphics_dir(ws: Workspace) -> Path:
    return ws.root / "graphics"


def next_id(edl: EDL) -> str:
    used = {g.graphic_id for g in edl.graphics}
    n = len(edl.graphics) + 1
    while f"g{n:03d}" in used:
        n += 1
    return f"g{n:03d}"


def add_graphic(ws: Workspace, *, anchor: str, duration: float, template: str | None = None,
                composition: str | None = None, offset: float = 0.0,
                variables: dict | None = None,
                box: tuple[float, float, float, float] | None = None) -> Graphic:
    """Append a graphic to ``edl_raw.json`` (run cut-audit afterwards)."""
    edl = Workspace.read(ws.edl_raw, EDL)
    if anchor not in {c.clip_id for c in edl.clips}:
        raise ValueError(f"no clip {anchor!r} in edl_raw.json "
                         f"(have {', '.join(c.clip_id for c in edl.clips)})")
    g = Graphic(graphic_id=next_id(edl), template=template,
                composition=str(Path(composition).expanduser().resolve()) if composition else None,
                variables=variables or {}, anchor_clip_id=anchor, offset=offset,
                duration=duration, box=box)
    Workspace.write(ws.edl_raw, edl.model_copy(update={"graphics": [*edl.graphics, g]}))
    return g


def chapter_cards(ws: Workspace, *, template: str = "title-card", duration: float = 2.5,
                  variables: dict | None = None, numbered: bool = True) -> list[Graphic]:
    """One title card per ``EDLClip.chapter`` in edl_raw.json (replaces earlier cards)."""
    edl = Workspace.read(ws.edl_raw, EDL)
    kept = [g for g in edl.graphics if not g.graphic_id.startswith("ch")]
    cards = []
    chapters = [c for c in edl.clips if c.chapter]
    for n, clip in enumerate(chapters, 1):
        vars_ = {"title": clip.chapter, **(variables or {})}
        if numbered and template == "title-card":
            vars_.setdefault("kicker", f"Chapter {n}")
        cards.append(Graphic(graphic_id=f"ch{n:02d}", template=template, variables=vars_,
                             anchor_clip_id=clip.clip_id, duration=duration))
    Workspace.write(ws.edl_raw, edl.model_copy(update={"graphics": [*kept, *cards]}))
    return cards


def _auto_accent(edl: EDL, g) -> str:
    """Accent for ``accent: "auto"``: matches the anchor shot, or contrasts with it."""
    from ..ffmpeg.runner import FFmpegError
    from .text_safety import auto_accent, frame_rgb

    clip = next(c for c in edl.clips if c.clip_id == g.anchor_clip_id)
    t = clip.start + min(max(0.0, g.offset) + 0.5, max(0.0, clip.duration - 0.1))
    try:
        return auto_accent(frame_rgb(edl, clip.source_file, t, clip.crop_center, clip.zoom))
    except (ValueError, FFmpegError):
        return "#FFC93C"  # the templates' default yellow


def graphic_specs(ws: Workspace, edl: EDL, starts: list[float], *, quality: str = "standard"
                  ) -> list[GraphicSpec]:
    """Render (or reuse) every graphic and place it on the output timeline."""
    if not edl.graphics:
        return []
    from ..hyperframes import render_graphic

    by_id = {c.clip_id: s for c, s in zip(edl.clips, starts, strict=True)}
    out_dir = graphics_dir(ws)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = []
    for g in edl.graphics:
        if g.anchor_clip_id not in by_id:
            continue
        dur = quantize(g.duration, edl.fps)
        if g.box is None:
            w, h = edl.width, edl.height
        else:
            w = max(2, round(g.box[2] * edl.width) // 2 * 2)
            h = max(2, round(g.box[3] * edl.height) // 2 * 2)
        variables = dict(g.variables)
        if variables.get("accent") == "auto":  # a colour taken from the shot it sits on
            variables["accent"] = _auto_accent(edl, g)
        path = render_graphic(out_dir, template=g.template, composition=g.composition,
                              variables=variables, width=w, height=h, duration=dur,
                              fps=edl.fps, quality=quality)
        specs.append(GraphicSpec(str(path), quantize(by_id[g.anchor_clip_id] + g.offset, edl.fps),
                                 dur, g.box))
    return specs
