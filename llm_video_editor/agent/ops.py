"""Atomic edit operations: the agent's hands on the edit.

Each operation is a pure function ``(edl, manifest, **params) -> EDL``. They only
execute and check correctness (clip exists, times inside the source, ids unique);
every editorial choice comes from the parameters the agent passes. Errors are
``EditError`` with a message that says what to do instead.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ..schemas import (
    EDL,
    CaptionSettings,
    ClipColor,
    ClipsManifest,
    EDLClip,
    Grade,
    Graphic,
    MixSettings,
    MusicTrack,
    Overlay,
    Pacing,
)
from .edits import EditError


# -- helpers ------------------------------------------------------------------------------
def _find(edl: EDL, clip_id: str) -> int:
    for i, c in enumerate(edl.clips):
        if c.clip_id == clip_id:
            return i
    raise EditError(f"no clip {clip_id!r}; clips are {', '.join(c.clip_id for c in edl.clips)}")


def _source(manifest: ClipsManifest, source_clip_id: str):
    info = manifest.get(source_clip_id)
    if info is None:
        known = ", ".join(c.clip_id for c in manifest.clips[:20])
        raise EditError(f"no source clip {source_clip_id!r}; sources are {known}")
    return info


def _check_range(info, start: float, end: float) -> None:
    if not (0 <= start < end <= info.duration + 1e-6):
        raise EditError(f"{start:.3f}-{end:.3f}s is not inside source {info.clip_id} "
                        f"(0-{info.duration:.3f}s); start must be before end")


def _new_id(existing: set[str], prefix: str) -> str:
    n = 1
    while f"{prefix}{n:03d}" in existing:
        n += 1
    return f"{prefix}{n:03d}"


def _merge(model, obj, fields: dict[str, Any]):
    """Apply ``fields`` to a pydantic object; unknown fields and bad values are errors."""
    unknown = set(fields) - set(type(obj).model_fields)
    if unknown:
        raise EditError(f"unknown field(s) {sorted(unknown)} for {type(obj).__name__}; "
                        f"allowed: {sorted(type(obj).model_fields)}")
    try:
        return model.model_validate({**obj.model_dump(), **fields})
    except ValidationError as e:
        raise EditError(f"invalid value: {e.errors()[0]['msg']} "
                        f"({'.'.join(map(str, e.errors()[0]['loc']))})") from e


def _replace_clip(edl: EDL, i: int, clip: EDLClip) -> EDL:
    clips = list(edl.clips)
    clips[i] = clip
    return edl.model_copy(update={"clips": clips})


# -- clips ------------------------------------------------------------------------------------
def add_clip(edl: EDL, manifest: ClipsManifest, *, source_clip_id: str, start: float,
             end: float, position: int | None = None, after: str | None = None,
             role: str = "a_roll", chapter: str | None = None, zoom: float = 1.0,
             crop_center: tuple[float, float] | None = None, notes: str = "") -> EDL:
    info = _source(manifest, source_clip_id)
    _check_range(info, start, end)
    cid = _new_id({c.clip_id for c in edl.clips}, "c")
    clip = _merge(EDLClip, EDLClip(clip_id=cid, source_clip_id=info.clip_id,
                                   source_file=info.source_file, start=start, end=end),
                  {"role": role, "chapter": chapter, "zoom": zoom, "crop_center": crop_center,
                   "notes": notes})
    clips = list(edl.clips)
    if after is not None:
        idx = _find(edl, after) + 1
    elif position is not None:
        idx = max(0, min(len(clips), position))
    else:
        idx = len(clips)
    clips.insert(idx, clip)
    return edl.model_copy(update={"clips": clips})


def remove_clip(edl: EDL, manifest: ClipsManifest, *, clip_id: str) -> EDL:
    i = _find(edl, clip_id)
    clips = [c for j, c in enumerate(edl.clips) if j != i]
    anchored = [getattr(x, id_field) for kind, (_, id_field, _) in _ITEMS.items()
                for x in getattr(edl, kind) if x.anchor_clip_id == clip_id]
    if anchored:
        raise EditError(f"{clip_id} anchors {', '.join(anchored)}; "
                        "re-anchor (update_item) or remove them first")
    return edl.model_copy(update={"clips": clips})


def trim_clip(edl: EDL, manifest: ClipsManifest, *, clip_id: str, start: float | None = None,
              end: float | None = None) -> EDL:
    i = _find(edl, clip_id)
    c = edl.clips[i]
    s = c.start if start is None else start
    e = c.end if end is None else end
    _check_range(_source(manifest, c.source_clip_id), s, e)
    return _replace_clip(edl, i, c.model_copy(update={"start": round(s, 3), "end": round(e, 3)}))


def split_clip(edl: EDL, manifest: ClipsManifest, *, clip_id: str, at: float) -> EDL:
    """Split at source time ``at``; the second part gets a new id and no chapter."""
    i = _find(edl, clip_id)
    c = edl.clips[i]
    if not (c.start < at < c.end):
        raise EditError(f"split point {at:.3f}s must be inside {clip_id} "
                        f"({c.start:.3f}-{c.end:.3f}s, source time)")
    second = c.model_copy(update={"clip_id": _new_id({x.clip_id for x in edl.clips}, "c"),
                                  "start": round(at, 3), "chapter": None})
    clips = list(edl.clips)
    clips[i] = c.model_copy(update={"end": round(at, 3)})
    clips.insert(i + 1, second)
    return edl.model_copy(update={"clips": clips})


def move_clip(edl: EDL, manifest: ClipsManifest, *, clip_id: str, position: int | None = None,
              after: str | None = None) -> EDL:
    i = _find(edl, clip_id)
    clip = edl.clips[i]
    clips = [c for j, c in enumerate(edl.clips) if j != i]
    if after is not None:
        if after == clip_id:
            raise EditError("a clip cannot be placed after itself")
        _find(edl, after)  # unknown id: helpful error
        idx = next(j for j, c in enumerate(clips) if c.clip_id == after) + 1
    elif position is not None:
        idx = max(0, min(len(clips), position))
    else:
        raise EditError("give position (0 = first) or after (a clip id)")
    clips.insert(idx, clip)
    return edl.model_copy(update={"clips": clips})


def set_clip(edl: EDL, manifest: ClipsManifest, *, clip_id: str, **fields: Any) -> EDL:
    """zoom, crop_center, chapter ("" clears), role, priority, notes, caption_position."""
    i = _find(edl, clip_id)
    blocked = {"clip_id", "source_clip_id", "source_file", "start", "end"} & set(fields)
    if blocked:
        raise EditError(f"use trim_clip / add_clip for {sorted(blocked)}")
    if fields.get("chapter") == "":
        fields["chapter"] = None
    return _replace_clip(edl, i, _merge(EDLClip, edl.clips[i], fields))


# -- whole-edit settings --------------------------------------------------------------------
def set_pacing(edl: EDL, manifest: ClipsManifest, **fields: Any) -> EDL:
    return edl.model_copy(update={"pacing": _merge(Pacing, edl.pacing, fields)})


def set_captions(edl: EDL, manifest: ClipsManifest, **fields: Any) -> EDL:
    return edl.model_copy(update={"captions": _merge(CaptionSettings, edl.captions, fields)})


def set_mix(edl: EDL, manifest: ClipsManifest, **fields: Any) -> EDL:
    return edl.model_copy(update={"mix": _merge(MixSettings, edl.mix, fields)})


def set_grade(edl: EDL, manifest: ClipsManifest, **fields: Any) -> EDL:
    """One look for the whole video: look, strength, lut."""
    return edl.model_copy(update={"grade": _merge(Grade, edl.grade, fields)})


def set_color(edl: EDL, manifest: ClipsManifest, *, clip_id: str, **fields: Any) -> EDL:
    """Per-clip correction (merged into the clip's current one); no fields resets it."""
    i = _find(edl, clip_id)
    clip = edl.clips[i]
    color = None if not fields else _merge(ClipColor, clip.color or ClipColor(), fields)
    clips = list(edl.clips)
    clips[i] = clip.model_copy(update={"color": color})
    return edl.model_copy(update={"clips": clips})


def set_output(edl: EDL, manifest: ClipsManifest, *, target_duration: float | None = None,
               clear_target: bool = False, fit_mode: str | None = None) -> EDL:
    update: dict[str, Any] = {}
    if clear_target:
        update["target_duration"] = None
    elif target_duration is not None:
        if target_duration <= 0:
            raise EditError("target_duration must be positive (seconds)")
        update["target_duration"] = target_duration
    if fit_mode is not None:
        if fit_mode not in ("cover", "contain"):
            raise EditError("fit_mode is cover or contain")
        update["fit_mode"] = fit_mode
    return edl.model_copy(update=update)


# -- anchored items: overlays, graphics, music ----------------------------------------------
_SINGULAR = {"overlays": "overlay", "graphics": "graphic", "music": "music track"}
_ITEMS = {
    "overlays": (Overlay, "overlay_id", "x"),
    "graphics": (Graphic, "graphic_id", "g"),
    "music": (MusicTrack, "track_id", "a"),
}


def add_item(edl: EDL, manifest: ClipsManifest, *, kind: str, **fields: Any) -> EDL:
    model, id_field, prefix = _ITEMS[kind]
    items = list(getattr(edl, kind))
    _find(edl, fields.get("anchor_clip_id", ""))
    fields.setdefault(id_field, _new_id({getattr(x, id_field) for x in items}, prefix))
    if any(getattr(x, id_field) == fields[id_field] for x in items):
        raise EditError(f"{fields[id_field]} already exists in {kind}")
    if kind == "overlays":
        info = _source(manifest, fields.get("source_clip_id", ""))
        fields.setdefault("source_file", info.source_file)
    try:
        item = model.model_validate(fields)
    except ValidationError as e:
        err = e.errors()[0]
        raise EditError(f"invalid {_SINGULAR[kind]}: {err['msg']} ({'.'.join(map(str, err['loc']))})") from e
    return edl.model_copy(update={kind: [*items, item]})


def update_item(edl: EDL, manifest: ClipsManifest, *, kind: str, item_id: str,
                **fields: Any) -> EDL:
    model, id_field, _ = _ITEMS[kind]
    items = list(getattr(edl, kind))
    for i, x in enumerate(items):
        if getattr(x, id_field) == item_id:
            if "anchor_clip_id" in fields:
                _find(edl, fields["anchor_clip_id"])
            items[i] = _merge(model, x, fields)
            return edl.model_copy(update={kind: items})
    raise EditError(f"no {_SINGULAR[kind]} {item_id!r}; have "
                    f"{', '.join(getattr(x, id_field) for x in items) or 'none'}")


def remove_item(edl: EDL, manifest: ClipsManifest, *, kind: str, item_id: str) -> EDL:
    _, id_field, _ = _ITEMS[kind]
    items = list(getattr(edl, kind))
    kept = [x for x in items if getattr(x, id_field) != item_id]
    if len(kept) == len(items):
        raise EditError(f"no {_SINGULAR[kind]} {item_id!r}")
    return edl.model_copy(update={kind: kept})


def new_edit(edl: EDL | None, manifest: ClipsManifest, *, target_platform: str,
             target_duration: float | None = None) -> EDL:
    """An empty edit for a platform (the agent then adds clips itself)."""
    from ..presets.platform_presets import get_platform_specs

    try:
        spec = get_platform_specs(target_platform)
    except ValueError as e:
        raise EditError(str(e)) from e
    w, h = map(int, spec["resolution"].split("x"))
    return EDL(target_platform=target_platform, width=w, height=h,
               target_duration=target_duration)
