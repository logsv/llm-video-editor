"""Stage 6 - ``publish``: everything to upload a render, next to the video file.

Writes ``<video>.upload.md`` (copy-paste ready: titles, description with chapters
and credits, tags, hashtags, thumbnails, location, checklist), ``<video>.upload.json``
and three thumbnails ``<video>.thumb1.jpg`` .. ``thumb3.jpg``.

The planner writes the copy from text only (numbered transcript lines of the edit,
chapter titles, the brief, and the place name only with ``use_location``); it picks
thumbnail moments by segment id, resolved locally. YouTube's limits are enforced
here, not trusted to the model.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..ffmpeg.probe import probe
from ..llm import Provider, structured
from ..media.library import load_credits, sidecar_credit
from ..schemas import EDL, ClipsManifest, Credit, RenderQC, Strict, Transcript
from ..schemas.publish import SourceMeta, Thumbnail, UploadPackage
from ..workspace import Workspace
from .chapters import chapter_marks, chapter_warnings, description_block, stamp
from .source_meta import first_meta, reverse_geocode
from .thumbnails import pick_frame, render_thumbnail

TITLE_MAX, TITLE_GOOD = 100, 60
DESCRIPTION_MAX = 5000
TAGS_MAX = 500
MATERIAL_MAX = 30_000
CATEGORIES = ("Film & Animation", "Autos & Vehicles", "Music", "Pets & Animals", "Sports",
              "Travel & Events", "Gaming", "People & Blogs", "Comedy", "Entertainment",
              "News & Politics", "Howto & Style", "Education", "Science & Technology",
              "Nonprofits & Activism")

SYSTEM = (
    "You write YouTube upload copy that ranks and gets clicked without lying. Titles: "
    "front-load the main keyword, create curiosity or promise a concrete payoff the video "
    "really delivers, at most 60 characters, no clickbait the content does not back up, no "
    "ALL CAPS sentences. Description: the first two lines are the hook shown in search "
    "(include the main keyword naturally), then 2-4 short paragraphs on what the viewer "
    "learns or sees; no links, no timestamps, no invented facts. Thumbnail text: 2-5 punchy "
    "words that add to the title instead of repeating it, true to the moment shown. Answer "
    "strictly in the requested JSON schema."
)


class _Thumb(Strict):
    clip_id: str
    segment_id: int | None = Field(
        None, description="transcript segment whose moment makes a striking frame")
    at: float | None = Field(None, description="or: source seconds in the clip (footage "
                             "without speech); the sharpest frame near it is used")
    text: str = Field(description="2-5 words, no emoji")
    style: Literal["bold", "question", "reveal"]
    why: str = Field(description="why a viewer would click, <= 15 words")


class _Copy(Strict):
    """Upload copy; written by the planner, or by the editor agent in agent mode."""

    titles: list[str] = Field(description="3 options, best first, <= 60 characters each")
    hook: str = Field(description="first two description lines, <= 150 characters")
    body: str = Field(description="2-4 short paragraphs separated by blank lines")
    tags: list[str] = Field(description="10-20 search phrases, most specific first")
    hashtags: list[str] = Field(description="3 hashtags without spaces")
    category: str = Field(description="one YouTube category name")
    pinned_comment: str = Field(description="a question that invites replies, <= 200 characters")
    thumbnails: list[_Thumb] = Field(description="3 ideas, each a different style")


# -- material ---------------------------------------------------------------------------
def edit_lines(edl: EDL, transcript: Transcript, starts: list[float]) -> list[str]:
    """``[clip c003 #12 @ 1:05] text`` for every spoken line of the edit, in output order."""
    lines = []
    for clip, t0 in zip(edl.clips, starts, strict=True):
        ct = transcript.for_clip(clip.source_clip_id)
        if clip.role != "a_roll" or ct is None:
            continue
        for seg in ct.segments:
            if seg.end > clip.start and seg.start < clip.end:
                at = t0 + max(0.0, seg.start - clip.start)
                lines.append(f"[clip {clip.clip_id} #{seg.id} @ {stamp(at)}] {seg.text}")
    return lines


def _fit(lines: list[str], limit: int = MATERIAL_MAX) -> str:
    text = "\n".join(lines)
    if len(text) <= limit:
        return text
    keep = max(1, int(len(lines) * limit / len(text)))
    step = len(lines) / keep
    return "\n".join(lines[int(i * step)] for i in range(keep)) + "\n(sampled evenly)"


# -- YouTube limits ---------------------------------------------------------------------
def clean(text: str) -> str:
    return text.replace("<", "").replace(">", "").strip()


def fix_titles(titles: list[str], warnings: list[str]) -> list[str]:
    out = []
    for t in titles:
        t = re.sub(r"\s+", " ", clean(t))
        if not t:
            continue
        if len(t) > TITLE_MAX:
            t = t[:TITLE_MAX].rsplit(" ", 1)[0]
        if len(t) > TITLE_GOOD:
            warnings.append(f"title is {len(t)} characters; search shows about {TITLE_GOOD}: {t!r}")
        out.append(t)
    return list(dict.fromkeys(out)) or ["Untitled video"]


def fix_hashtags(tags: list[str]) -> list[str]:
    out = []
    for h in tags:
        h = "#" + re.sub(r"[^\w]", "", h.lstrip("#"))
        if len(h) > 1 and h.lower() not in {x.lower() for x in out}:
            out.append(h)
    return out[:3]


def fix_tags(tags: list[str]) -> list[str]:
    """Dedupe and keep within YouTube's 500 characters (spaces cost 2 for quotes)."""
    out, used = [], 0
    for t in tags:
        t = re.sub(r"\s+", " ", clean(t).replace(",", " ")).strip()
        if not t or t.lower() in {x.lower() for x in out}:
            continue
        cost = len(t) + (2 if " " in t else 0) + (1 if out else 0)
        if used + cost > TAGS_MAX:
            break
        out.append(t)
        used += cost
    return out


# -- credits ----------------------------------------------------------------------------
def used_credits(ws: Workspace, edl: EDL) -> tuple[list[Credit], list[str]]:
    """Credits for every third-party file the cut uses (music, sfx, voice-over, B-roll)."""
    registry = {c.file: c for c in load_credits(ws).items}
    own = {str(p.resolve()) for p in ws.sources}
    files = [m.file for m in edl.music if not m.generator]  # generated narration: disclosed
    files += [o.source_file for o in edl.overlays]
    files += [c.source_file for c in edl.clips if c.role == "broll"]
    credits, missing = [], []
    for f in dict.fromkeys(str(Path(x).expanduser().resolve()) for x in files):
        if f in own:
            continue
        c = registry.get(f) or sidecar_credit(f)
        if c is None:
            missing.append(f"no licence/credit recorded for {Path(f).name}: check it before "
                           "uploading (`llmve media add` records one)")
        else:
            credits.append(c)
    return credits, missing


def credit_line(c: Credit) -> str:
    if c.attribution:
        line = c.attribution
    else:
        by = f" by {c.creator}" if c.creator else ""
        line = f"{c.title}{by} ({c.license})"
    return line + (f" {c.landing_url}" if c.landing_url and c.landing_url not in line else "")


def ai_voices(edl: EDL) -> list[str]:
    """Engines and voices of generated narration in the cut, e.g. ["kokoro:af_heart"]."""
    return sorted({m.generator for m in edl.music if m.role == "voiceover" and m.generator})


def build_description(copy: _Copy, chapters: list[tuple[float, str]], credits: list[Credit],
                      hashtags: list[str], ai_voice: list[str] | None = None) -> str:
    parts = [clean(copy.hook), clean(copy.body)]
    if chapters:
        parts.append("Chapters\n" + description_block(chapters).strip())
    lines = [f"- {credit_line(c)}" for c in credits]
    if ai_voice:
        engines = sorted({v.split(":")[0].capitalize() for v in ai_voice})
        lines.append(f"- Narration: AI-generated voice ({', '.join(engines)})")
    if lines:
        parts.append("Music and footage\n" + "\n".join(lines))
    if hashtags:
        parts.append(" ".join(hashtags))
    text = "\n\n".join(p for p in parts if p)
    if len(text) > DESCRIPTION_MAX:  # shorten the body, never the chapters or credits
        over = len(text) - DESCRIPTION_MAX
        body = clean(copy.body)[: max(0, len(clean(copy.body)) - over - 3)] + "..."
        text = "\n\n".join(p for p in [clean(copy.hook), body, *parts[2:]] if p)
    return text


# -- stage --------------------------------------------------------------------------------
def publish(ws: Workspace, provider: Provider | None, *, video: Path | None = None,
            brief: str = "", geocode: bool = False, use_location: bool = False,
            thumbnails: bool = True, accent: str = "#FFD400", prefer_hyperframes: bool = True,
            copy: _Copy | None = None) -> UploadPackage:
    """``copy`` given (agent mode: the agent wrote it): no LLM call; limits still enforced."""
    from .render import timeline

    edl = Workspace.read(ws.edl_audited, EDL)
    manifest = Workspace.read(ws.clips_manifest, ClipsManifest)
    transcript = Workspace.read(ws.transcript, Transcript)
    qc_path = ws.render_dir / "qc_report.json"
    qc = Workspace.read(qc_path, RenderQC) if qc_path.exists() else None
    video = Path(video or (qc.output if qc else ws.render_dir / f"{edl.target_platform}.mp4"))
    warnings: list[str] = []
    if not video.exists():
        warnings.append(f"{video} does not exist yet: render first, then re-run publish")

    segs, starts = timeline(edl, manifest)
    total = sum(s.duration for s in segs)
    chapters = [tuple(c) for c in qc.chapters] if qc and qc.chapters else chapter_marks(edl, starts)
    warnings += [f"{w} (YouTube will hide all chapters)" for w in chapter_warnings(chapters, total)]

    sources = [c.source_file for c in edl.clips if c.role == "a_roll"] or \
              [c.source_file for c in edl.clips]
    meta: SourceMeta = first_meta(list(dict.fromkeys(sources))) if sources else SourceMeta()
    if geocode and meta.latitude is not None:
        meta = meta.model_copy(update={"place": reverse_geocode(meta.latitude, meta.longitude)})
        if meta.place is None:
            warnings.append("reverse geocoding failed (offline or rate limited)")

    lines = edit_lines(edl, transcript, starts)
    prompt = (
        f"VIDEO: {stamp(total)} long, {edl.target_platform}, {edl.width}x{edl.height}.\n"
        + (f"BRIEF: {brief}\n" if brief else "")
        + (f"FILMED IN: {meta.place}\n" if use_location and meta.place else "")
        + ("CHAPTERS:\n" + description_block(chapters) if chapters else "")
        + "\nTRANSCRIPT OF THE EDIT (pick thumbnail moments by clip id and #segment id):\n"
        + (_fit(lines) or "(no speech)"))
    if copy is None:
        if provider is None:
            raise ValueError("publish needs either copy (written by the agent) or a provider")
        copy = structured(provider, SYSTEM, prompt, _Copy)

    titles = fix_titles(copy.titles, warnings)
    hashtags = fix_hashtags(copy.hashtags)
    credits, missing = used_credits(ws, edl)
    warnings += missing
    voices = ai_voices(edl)
    description = build_description(copy, chapters, credits, hashtags, voices)
    if voices:
        warnings.append(f"AI narration ({', '.join(voices)}): the description says so. In "
                        "YouTube Studio, tick 'altered or synthetic content' if the voice "
                        "could be mistaken for a real person")
    category = copy.category if copy.category in CATEGORIES else ""
    if not category:
        warnings.append(f"category {copy.category!r} is not a YouTube category; pick one in Studio")

    thumbs: list[Thumbnail] = []
    if thumbnails:
        thumbs = make_thumbnails(edl, transcript, starts, copy.thumbnails, video, warnings,
                                 accent=accent, prefer_hyperframes=prefer_hyperframes)

    info = probe(video) if video.exists() else None
    pkg = UploadPackage(
        video=str(video), titles=titles, description=description, tags=fix_tags(copy.tags),
        hashtags=hashtags, category=category, pinned_comment=clean(copy.pinned_comment)[:500],
        chapters=chapters, credits=[credit_line(c) for c in credits], thumbnails=thumbs,
        source=meta, duration=round(info.duration if info else total, 3),
        width=info.width if info else edl.width, height=info.height if info else edl.height,
        warnings=warnings)
    Workspace.write(video.with_suffix(".upload.json"), pkg)
    video.with_suffix(".upload.md").write_text(upload_markdown(pkg, credits), encoding="utf-8")
    return pkg


def make_thumbnails(edl: EDL, transcript: Transcript, starts: list[float], ideas: list[_Thumb],
                    video: Path, warnings: list[str], *, accent: str,
                    prefer_hyperframes: bool) -> list[Thumbnail]:
    clips = {c.clip_id: (c, t0) for c, t0 in zip(edl.clips, starts, strict=True)}
    styles = ["bold", "question", "reveal"]
    out: list[Thumbnail] = []
    for idea in ideas:
        if len(out) == 3:
            break
        hit = clips.get(idea.clip_id)
        where = f"{idea.clip_id}@{idea.at}s" if idea.at is not None else \
            f"{idea.clip_id}#{idea.segment_id}"
        if hit is not None and idea.at is not None:
            clip, t0 = hit
            if not clip.start <= idea.at <= clip.end:
                warnings.append(f"thumbnail idea {where} is outside the clip "
                                f"({clip.start:.1f}-{clip.end:.1f}s)")
                continue
            a = max(clip.start, idea.at - 0.5)
            b = min(clip.end, idea.at + 0.5)
        else:
            ct = transcript.for_clip(hit[0].source_clip_id) if hit else None
            seg = next((s for s in ct.segments if s.id == idea.segment_id), None) \
                if ct and idea.segment_id is not None else None
            if hit is None or seg is None:
                warnings.append(f"thumbnail idea {where} is not in the cut")
                continue
            clip, t0 = hit
            a, b = max(seg.start, clip.start), min(seg.end, clip.end)
        style = idea.style if idea.style not in {t.style for t in out} else \
            next(s for s in styles if s not in {t.style for t in out})
        n = len(out) + 1
        target = video.with_name(f"{video.stem}.thumb{n}.jpg")
        frame = pick_frame(clip.source_file, a, max(b, a + 0.2),
                           video.with_name(f".{video.stem}.frame{n}.jpg"))
        frame = _graded(frame, edl, clip)
        text = re.sub(r"\s+", " ", clean(idea.text)).strip()[:40] or "Watch this"
        used = render_thumbnail(frame, text, style, target, width=edl.width, height=edl.height,
                                accent=accent, prefer_hyperframes=prefer_hyperframes)
        frame.unlink(missing_ok=True)
        out.append(Thumbnail(path=str(target), text=text, style=style,
                             t_out=round(t0 + a - clip.start, 3), why=clean(idea.why),
                             renderer=used))
    if len(out) < 3:
        warnings.append(f"only {len(out)} thumbnail(s) could be made")
    return out


def _graded(frame: Path, edl: EDL, clip) -> Path:
    """The thumbnail frame with the clip's correction and the video's look."""
    from ..ffmpeg.runner import ffmpeg
    from .grade import correction_filter, look_filter

    pre, post = correction_filter(clip.color)
    chain = ",".join(x for x in (pre, post, look_filter(edl.grade)) if x)
    if not chain:
        return frame
    out = frame.with_name(frame.stem + ".graded.jpg")
    ffmpeg("-i", str(frame), "-vf", chain, "-q:v", "2", str(out), timeout=60)
    frame.unlink(missing_ok=True)
    return out


def upload_markdown(p: UploadPackage, credits: list[Credit]) -> str:
    v = Path(p.video)
    rows = [f"# Upload: {v.name}", "",
            f"{stamp(p.duration)} · {p.width}x{p.height}"
            + (f" · category: {p.category}" if p.category else ""), ""]
    if p.warnings:
        rows += ["> **Check before uploading**", ""] + [f"> - {w}" for w in p.warnings] + [""]
    rows += ["## Title", "", f"**{p.titles[0]}** ({len(p.titles[0])} characters, recommended)", ""]
    if len(p.titles) > 1:
        rows += ["Alternatives:"] + [f"- {t} ({len(t)})" for t in p.titles[1:]] + [""]
    rows += ["## Description", "", "```text", p.description, "```", "",
             f"{len(p.description)} / {DESCRIPTION_MAX} characters.", ""]
    rows += ["## Tags", "", "```text", ", ".join(p.tags), "```", ""]
    if p.hashtags:
        rows += ["## Hashtags", "", " ".join(p.hashtags),
                 "(already at the end of the description; YouTube shows them above the title)", ""]
    if p.thumbnails:
        rows += ["## Thumbnails", "", "Upload one; A/B test the others with YouTube's "
                 "Test & Compare.", "", "| File | Text | Style | Moment | Why |", "|---|---|---|---|---|"]
        rows += [f"| ![{Path(t.path).name}]({Path(t.path).name}) `{Path(t.path).name}` | {t.text} | "
                 f"{t.style} | {stamp(t.t_out)} | {t.why} |" for t in p.thumbnails]
        rows.append("")
    if p.chapters:
        rows += ["## Chapters", "", "```text", description_block(p.chapters).strip(), "```", ""]
    if credits:
        rows += ["## Music and footage credits", "", "| Item | Licence | Credit required | Source |",
                 "|---|---|---|---|"]
        rows += [f"| {c.title} ({c.kind}) | {c.license} | {'yes' if c.attribution_required else 'no'}"
                 f" | {c.landing_url or c.provider} |" for c in credits]
        rows += ["", "Every credit line is already in the description. Tracks from YouTube's Audio "
                 "Library may still ask for their exact credit text: compare it there.", ""]
    s = p.source
    rows += ["## Video details (YouTube Studio > Details)", ""]
    if s.recorded_at:
        rows.append(f"- **Recording date:** {s.recorded_at[:10]}")
    if s.latitude is not None:
        where = s.place or f"{s.latitude:.4f}, {s.longitude:.4f}"
        rows.append(f"- **Video location:** {where} "
                    f"([map](https://www.openstreetmap.org/?mlat={s.latitude:.5f}"
                    f"&mlon={s.longitude:.5f}#map=12/{s.latitude:.4f}/{s.longitude:.4f})). "
                    "Public if you set it: only add it if you are happy to share where you filmed.")
    else:
        rows.append("- **Video location:** not in the footage's metadata")
    if s.device:
        rows.append(f"- **Filmed on:** {s.device}")
    rows += [f"- **Captions:** upload `{v.stem}.srt` (Subtitles > Add language > Upload file)"
             if v.with_suffix(".srt").exists() else "- **Captions:** none written",
             "- **Altered or synthetic content:** answer Yes if any AI-generated B-roll was used",
             "- **Audience:** set Made for kids / Not made for kids", ""]
    if p.pinned_comment:
        rows += ["## Pinned comment", "", p.pinned_comment, ""]
    return "\n".join(rows)


UploadCopy = _Copy  # public name for agent mode
UploadThumbnail = _Thumb
