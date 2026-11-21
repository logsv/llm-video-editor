"""``llmve auto``: the whole edit in one command, decided and approved automatically.

watch -> plan -> [B-roll library] -> auto choices (pacing, captions, chapters, music,
cutaways, graphics) -> cut-audit -> geometry -> review (auto-approved) -> render ->
publish.

Core steps (watch, plan, cut-audit, render) must succeed. Every enhancement is
best-effort: if it cannot run (no network, no Node, no API key) it is recorded as
skipped with the reason and the edit carries on. Every decision lands in
``auto_report.json`` and the printed summary, so nothing happens silently.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..llm import Provider
from ..presets.platform_presets import get_platform_specs
from ..schemas import EDL, CaptionSettings, ClipsManifest, MusicTrack, Pacing, Strict, Transcript
from ..schemas.base import Checkpoint
from ..workspace import Workspace

VERTICAL = ("shorts", "reels", "tiktok")
AUTO_APPROVE_NOTE = "auto-approved by `llmve auto` (review sheet kept for reference)"


@dataclass
class AutoOptions:
    target: str = "youtube"
    target_duration: float | None = None
    out: Path | None = None
    pacing: Literal["auto", "tight", "natural", "off"] = "auto"
    captions: Literal["auto", "burn", "srt"] = "auto"
    chapters: bool = True
    music: str = "auto"  # auto | off | path to an audio file
    music_query: str | None = None
    cutaways: bool = True
    humor: bool = True
    graphics: bool = True
    name: str | None = None  # "Jane Doe|Founder" -> lower third in the first minute
    broll_library: Path | None = None
    approve: bool = True  # False: stop at the review sheet like `llmve run`
    publish: bool = True
    geocode: bool = False
    use_location: bool = False
    draft: bool = False
    encoder: str = "auto"
    detector: str = "auto"
    chunk_asr_over: float | None = 10.0  # minutes; None = never chunk
    language: str | None = None
    asr_model: str = "distil-large-v3"
    brief_extra: list[str] = field(default_factory=list)

    @property
    def vertical(self) -> bool:
        return self.target in VERTICAL


class StepReport(Strict):
    step: str
    status: Literal["done", "skipped", "failed"]
    detail: str = ""
    seconds: float = 0.0


class AutoReport(Checkpoint):
    inputs: list[str]
    prompt: str
    target: str
    workspace: str
    output: str | None = None
    upload: str | None = None
    approved: bool = False
    steps: list[StepReport] = field(default_factory=list)  # type: ignore[assignment]
    warnings: list[str] = field(default_factory=list)  # type: ignore[assignment]


class StepFailed(RuntimeError):
    """A core step failed; the report says which and why."""


@dataclass
class _Run:
    report: AutoReport
    on_step: Callable[[StepReport], None] | None = None

    def step(self, name: str, fn: Callable[[], str | None], *, core: bool = False) -> bool:
        t = time.monotonic()
        try:
            detail = fn() or ""
            rep = StepReport(step=name, status="skipped" if detail.startswith("skipped:")
                             else "done", detail=detail.removeprefix("skipped:").strip(),
                             seconds=round(time.monotonic() - t, 1))
        except Exception as e:  # noqa: BLE001 - optional steps must never stop the edit
            rep = StepReport(step=name, status="failed", detail=f"{type(e).__name__}: {e}"[:600],
                             seconds=round(time.monotonic() - t, 1))
            if core:
                self._add(rep)
                raise StepFailed(f"{name} failed: {rep.detail}") from e
            self.report.warnings.append(f"{name}: {rep.detail}")
        self._add(rep)
        return rep.status == "done"

    def _add(self, rep: StepReport) -> None:
        self.report.steps.append(rep)
        if self.on_step:
            self.on_step(rep)


# -- automatic choices ------------------------------------------------------------------
def choose_pacing(edl: EDL, manifest: ClipsManifest, opts: AutoOptions) -> tuple[Pacing, str]:
    if opts.pacing == "off":
        return Pacing(), "off"
    a_roll = [manifest.get(c.source_clip_id) for c in edl.clips if c.role == "a_roll"]
    talking = [i for i in a_roll if i is not None and i.kind == "talking_head"]
    if opts.pacing == "auto" and not talking:
        return Pacing(), "off (no talking-head footage)"
    natural = opts.pacing == "natural"
    gap = 1.0 if natural else (0.4 if opts.vertical else 0.6)
    # punch-ins need resolution to spare: only when sources are at least 1080 px tall
    sharp = bool(talking) and min(i.height for i in talking) >= 1080
    punch = "alternate" if (sharp and not natural) else "off"
    return (Pacing(tighten=True, max_gap=gap, drop_fillers=True, punch_in=punch),
            f"{'natural' if natural else 'tight'} (pauses > {gap:g}s removed, punch-ins {punch})")


def choose_captions(edl: EDL, opts: AutoOptions) -> tuple[CaptionSettings, str]:
    burn = opts.captions == "burn" or (opts.captions == "auto" and opts.vertical)
    if burn:
        return (edl.captions.model_copy(update={"enabled": True, "style": "word_highlight"}),
                "burned in, word highlight (+ .srt)")
    return edl.captions.model_copy(update={"enabled": False}), "sidecar .srt (upload in Studio)"


def music_query(opts: AutoOptions) -> str:
    if opts.music_query:
        return opts.music_query
    return ("upbeat energetic background instrumental" if opts.vertical
            else "calm ambient background instrumental")


# -- the run ------------------------------------------------------------------------------
def auto_edit(inputs: list[str], prompt: str, provider: Provider, opts: AutoOptions, *,
              transcriber=None, home: Path | None = None,
              on_step: Callable[[StepReport], None] | None = None) -> AutoReport:
    from ..stages.broll import index_library, match_broll
    from ..stages.plan import plan
    from ..stages.watch import watch
    from .chunked_asr import ChunkConfig

    ws = Workspace.open(inputs if len(inputs) > 1 else inputs[0], home=home)
    report = AutoReport(inputs=[str(i) for i in inputs], prompt=prompt, target=opts.target,
                        workspace=str(ws.root))
    run = _Run(report, on_step)
    get_platform_specs(opts.target)  # unknown platform: fail before any work
    brief = " ".join([prompt, *opts.brief_extra]).strip()
    state: dict = {}

    def do_watch():
        chunking = (ChunkConfig(min_duration=opts.chunk_asr_over * 60)
                    if opts.chunk_asr_over else None)
        from .asr import FasterWhisperTranscriber

        res = watch(ws, transcriber=transcriber or FasterWhisperTranscriber(opts.asr_model),
                    language=opts.language, chunking=chunking)
        talking = sum(c.kind == "talking_head" for c in res.manifest.clips)
        return (f"{len(res.manifest.clips)} clip(s), {talking} talking-head, "
                f"{res.manifest.total_duration:.0f}s of footage" + (" (cached)" if res.cached else ""))

    run.step("watch", do_watch, core=True)

    def do_plan():
        asset_plan, edl = plan(ws, prompt, provider, target_platform=opts.target,
                               target_duration=opts.target_duration)
        return f"{len(edl.clips)} clips, {edl.total_duration:.0f}s of {asset_plan.target_duration:.0f}s target"

    run.step("plan", do_plan, core=True)

    if opts.broll_library:
        def do_broll():
            index_library(opts.broll_library)
            sel = match_broll(ws, opts.broll_library, provider=provider)
            return f"{sum(1 for s in sel.slots if s.chosen)} of {len(sel.slots)} slots filled"

        run.step("b-roll library", do_broll)

    manifest = Workspace.read(ws.clips_manifest, ClipsManifest)
    transcript = Workspace.read(ws.transcript, Transcript)

    def edit(update: dict) -> EDL:
        edl = Workspace.read(ws.edl_raw, EDL).model_copy(update=update)
        Workspace.write(ws.edl_raw, edl)
        return edl

    def do_pacing():
        pacing, why = choose_pacing(Workspace.read(ws.edl_raw, EDL), manifest, opts)
        edit({"pacing": pacing})
        return why if pacing.tighten else f"skipped: {why}"

    run.step("pacing", do_pacing)

    def do_captions():
        caps, why = choose_captions(Workspace.read(ws.edl_raw, EDL), opts)
        edit({"captions": caps})
        return why

    run.step("captions", do_captions)

    def do_chapters():
        from .chapters import suggest_chapters

        if not opts.chapters or opts.target != "youtube":
            return "skipped: chapters are for YouTube long-form"
        edl = Workspace.read(ws.edl_raw, EDL)
        names = suggest_chapters(edl, transcript, provider)
        if not names:
            return "skipped: too short or no clear topic changes for YouTube's chapter rules"
        edit({"clips": [c.model_copy(update={"chapter": names.get(c.clip_id)})
                        for c in edl.clips]})
        state["chapters"] = names
        return f"{len(names)} chapters: " + "; ".join(names.values())

    run.step("chapters", do_chapters)

    def do_music():
        from ..media.library import fetch, register, search

        if opts.music == "off":
            return "skipped: --music off"
        edl = Workspace.read(ws.edl_raw, EDL)
        if any(m.role == "music" for m in edl.music):
            return "skipped: the edit already has music"
        if opts.music != "auto":
            path = Path(opts.music).expanduser()
            credit = register(ws, path, kind="music", title=path.stem, license="provided by user",
                              provider="manual")
            file = credit.file
            label = f"{path.name} (yours: add its credit with `llmve media add` if needed)"
        else:
            q = music_query(opts)
            res = search(q, "music", length="long" if edl.total_duration > 240 else "medium",
                         ws=ws)
            hit = next((r for r in res.results if (r.duration or 0) >= 30), None)
            if hit is None:
                why = "; ".join(res.errors) or "no long-enough track"
                return f"skipped: no licensed music found for {q!r} ({why})"
            file = str(fetch(ws, hit))
            label = f"{hit.title!r} by {hit.creator or 'unknown'} ({hit.license}, {hit.provider})"
        bed = MusicTrack(track_id="auto-music", file=file, anchor_clip_id=edl.clips[0].clip_id,
                         volume_db=-16.0 if opts.vertical else -20.0, loop=True, duck=True)
        edit({"music": [*edl.music, bed]})
        return f"{label}, ducked under speech"

    run.step("music", do_music)

    def do_cutaways():
        from .cutaways import Budget, fetch, suggest

        if not opts.cutaways:
            return "skipped: --no-cutaways"
        plan_ = suggest(ws, provider, budget=Budget(humor=opts.humor), brief=brief)
        if not plan_.slots:
            return "skipped: none needed"
        plan_ = fetch(ws)
        placed = [s for s in plan_.slots if s.overlay_id]
        notes = {s.note for s in plan_.slots if not s.overlay_id and s.note}
        if not placed:
            return f"skipped: {len(plan_.slots)} suggested, none found ({'; '.join(notes)})"
        return (f"{len(placed)} of {len(plan_.slots)} placed: "
                + "; ".join(f"{s.kind} '{s.query}'" for s in placed))

    run.step("cutaways", do_cutaways)

    def do_graphics():
        from ..hyperframes import toolchain
        from .graphics import add_graphic, chapter_cards

        if not opts.graphics:
            return "skipped: --no-graphics"
        if (toolchain.node_major() or 0) < toolchain.MIN_NODE_MAJOR:
            return "skipped: needs Node.js 22+ (then `llmve graphics doctor --install`)"
        made = []
        if state.get("chapters"):
            made += [g.graphic_id for g in chapter_cards(ws, duration=2.5)]
        if opts.name:
            title, _, subtitle = opts.name.partition("|")
            edl = Workspace.read(ws.edl_raw, EDL)
            first = next((c for c in edl.clips if c.duration >= 5), edl.clips[0])
            made.append(add_graphic(ws, anchor=first.clip_id, template="lower-third",
                                    offset=min(1.5, first.duration / 3),
                                    duration=min(4.0, max(1.0, first.duration - 1.5)),
                                    variables={"title": title.strip(),
                                               "subtitle": subtitle.strip()}).graphic_id)
        if not made:
            return "skipped: no chapters and no --name for a lower third"
        toolchain.ensure()  # first run installs the pinned toolchain
        return f"{len(made)} graphic(s): {', '.join(made)}"

    run.step("graphics", do_graphics)

    finish(ws, provider, opts, run=run, brief=brief, inputs=inputs)
    return report


def _stem(inputs: list[str]) -> str:
    p = Path(inputs[0])
    return (p.stem if p.suffix else p.name) or "edit"


def finish(ws: Workspace, provider: Provider, opts: AutoOptions, *, run: _Run, brief: str,
           inputs: list[str]) -> None:
    """cut-audit -> geometry -> review (auto-approved) -> render -> publish."""
    from .cut_audit import run_cut_audit

    report = run.report

    def do_audit():
        _, rep = run_cut_audit(ws)
        if not rep.passed:  # one automatic retry: drop the duration budget
            edl = Workspace.read(ws.edl_raw, EDL)
            if any(v.kind == "duration_budget" for v in rep.unresolved) and edl.target_duration:
                Workspace.write(ws.edl_raw, edl.model_copy(update={"target_duration": None}))
                _, rep = run_cut_audit(ws)
                report.warnings.append("duration budget could not be met; kept the cut longer")
        if not rep.passed:
            raise RuntimeError("; ".join(f"{v.kind}: {v.message}" for v in rep.unresolved))
        fixed = sum(v.fixed for v in rep.violations)
        return (f"{rep.clips_after} clips, {rep.duration_after:.1f}s, {fixed} fixes"
                + (f", {rep.pauses_removed:.1f}s of pauses removed" if rep.pauses_removed else ""))

    run.step("cut-audit", do_audit, core=True)

    def do_geometry():
        from .detectors import get_detector
        from .geometry import run_geometry

        det = get_detector(opts.detector)
        try:
            gmap = run_geometry(ws, detector=det)
        finally:
            det.close()
        return f"detector {gmap.detector}, {len(gmap.placements)} placements"

    run.step("geometry", do_geometry)

    def do_review():
        from .review import build_review, decide

        review = build_review(ws)
        report.warnings += [f"review: {w}" for w in review.warnings]
        if not opts.approve:
            return f"skipped: waiting for your approval ({', '.join(review.sheets)})"
        decide(ws, True, AUTO_APPROVE_NOTE)
        report.approved = True
        return f"approved automatically; sheet kept at {', '.join(review.sheets)}"

    run.step("review", do_review, core=True)
    if not report.approved:
        Workspace.write(ws.root / "auto_report.json", report)
        return

    def do_render():
        from .render import render

        prev = report.output and Path(report.output)
        out = opts.out or prev or Path.cwd() / f"{_stem(inputs)}-{opts.target}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        res = render(ws, out=out, encoder=opts.encoder,
                     quality="draft" if opts.draft else "high",
                     burn_captions=True, require_review=True)
        report.output = str(res.output)
        report.warnings += [f"render: {w}" for w in res.qc.warnings]
        if not res.qc.passed:
            report.warnings += [f"QC: {p}" for p in res.qc.problems]
        return (f"{res.output.name}: {res.qc.duration:.1f}s, {res.qc.integrated_lufs} LUFS, "
                f"A/V {res.qc.av_offset * 1000:.0f} ms, QC {'passed' if res.qc.passed else 'FAILED'}")

    run.step("render", do_render, core=True)

    def do_publish():
        from .publish import publish

        if not opts.publish:
            return "skipped: --no-publish"
        pkg = publish(ws, provider, video=Path(report.output), brief=brief, geocode=opts.geocode,
                      use_location=opts.use_location)
        report.upload = str(Path(pkg.video).with_suffix(".upload.md"))
        report.warnings += [f"publish: {w}" for w in pkg.warnings]
        return f"{Path(report.upload).name}, {len(pkg.thumbnails)} thumbnails: {pkg.titles[0]!r}"

    run.step("publish", do_publish)
    Workspace.write(ws.root / "auto_report.json", report)
    if report.output:
        Workspace.write(Path(report.output).with_suffix(".auto_report.json"), report)




def finish_edit(ws: Workspace, provider: Provider, opts: AutoOptions, *,
                on_step: Callable[[StepReport], None] | None = None) -> AutoReport:
    """Re-run everything after an edit to edl_raw.json (the refine loop): no new
    transcription or planning, so a note costs seconds to minutes, not a full edit."""
    prev_path = ws.root / "auto_report.json"
    prev = Workspace.read(prev_path, AutoReport) if prev_path.exists() else None
    report = AutoReport(inputs=prev.inputs if prev else [str(p) for p in ws.sources],
                        prompt=prev.prompt if prev else "", target=opts.target,
                        workspace=str(ws.root), output=prev.output if prev else None)
    run = _Run(report, on_step)
    finish(ws, provider, opts, run=run, brief=" ".join([report.prompt, *opts.brief_extra]),
           inputs=report.inputs)
    return report
