"""LangGraph orchestration of the stage pipeline with a human review interrupt.

    watch -> plan -> broll -> cut_audit -> geometry -> review -> [interrupt] -> render

State holds only plain, checkpointable values (paths, names, statuses); the
heavy collaborators (transcriber, planner provider, detector) are injected via
``Deps``. The graph pauses at ``gate`` with the review sheets as the interrupt
payload; resume with ``Command(resume={"approve": True, "notes": "..."})``.

Requires the ``graph`` extra (``langgraph``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from .llm import Provider, get_provider
from .stages.asr import FasterWhisperTranscriber, Transcriber
from .stages.chunked_asr import ChunkConfig
from .stages.detectors import Detector, get_detector
from .workspace import Workspace


class EditState(TypedDict, total=False):
    inputs: list[str]
    prompt: str
    target: str
    target_duration: float | None
    broll_library: str | None
    workspace: str
    status: str
    review_sheets: list[str]
    review_warnings: list[str]
    cut_audit_problems: list[str]
    decision: dict[str, Any]
    output: str
    qc_passed: bool
    qc_problems: list[str]


@dataclass
class Deps:
    transcriber: Transcriber | None = None
    provider: Provider | None = None
    detector: Detector | None = None
    encoder: str = "auto"
    quality: str = "high"
    home: Path | None = None
    language: str | None = None
    asr_chunking: ChunkConfig | None = None  # chunked/parallel ASR for long clips
    extra: dict[str, Any] = field(default_factory=dict)


def build_graph(deps: Deps | None = None, checkpointer=None):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    from .stages.broll import index_library, match_broll
    from .stages.cut_audit import run_cut_audit
    from .stages.geometry import run_geometry
    from .stages.plan import plan
    from .stages.render import render
    from .stages.review import build_review, decide
    from .stages.watch import watch

    deps = deps or Deps()
    provider = deps.provider or get_provider()

    def ws_of(state: EditState) -> Workspace:
        return Workspace.load(Path(state["workspace"]))

    def watch_node(state: EditState) -> EditState:
        inputs = state["inputs"]
        ws = Workspace.open(inputs if len(inputs) > 1 else inputs[0], home=deps.home)
        watch(ws, transcriber=deps.transcriber or FasterWhisperTranscriber(),
              language=deps.language, chunking=deps.asr_chunking)
        return {"workspace": str(ws.root), "status": "watched"}

    def plan_node(state: EditState) -> EditState:
        plan(ws_of(state), state["prompt"], provider, target_platform=state.get("target", "youtube"),
             target_duration=state.get("target_duration"))
        return {"status": "planned"}

    def broll_node(state: EditState) -> EditState:
        lib = state.get("broll_library")
        if lib:
            index_library(Path(lib))
        match_broll(ws_of(state), Path(lib) if lib else None, provider=provider if lib else None)
        return {"status": "broll_selected"}

    def cut_audit_node(state: EditState) -> EditState:
        _, report = run_cut_audit(ws_of(state))
        if report.passed:
            return {"status": "audited", "cut_audit_problems": []}
        return {"status": "cut_audit_failed",
                "cut_audit_problems": [f"{v.kind}: {v.message}" for v in report.unresolved]}

    def geometry_node(state: EditState) -> EditState:
        det = deps.detector or get_detector()
        try:
            run_geometry(ws_of(state), detector=det)
        finally:
            if deps.detector is None:
                det.close()
        return {"status": "geometry_done"}

    def review_node(state: EditState) -> EditState:
        ws = ws_of(state)
        review = build_review(ws)
        return {"status": "review_pending",
                "review_sheets": [str(ws.root / s) for s in review.sheets],
                "review_warnings": review.warnings}

    def gate_node(state: EditState) -> EditState:
        decision = interrupt({"review_sheets": state.get("review_sheets", []),
                              "warnings": state.get("review_warnings", []),
                              "workspace": state["workspace"]})
        approve = bool(decision.get("approve")) if isinstance(decision, dict) else bool(decision)
        notes = decision.get("notes", "") if isinstance(decision, dict) else ""
        decide(ws_of(state), approve, notes)
        return {"decision": {"approve": approve, "notes": notes},
                "status": "approved" if approve else "rejected"}

    def render_node(state: EditState) -> EditState:
        res = render(ws_of(state), encoder=deps.encoder, quality=deps.quality,  # type: ignore[arg-type]
                     require_review=True)
        return {"status": "rendered", "output": str(res.output), "qc_passed": res.qc.passed,
                "qc_problems": res.qc.problems}

    g = StateGraph(EditState)
    for name, fn in [("watch", watch_node), ("plan", plan_node), ("broll", broll_node),
                     ("cut_audit", cut_audit_node), ("geometry", geometry_node),
                     ("review", review_node), ("gate", gate_node), ("render", render_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "watch")
    g.add_edge("watch", "plan")
    g.add_edge("plan", "broll")
    g.add_edge("broll", "cut_audit")
    g.add_conditional_edges("cut_audit",
                            lambda s: "geometry" if s["status"] == "audited" else END)
    g.add_edge("geometry", "review")
    g.add_edge("review", "gate")
    g.add_conditional_edges("gate", lambda s: "render" if s["status"] == "approved" else END)
    g.add_edge("render", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
