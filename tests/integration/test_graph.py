import pytest

from llm_video_editor.stages.plan import ReduceResult
from tests.conftest import requires_ffmpeg
from tests.fakes import FakeDetector, FakeProvider, FakeTranscriber

pytestmark = [requires_ffmpeg, pytest.mark.integration]
langgraph = pytest.importorskip("langgraph")


def _deps(tmp_path):
    from llm_video_editor.graph import Deps

    def responder(schema, prompt):
        import re

        cid = sorted(set(re.findall(r"chunk (v[0-9a-f]{8})#", prompt)))[0]
        return ReduceResult.model_validate({"selections": [
            {"clip_id": cid, "first_segment": 0, "last_segment": 1, "purpose": "all",
             "priority": 1}]})

    return Deps(transcriber=FakeTranscriber(), provider=FakeProvider(responder),
                detector=FakeDetector(), quality="draft", home=tmp_path / "home")


@pytest.mark.parametrize("approve", [True, False])
def test_graph_pauses_at_review_then_resumes(media_dir, tmp_path, approve):
    from langgraph.types import Command

    from llm_video_editor.graph import build_graph

    app = build_graph(_deps(tmp_path))
    cfg = {"configurable": {"thread_id": f"t-{approve}"}}
    first = app.invoke({"inputs": [str(media_dir)], "prompt": "4s reel", "target": "reels"}, cfg)
    assert first["status"] == "review_pending"
    payload = first["__interrupt__"][0].value
    assert payload["review_sheets"] and all(s.endswith(".png") for s in payload["review_sheets"])
    assert "output" not in first

    final = app.invoke(Command(resume={"approve": approve, "notes": "ok"}), cfg)
    if approve:
        assert final["status"] == "rendered" and final["qc_passed"], final.get("qc_problems")
    else:
        assert final["status"] == "rejected" and "output" not in final
