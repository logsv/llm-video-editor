import json

import pytest

from llm_video_editor.agent.edits import EditConflict, EditError, EditStore
from llm_video_editor.schemas import EDL, EDLClip
from llm_video_editor.workspace import Workspace


def _ws(tmp_path):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    return Workspace(tmp_path / "ws", [src])


def _edl(n=2):
    return EDL(target_platform="youtube", width=1920, height=1080, clips=[
        EDLClip(clip_id=f"c{i}", source_clip_id="va", source_file="/a.mp4", start=i * 10,
                end=i * 10 + 5) for i in range(1, n + 1)])


def trim(edl, cid, end):
    return edl.model_copy(update={"clips": [c.model_copy(update={"end": end})
                                            if c.clip_id == cid else c for c in edl.clips]})


def test_versions_log_and_mirror(tmp_path):
    ws = _ws(tmp_path)
    Workspace.write(ws.edl_raw, _edl())
    store = EditStore(ws)
    assert store.current().clips[0].clip_id == "c1" and store.head() == 1  # imported
    ch = store.apply("trim_clip", {"clip_id": "c1", "end": 13}, "drop the stumble",
                     lambda e: trim(e, "c1", 13))
    assert (ch.version, ch.previous, ch.changed) == (2, 1, True)
    assert "changed c1" in ch.summary
    assert Workspace.read(ws.edl_raw, EDL).clips[0].end == 13  # stages see it
    noop = store.apply("trim_clip", {}, "same again", lambda e: trim(e, "c1", 13))
    assert not noop.changed and store.head() == 2
    log = [json.loads(x) for x in store.log_path.read_text().splitlines()]
    assert [r["tool"] for r in log] == ["import", "trim_clip", "trim_clip"]
    assert log[1]["reason"] == "drop the stumble" and log[1]["version_after"] == 2
    md = store.decisions_path.read_text()
    assert "v0002 **trim_clip**" in md and "drop the stumble" in md


def test_undo_is_a_new_version(tmp_path):
    ws = _ws(tmp_path)
    Workspace.write(ws.edl_raw, _edl())
    store = EditStore(ws)
    store.apply("t", {}, "", lambda e: trim(e, "c1", 13))
    ch = store.undo(1, "went too far")
    assert ch.version == 3 and store.current().clips[0].end == 15
    assert store.load(2).clips[0].end == 13  # history kept
    with pytest.raises(EditError, match="does not exist"):
        store.undo(9, "")


def test_conflicts_and_hand_edits(tmp_path):
    ws = _ws(tmp_path)
    Workspace.write(ws.edl_raw, _edl())
    store = EditStore(ws)
    seen = store.head() if store.current() else 0
    # the user edits edl_raw.json by hand while the agent works
    Workspace.write(ws.edl_raw, trim(_edl(), "c2", 22))
    with pytest.raises(EditConflict, match="version 2, not 1"):
        store.apply("t", {}, "", lambda e: trim(e, "c1", 13), expected_version=seen)
    assert store.history()[-1]["tool"] == "manual edit"
    ch = store.apply("t", {}, "", lambda e: trim(e, "c1", 13), expected_version=2)
    edl = store.current()
    assert ch.version == 3 and edl.clips[0].end == 13 and edl.clips[1].end == 22  # both kept


def test_invalid_results_are_rejected(tmp_path):
    from pydantic import ValidationError

    ws = _ws(tmp_path)
    Workspace.write(ws.edl_raw, _edl())
    store = EditStore(ws)
    bad = lambda e: e.model_copy(update={"clips": [e.clips[0].model_copy(update={"zoom": 5.0})]})  # noqa: E731
    with pytest.raises(ValidationError):
        store.apply("set_clip", {}, "", bad)
    assert store.head() == 1


def test_no_edit_yet(tmp_path):
    with pytest.raises(EditError, match="no edit yet"):
        EditStore(_ws(tmp_path)).current()
