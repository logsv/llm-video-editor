import os

import pytest

from llm_video_editor.schemas import Transcript
from llm_video_editor.workspace import SourceFingerprint, Workspace, batch_hash, resolve_inputs


def _touch(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_resolve_dir_glob_and_file(tmp_path):
    a = _touch(tmp_path / "raw" / "a.mp4")
    b = _touch(tmp_path / "raw" / "sub" / "b.MOV")
    _touch(tmp_path / "raw" / "readme.txt")
    assert resolve_inputs(tmp_path / "raw") == sorted([a.resolve(), b.resolve()])
    assert resolve_inputs(str(tmp_path / "raw" / "*.mp4")) == [a.resolve()]
    assert resolve_inputs(a) == [a.resolve()]
    with pytest.raises(FileNotFoundError):
        resolve_inputs(tmp_path / "nothing_here")
    with pytest.raises(ValueError):
        resolve_inputs(tmp_path / "raw" / "readme.txt")


def test_batch_hash_is_order_independent_and_tracks_changes(tmp_path):
    a = _touch(tmp_path / "a.mp4")
    b = _touch(tmp_path / "b.mp4")
    h1 = batch_hash([a, b])
    assert h1 == batch_hash([b, a])
    os.utime(b, ns=(1, 1))
    assert batch_hash([a, b]) != h1


def test_clip_id_is_path_free(tmp_path):
    a = _touch(tmp_path / "secret_name.mp4")
    cid = SourceFingerprint.of(a).clip_id
    assert cid.startswith("v") and "secret" not in cid and len(cid) == 9


def test_workspace_cache_records(tmp_path):
    _touch(tmp_path / "raw" / "a.mp4")
    ws = Workspace.open(tmp_path / "raw", home=tmp_path / "home")
    assert ws.root.parent == tmp_path / "home"
    key = ws.stage_key("watch", 1)
    assert not ws.is_fresh("watch", key)
    ws.write(ws.transcript, Transcript())
    ws.record("watch", key, [ws.transcript])
    reopened = Workspace.load(ws.root)
    assert reopened.is_fresh("watch", key)
    assert not reopened.is_fresh("watch", ws.stage_key("watch", 2))
    outside = _touch(tmp_path / "exports" / "final.mp4")
    ws.record("render", "k", [outside])
    assert ws.is_fresh("render", "k")
    ws.transcript.unlink()
    assert not Workspace.load(ws.root).is_fresh("watch", key)
