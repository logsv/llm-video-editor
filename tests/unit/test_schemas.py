import json

import pytest
from pydantic import ValidationError

from llm_video_editor.schemas import CHECKPOINTS, EDL, EDLClip, Transcript


def test_every_checkpoint_exports_json_schema():
    for name, model in CHECKPOINTS.items():
        schema = model.model_json_schema()
        assert schema["type"] == "object", name
        assert "schema_version" in schema["properties"], name
        json.dumps(schema)


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        Transcript.model_validate({"clips": [], "surprise": 1})


def test_wrong_schema_version_rejected():
    with pytest.raises(ValidationError):
        Transcript.model_validate({"schema_version": "999", "clips": []})


def test_edl_roundtrip_and_output_starts():
    edl = EDL(
        target_platform="reels", width=1080, height=1920,
        clips=[
            EDLClip(clip_id="c001", source_clip_id="v1", source_file="/a.mp4", start=1, end=3),
            EDLClip(clip_id="c002", source_clip_id="v2", source_file="/b.mp4", start=0, end=1.5),
        ],
    )
    again = EDL.model_validate_json(edl.model_dump_json())
    assert again == edl
    assert again.total_duration == pytest.approx(3.5)
    assert again.output_starts() == [0.0, 2.0]
