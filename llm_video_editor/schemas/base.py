"""Shared base for versioned pipeline checkpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "1"


class Strict(BaseModel):
    """Base model that rejects unknown fields (keeps LLM output honest)."""

    model_config = ConfigDict(extra="forbid")


class Checkpoint(Strict):
    """A JSON checkpoint written to the workspace by one pipeline stage."""

    schema_version: Literal["1"] = SCHEMA_VERSION
