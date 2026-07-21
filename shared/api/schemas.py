from __future__ import annotations

from pydantic import BaseModel, Field


class SnapshotRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=20_000)


class CommitRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=20_000)
    query_time: str = Field(default="", max_length=128)


class CommitAccepted(BaseModel):
    run_id: str
    turn_id: str
    path: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    events_url: str


class SnapshotAccepted(BaseModel):
    turn_id: str
    revision: int
    events_url: str
