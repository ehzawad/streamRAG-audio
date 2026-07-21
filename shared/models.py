from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, model_validator

RETRIEVAL_QUERY_MAX_CHARS = 2_000


class InputSnapshot(BaseModel):
    turn_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=0)
    text: str = Field(max_length=20_000)


class TriggerDecision(BaseModel):
    action: Literal["wait", "retrieve", "keep_previous"]
    candidate_query_compatible: bool = False
    retrieval_query: str | None = Field(
        default=None,
        max_length=RETRIEVAL_QUERY_MAX_CHARS,
    )

    @model_validator(mode="after")
    def query_matches_action(self) -> TriggerDecision:
        if self.action == "retrieve" and not (self.retrieval_query or "").strip():
            raise ValueError("retrieve action requires retrieval_query")
        if self.action != "retrieve":
            self.retrieval_query = None
        if self.action == "wait":
            self.candidate_query_compatible = False
        elif self.action == "keep_previous":
            self.candidate_query_compatible = True
        return self


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    title: str
    url: str
    text: str
    snippet: str
    domain: str
    query_time: str
    content_sha256: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    domain: str
    text: str
    token_count: int
    content_sha256: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int


@dataclass
class Usage:
    input_tokens: int = 0
    cache_write_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    names: list[str] = field(default_factory=list)

    def add(self, other: Usage, name: str | None = None) -> None:
        self.input_tokens += other.input_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls
        self.names.extend(other.names)
        if name:
            self.names.append(name)


@dataclass(frozen=True)
class SearchResult:
    query: str
    hits: list[Hit]
    embedding_tokens: int
    elapsed_ms: float
    cache_scope: str = "default"
    cache_hit: bool = False
    query_vector_ms: float = 0.0
    ann_ms: float = 0.0
