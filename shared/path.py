from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from shared.models import InputSnapshot, SearchResult, Usage

EventSink = Callable[[dict], Awaitable[None]]
# v2: removed dollar-cost accounting (estimated_cost_usd) and renamed the
# usage-completeness fields off cost vocabulary in the metrics-log record.
METRICS_CONTRACT_VERSION = 2


@dataclass
class PathTelemetry:
    """Common result envelope consumed by the shared answer lifecycle."""

    result: SearchResult
    retrieval_started_ms: float
    retrieval_ready_ms: float
    controller_usage: Usage = field(default_factory=Usage)
    controller_calls: int = 0
    controller_timeouts: int = 0
    controller_failures: int = 0
    controller_elapsed_ms: float = 0.0
    retrieval_calls: int = 1
    raw_retrieval_calls: int = 0
    settled_draft_retrievals: int = 0
    retrieval_embedding_tokens: int = 0
    retrieval_query_vector_ms: float = 0.0
    retrieval_ann_ms: float = 0.0
    retrieval_timeouts: int = 0
    retrieval_failures: int = 0
    stale_discards: int = 0
    trigger_cancellations: int = 0
    trigger_cancellations_without_usage: int = 0
    retrieval_cancellations: int = 0
    evidence_reuses: int = 0
    evidence_revalidations: int = 0
    commit_fallbacks: int = 0
    first_speculative_started_ms: float | None = None
    first_speculative_ready_ms: float | None = None
    accepted_revision: int | None = None
    accepted_from_fallback: bool | None = None
    accepted_ready_before_commit: bool | None = None
    accepted_speculative_completed_ms: float | None = None
    commit_gate_ms: float = 0.0
    reuse_mode: str = "committed_text_retrieval"


class PathTurn(Protocol):
    @property
    def committed(self) -> bool: ...

    @property
    def last_activity_ms(self) -> float: ...

    async def update(self, snapshot: InputSnapshot) -> None: ...

    def freeze_at_commit(
        self,
        snapshot: InputSnapshot,
        committed_ms: float | None = None,
    ) -> None: ...

    async def close(self) -> None: ...


class RagPath(Protocol):
    name: str
    supports_snapshots: bool
    evaluation_metrics: dict[str, tuple[str, ...]]

    def public_metadata(self) -> dict[str, object]: ...

    def open_turn(
        self,
        *,
        turn_id: str,
        session_id: str,
        send: EventSink,
        conversation_context: str,
    ) -> PathTurn: ...

    async def commit(
        self,
        *,
        snapshot: InputSnapshot,
        session_id: str,
        committed_ms: float,
        turn: PathTurn | None,
        conversation_context: str = "",
    ) -> PathTelemetry: ...

    async def close(self) -> None: ...


def cache_scope(session_id: str, implementation: str) -> str:
    """Keep implementation retrieval work causally isolated."""

    return f"{session_id}:{implementation}"
