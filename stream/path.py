from __future__ import annotations

import time

from shared.data.vector_store import QdrantVectorStore
from shared.models import InputSnapshot
from shared.path import EventSink, PathTelemetry, PathTurn, cache_scope
from stream.config import StreamSettings
from stream.coordinator import StreamCoordinator, StreamMetrics
from stream.evaluation import EVALUATION_METRICS
from stream.snapshot import SnapshotAnalyzer
from stream.trigger import ModelTrigger


def reuse_mode(metrics: StreamMetrics, committed_ms: float) -> str:
    accepted_completed_ms = metrics.accepted_retrieval_completed_ms
    if metrics.accepted_from_fallback is False and metrics.accepted_ready_before_commit:
        return "precommit_revalidated" if metrics.evidence_revalidations else "precommit_exact"
    if (
        metrics.accepted_from_fallback is False
        and accepted_completed_ms is not None
        and committed_ms > accepted_completed_ms
    ):
        return "presubmit_retrieval_revalidated_at_commit"
    if metrics.accepted_from_fallback is False:
        return "inflight_completed_postcommit"
    return "committed_text_retrieval"


class StreamRagPath:
    """Path B: correction-safe speculative retrieval over typed snapshots."""

    name = "stream"
    supports_snapshots = True
    evaluation_metrics = EVALUATION_METRICS

    def __init__(
        self,
        settings: StreamSettings,
        store: QdrantVectorStore,
        trigger: ModelTrigger,
    ):
        self.settings = settings
        self.store = store
        self.trigger = trigger

    def public_metadata(self) -> dict[str, object]:
        return self.settings.public_metadata()

    def open_turn(
        self,
        *,
        turn_id: str,
        session_id: str,
        send: EventSink,
        conversation_context: str,
    ) -> StreamCoordinator:
        return StreamCoordinator(
            turn_id=turn_id,
            index=self.store,
            trigger=self.trigger,
            settings=self.settings,
            send=send,
            conversation_context=conversation_context,
            analyzer=SnapshotAnalyzer(),
            cache_scope=cache_scope(session_id or turn_id, self.name),
        )

    async def commit(
        self,
        *,
        snapshot: InputSnapshot,
        session_id: str,
        committed_ms: float,
        turn: PathTurn | None,
        conversation_context: str = "",
    ) -> PathTelemetry:
        # StreamRAG captured conversation context pre-Send at open_turn (held in
        # the coordinator), so the commit-time copy is redundant here.
        del session_id, conversation_context
        if not isinstance(turn, StreamCoordinator):
            raise RuntimeError("StreamRAG commit requires its frozen typed turn")
        commit_gate_started_ms = time.perf_counter() * 1000
        evidence = await turn.commit(snapshot)
        commit_gate_ms = time.perf_counter() * 1000 - commit_gate_started_ms
        metrics = turn.metrics
        return PathTelemetry(
            result=evidence.result,
            retrieval_started_ms=(metrics.accepted_retrieval_started_ms or evidence.started_ms),
            retrieval_ready_ms=metrics.accepted_retrieval_ready_ms or evidence.completed_ms,
            controller_usage=metrics.trigger_usage,
            controller_calls=metrics.trigger_calls,
            controller_timeouts=metrics.trigger_timeouts,
            controller_failures=metrics.controller_failures,
            controller_elapsed_ms=metrics.trigger_elapsed_ms,
            retrieval_calls=metrics.retrieval_calls,
            raw_retrieval_calls=metrics.raw_retrieval_calls,
            settled_draft_retrievals=metrics.settled_draft_retrievals,
            retrieval_embedding_tokens=metrics.retrieval_embedding_tokens,
            retrieval_query_vector_ms=metrics.retrieval_query_vector_ms,
            retrieval_ann_ms=metrics.retrieval_ann_ms,
            retrieval_timeouts=metrics.retrieval_timeouts,
            retrieval_failures=metrics.retrieval_failures,
            stale_discards=metrics.stale_discards,
            trigger_cancellations=metrics.trigger_cancellations,
            trigger_cancellations_without_usage=metrics.trigger_cancellations_without_usage,
            retrieval_cancellations=metrics.retrieval_cancellations,
            evidence_reuses=metrics.evidence_reuses,
            evidence_revalidations=metrics.evidence_revalidations,
            commit_fallbacks=metrics.commit_fallbacks,
            first_speculative_started_ms=metrics.first_retrieval_started_ms,
            first_speculative_ready_ms=metrics.first_retrieval_ready_ms,
            accepted_revision=metrics.accepted_revision,
            accepted_from_fallback=metrics.accepted_from_fallback,
            accepted_ready_before_commit=metrics.accepted_ready_before_commit,
            accepted_speculative_completed_ms=metrics.accepted_retrieval_completed_ms,
            commit_gate_ms=commit_gate_ms,
            reuse_mode=reuse_mode(metrics, committed_ms),
        )

    async def close(self) -> None:
        await self.trigger.close()
