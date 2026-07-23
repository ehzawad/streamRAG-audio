from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Path, status
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from shared.agent.service import GroundedAgent
from shared.api.runtime import (
    IndexMaintenanceError,
    RagRuntime,
    TurnClosedError,
    TurnConflictError,
)
from shared.api.schemas import CommitAccepted, CommitRequest, SnapshotAccepted, SnapshotRequest
from shared.config import Settings
from shared.data.crag import (
    VerifiedDatasetSnapshot,
    capture_dataset_snapshot,
    chunk_documents,
    dataset_review_status,
    deduplicate_documents,
    load_snapshot_documents,
    require_dataset_snapshot,
)
from shared.data.index_state import IndexStateRepository
from shared.data.session_store import SessionStore
from shared.data.vector_store import IndexNotReadyError, QdrantVectorStore
from shared.fingerprints import (
    INDEX_PIPELINE_VERSION,
    config_payload,
    dataset_fingerprints,
    index_source_sha256,
    index_source_sha256_for_documents,
)
from shared.metrics import JsonlMetricLogger
from shared.path import METRICS_CONTRACT_VERSION, RagPath
from shared.rag.embeddings import OpenAIEmbedder

TurnId = Annotated[
    str,
    Path(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
SettingsProvider = Callable[[], Settings]
PathFactory = Callable[[Settings, QdrantVectorStore], RagPath]


def _event_cursor(last_event_id: str | None) -> int:
    if last_event_id is None:
        return 0
    try:
        cursor = int(last_event_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last-Event-ID must be a non-negative integer",
        ) from exc
    if cursor < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last-Event-ID must be a non-negative integer",
        )
    return cursor


def _dataset_health_state(settings: Settings) -> tuple[str, bool, str]:
    try:
        snapshot = capture_dataset_snapshot(settings.dataset_dir)
    except OSError, RuntimeError, ValueError:
        try:
            current_index_source = index_source_sha256(settings)
        except OSError, RuntimeError:
            current_index_source = "unavailable"
        try:
            approval = dataset_review_status(settings.dataset_dir)
        except OSError, ValueError:
            approval = "unknown"
        return current_index_source, False, approval
    return (
        index_source_sha256_for_documents(settings, snapshot.documents_sha256),
        True,
        snapshot.approval_status,
    )


def _dataset_status_state(
    settings: Settings,
    startup_fingerprints: dict[str, str],
) -> dict:
    try:
        snapshot = capture_dataset_snapshot(settings.dataset_dir)
        verified_files = snapshot.checksums()
        checksums_valid = True
        checksum_error = None
        approval_status = snapshot.approval_status
        current_index_source = index_source_sha256_for_documents(
            settings,
            snapshot.documents_sha256,
        )
        fingerprints = dataset_fingerprints(settings, snapshot)
    except (OSError, RuntimeError, ValueError) as exc:
        verified_files = {}
        checksums_valid = False
        checksum_error = str(exc)
        try:
            approval_status = dataset_review_status(settings.dataset_dir)
        except OSError, ValueError:
            approval_status = "unknown"
        try:
            current_index_source = index_source_sha256(settings)
        except OSError, RuntimeError:
            current_index_source = "unavailable"
        fingerprints = {
            "serving_dataset_checksum": "unavailable",
            "documents_sha256": "unavailable",
        }
    return {
        "approval_status": approval_status,
        "checksums_valid": checksums_valid,
        "checksum_error": checksum_error,
        "verified_files": verified_files,
        "current_index_source": current_index_source,
        "fingerprints": {
            "config_hash": startup_fingerprints["config_hash"],
            **fingerprints,
        },
    }


def _load_chunks(snapshot: VerifiedDatasetSnapshot, settings: Settings):
    documents = deduplicate_documents(load_snapshot_documents(snapshot))
    return chunk_documents(documents, settings.chunk_tokens, settings.chunk_overlap)


def create_app(
    *,
    implementation: str,
    api_title: str,
    settings_provider: SettingsProvider,
    path_factory: PathFactory,
    supports_snapshots: bool,
) -> FastAPI:
    identifier = implementation.replace("-", "").replace("_", "")
    if not identifier.isalnum():
        raise ValueError("implementation must be a non-empty identifier")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = settings_provider()
        settings.validate()
        sessions = SessionStore(settings.runtime_db)
        await sessions.setup()
        store = QdrantVectorStore(
            settings,
            OpenAIEmbedder(
                settings.embedding_model,
                timeout_s=settings.openai_embedding_timeout_s,
                max_retries=settings.openai_embedding_max_retries,
                base_url=settings.embedding_base_url,
                api_key=settings.llm_api_key,
            ),
            IndexStateRepository(settings.runtime_db),
        )
        await store.setup()
        path = path_factory(settings, store)
        if path.name != implementation or path.supports_snapshots is not supports_snapshots:
            await path.close()
            await store.close()
            raise RuntimeError("service entrypoint and path implementation disagree")
        runtime = RagRuntime(
            settings=settings,
            store=store,
            agent=GroundedAgent(settings, store, sessions),
            path=path,
            logger=JsonlMetricLogger(settings.metrics_log),
        )
        app.state.settings = settings
        app.state.runtime = runtime
        runtime.start_maintenance()
        yield
        await runtime.shutdown()

    initial_settings = settings_provider()
    app = FastAPI(
        title=api_title,
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(initial_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )

    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "service": f"{implementation}-rag-api",
            "implementation": implementation,
            "docs": "/docs",
            "health": "/v1/health",
        }

    @app.get("/v1/health")
    async def health() -> dict:
        runtime: RagRuntime = app.state.runtime
        settings: Settings = app.state.settings
        path_metadata = runtime.path.public_metadata()
        collection = await runtime.store.get_collection()
        index_metadata = await runtime.store.state.metadata(settings.qdrant_collection)
        current_index_source, checksums_valid, approval_status = await asyncio.to_thread(
            _dataset_health_state,
            settings,
        )
        approval_allowed = approval_status == "approved_frozen" or settings.allow_unreviewed_dataset
        indexed_chunks = int(collection.points_count or 0)
        desired_chunks = int(index_metadata["desired_chunks"] or 0)
        index_matches = (
            index_metadata["ready"] is True
            and index_metadata["index_source_sha256"] == current_index_source
            and indexed_chunks == desired_chunks
            and desired_chunks > 0
        )
        return {
            "ok": approval_allowed and checksums_valid and index_matches,
            "implementation": implementation,
            "metrics_contract_version": METRICS_CONTRACT_VERSION,
            "supports_snapshots": supports_snapshots,
            "index_ready": approval_allowed and checksums_valid and index_matches,
            "dataset_status": approval_status,
            "dataset_approval_allowed": approval_allowed,
            "collection": settings.qdrant_collection,
            "indexed_chunks": indexed_chunks,
            "indexed_desired_chunks": desired_chunks,
            "index_metadata_ready": index_metadata["ready"],
            "dataset_checksums_valid": checksums_valid,
            "index_matches_current_corpus": index_matches,
            "model": settings.openai_model,
            "embedding_model": settings.embedding_model,
            "instance_id": runtime.instance_id,
            **path_metadata,
        }

    @app.get("/v1/data/status")
    async def data_status() -> dict:
        runtime: RagRuntime = app.state.runtime
        settings: Settings = app.state.settings
        path_metadata = runtime.path.public_metadata()
        collection = await runtime.store.get_collection()
        index_metadata = await runtime.store.state.metadata(settings.qdrant_collection)
        dataset_state = await asyncio.to_thread(
            _dataset_status_state,
            settings,
            runtime.fingerprints,
        )
        indexed_chunks = int(collection.points_count or 0)
        desired_chunks = int(index_metadata["desired_chunks"] or 0)
        index_matches = (
            index_metadata["ready"] is True
            and index_metadata["index_source_sha256"] == dataset_state["current_index_source"]
            and desired_chunks > 0
            and indexed_chunks == desired_chunks
        )
        return {
            "implementation": implementation,
            "metrics_contract_version": METRICS_CONTRACT_VERSION,
            "supports_snapshots": supports_snapshots,
            "approval_status": dataset_state["approval_status"],
            "indexed_chunks": indexed_chunks,
            "index_version": index_metadata["version"],
            "index_checksum": index_metadata["index_checksum"],
            "indexed_desired_chunks": index_metadata["desired_chunks"],
            "index_metadata_ready": index_metadata["ready"],
            "dataset_checksums_valid": dataset_state["checksums_valid"],
            "dataset_checksum_error": dataset_state["checksum_error"],
            "dataset_verified_files": len(dataset_state["verified_files"]),
            "index_source_sha256": index_metadata["index_source_sha256"],
            "current_index_source_sha256": dataset_state["current_index_source"],
            "index_matches_current_corpus": index_matches,
            "instance_id": runtime.instance_id,
            "model": settings.openai_model,
            "embedding_model": settings.embedding_model,
            "index_pipeline_version": INDEX_PIPELINE_VERSION,
            "configuration": config_payload(settings),
            **dataset_state["fingerprints"],
            **path_metadata,
        }

    @app.post("/v1/data/sync")
    async def sync_data() -> dict:
        settings: Settings = app.state.settings
        try:
            snapshot = await asyncio.to_thread(
                require_dataset_snapshot,
                settings.dataset_dir,
                settings.allow_unreviewed_dataset,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        runtime: RagRuntime = app.state.runtime
        try:
            async with runtime.index_maintenance_guard():
                chunks = await asyncio.to_thread(_load_chunks, snapshot, settings)
                source = index_source_sha256_for_documents(
                    settings,
                    snapshot.documents_sha256,
                )
                report = await runtime.store.sync(chunks, index_source=source)
        except IndexMaintenanceError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return {**report.__dict__, "dataset_status": snapshot.approval_status}

    if supports_snapshots:

        @app.post(
            "/v1/turns/{turn_id}/snapshots",
            response_model=SnapshotAccepted,
            status_code=status.HTTP_202_ACCEPTED,
        )
        async def snapshot(turn_id: TurnId, payload: SnapshotRequest) -> SnapshotAccepted:
            try:
                await app.state.runtime.accept_snapshot(turn_id, payload)
            except (TurnClosedError, TurnConflictError) as exc:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
            except (IndexMaintenanceError, IndexNotReadyError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(exc),
                ) from exc
            return SnapshotAccepted(
                turn_id=turn_id,
                revision=payload.revision,
                events_url=f"/v1/turns/{turn_id}/events",
            )

        @app.get("/v1/turns/{turn_id}/events")
        async def turn_events(
            turn_id: TurnId,
            last_event_id: str | None = Header(default=None),
        ) -> EventSourceResponse:
            after = _event_cursor(last_event_id)
            channel = await app.state.runtime.events.existing(f"turn:{turn_id}")
            if channel is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown turn")
            return EventSourceResponse(channel.subscribe(after), ping=15)

    @app.delete("/v1/turns/{turn_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def cancel_turn(turn_id: TurnId) -> None:
        await app.state.runtime.cancel_turn(turn_id)

    @app.post(
        "/v1/turns/{turn_id}/commit",
        response_model=CommitAccepted,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def commit(turn_id: TurnId, payload: CommitRequest) -> CommitAccepted:
        try:
            run_id = await app.state.runtime.start_commit(turn_id, payload)
        except (TurnClosedError, TurnConflictError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except (IndexMaintenanceError, IndexNotReadyError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return CommitAccepted(
            run_id=run_id,
            turn_id=turn_id,
            path=implementation,
            events_url=f"/v1/runs/{run_id}/events",
        )

    @app.get("/v1/runs/{run_id}/events")
    async def run_events(
        run_id: str,
        last_event_id: str | None = Header(default=None),
    ) -> EventSourceResponse:
        after = _event_cursor(last_event_id)
        channel = await app.state.runtime.events.existing(f"run:{run_id}")
        if channel is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown run")
        return EventSourceResponse(channel.subscribe(after), ping=15)

    @app.get("/v1/ops/metrics")
    async def ops_metrics() -> dict:
        runtime: RagRuntime = app.state.runtime
        return {
            "implementation": implementation,
            **runtime.counters.__dict__,
            "active_runs": len(runtime.tasks),
            "active_typed_turns": len(runtime.turns),
        }

    @app.get("/v1/metrics/schema")
    async def metrics_schema() -> dict:
        runtime: RagRuntime = app.state.runtime
        return {
            "implementation": implementation,
            "metrics_contract_version": METRICS_CONTRACT_VERSION,
            **runtime.path.evaluation_metrics,
        }

    return app
