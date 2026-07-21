from __future__ import annotations

import hashlib
import json

from shared.config import Settings
from shared.data.crag import (
    VerifiedDatasetSnapshot,
    capture_dataset_snapshot,
    resolve_documents_path,
    sha256_file,
)

INDEX_PIPELINE_VERSION = "typed-crag-dedup-chunk-payload-v2"

CONFIG_FINGERPRINT_FIELDS = (
    "openai_model",
    "embedding_model",
    "reasoning_effort",
    "summary_reasoning_effort",
    "openai_service_tier",
    "qdrant_collection",
    "embedding_dimensions",
    "chunk_tokens",
    "chunk_overlap",
    "retrieve_candidates",
    "top_k",
    "context_token_budget",
    "history_token_budget",
    "history_keep_turns",
    "retrieval_timeout_s",
    "answer_timeout_s",
    "summary_timeout_s",
    "post_answer_persistence_timeout_s",
)


def config_payload(settings: Settings) -> dict[str, object]:
    return {name: getattr(settings, name) for name in CONFIG_FINGERPRINT_FIELDS}


def config_sha256(settings: Settings) -> str:
    return hashlib.sha256(
        json.dumps(config_payload(settings), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def dataset_fingerprints(
    settings: Settings,
    snapshot: VerifiedDatasetSnapshot | None = None,
) -> dict[str, str]:
    snapshot = snapshot or capture_dataset_snapshot(settings.dataset_dir)
    return {
        "serving_dataset_checksum": snapshot.serving_dataset_checksum,
        "documents_sha256": snapshot.documents_sha256,
    }


def runtime_fingerprints(settings: Settings) -> dict[str, str]:
    snapshot = capture_dataset_snapshot(settings.dataset_dir)
    return {
        "config_hash": config_sha256(settings),
        **dataset_fingerprints(settings, snapshot),
    }


def index_source_sha256(settings: Settings) -> str:
    return index_source_sha256_for_documents(
        settings,
        sha256_file(resolve_documents_path(settings.dataset_dir)),
    )


def index_source_sha256_for_documents(settings: Settings, documents_sha256: str) -> str:
    payload = {
        "documents_sha256": documents_sha256,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap": settings.chunk_overlap,
        "qdrant_collection": settings.qdrant_collection,
        "index_pipeline_version": INDEX_PIPELINE_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
