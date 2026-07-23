from __future__ import annotations

from typing import Any

# Must be identical across both services for the A/B comparison to be fair.
SHARED_IDENTITY_FIELDS = (
    "model",
    "embedding_model",
    "indexed_chunks",
    "index_checksum",
    "index_source_sha256",
    "index_pipeline_version",
    "serving_dataset_checksum",
)


def service_role_issues(status: dict[str, Any], implementation: str) -> list[str]:
    if implementation not in {"naive", "stream"}:
        raise ValueError(f"unknown RAG implementation: {implementation}")
    issues: list[str] = []
    if status.get("implementation") != implementation:
        issues.append(f"advertises {status.get('implementation')!r}, expected {implementation!r}")
    expected_snapshots = implementation == "stream"
    if status.get("supports_snapshots") is not expected_snapshots:
        issues.append(f"supports_snapshots must be {expected_snapshots}")
    if int(status.get("indexed_chunks") or 0) <= 0:
        issues.append("Qdrant index is empty; sync data before benchmarking")
    if status.get("dataset_checksums_valid") is not True:
        issues.append("dataset checksum verification failed")
    if status.get("index_matches_current_corpus") is not True:
        issues.append("index is stale for the current corpus/config")
    return issues


def comparison_issues(naive: dict[str, Any], stream: dict[str, Any]) -> list[str]:
    issues = [f"naive: {issue}" for issue in service_role_issues(naive, "naive")]
    issues += [f"stream: {issue}" for issue in service_role_issues(stream, "stream")]
    mismatched = [
        field for field in SHARED_IDENTITY_FIELDS if naive.get(field) != stream.get(field)
    ]
    if mismatched:
        issues.append(f"services do not share the same corpus/model identity: {mismatched}")
    naive_id = str(naive.get("instance_id") or "")
    stream_id = str(stream.get("instance_id") or "")
    if not naive_id or not stream_id or naive_id == stream_id:
        issues.append("services must be two distinct processes (distinct instance_id)")
    return issues
