from __future__ import annotations

import bz2
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from shared.models import Chunk, SourceDocument

FORBIDDEN_DOCUMENT_KEYS = {
    "query",
    "answer",
    "alt_ans",
    "alt_answers",
    "gold",
    "split",
    "source_split",
}

FULL_EVALUATION_REQUIRED_FILES = {
    "dataset_summary.json",
    "dev_queries.jsonl",
    "leakage_audit.json",
    "selection_manifest.json",
    "test_gold.jsonl",
    "test_queries.jsonl",
}
DOCUMENT_FILENAMES = ("documents.jsonl", "documents.jsonl.bz2")
SHA256_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


@dataclass(frozen=True)
class VerifiedDatasetSnapshot:
    """Checksum-verified dataset bytes captured as one immutable sync input."""

    approval_status: str
    verified_files: tuple[tuple[str, str], ...]
    documents_filename: str
    documents_sha256: str
    documents_bytes: bytes
    serving_dataset_checksum: str

    def checksums(self) -> dict[str, str]:
        return dict(self.verified_files)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict]:
    opener = bz2.open if path.suffix == ".bz2" else Path.open
    with opener(path, mode="rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def resolve_documents_path(dataset_dir: Path) -> Path:
    matches = [dataset_dir / name for name in DOCUMENT_FILENAMES if (dataset_dir / name).is_file()]
    if len(matches) != 1:
        raise RuntimeError(
            "dataset must contain exactly one of documents.jsonl or documents.jsonl.bz2"
        )
    return matches[0]


def dataset_review_status(dataset_dir: Path) -> str:
    summary = json.loads((dataset_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    return str(summary.get("selection", {}).get("approval_status", "unknown"))


def capture_dataset_snapshot(dataset_dir: Path) -> VerifiedDatasetSnapshot:
    """Read and verify every bound file once, retaining the exact corpus bytes."""
    root = dataset_dir.resolve()
    manifest = root / "checksums.sha256"
    if not manifest.is_file():
        raise RuntimeError(f"dataset checksum manifest is missing: {manifest}")

    manifest_bytes = manifest.read_bytes()
    try:
        manifest_text = manifest_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("dataset checksum manifest is not valid UTF-8") from exc

    entries: dict[str, str] = {}
    captured: dict[str, bytes] = {}
    for line_number, raw_line in enumerate(manifest_text.splitlines(), 1):
        if not raw_line.strip():
            continue
        match = SHA256_LINE.fullmatch(raw_line)
        if match is None:
            raise RuntimeError(f"invalid checksum manifest line at checksums.sha256:{line_number}")
        expected, name = match.groups()
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name == "checksums.sha256":
            raise RuntimeError(f"unsafe checksum manifest path: {name}")
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"checksum path escapes dataset directory: {name}") from exc
        if name in entries:
            raise RuntimeError(f"duplicate checksum manifest entry: {name}")
        if not target.is_file():
            raise RuntimeError(f"checksummed dataset file is missing: {name}")
        content = target.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise RuntimeError(f"dataset checksum mismatch: {name}")
        entries[name] = actual
        if name in {"dataset_summary.json", *DOCUMENT_FILENAMES}:
            captured[name] = content

    serving_dataset_checksum = hashlib.sha256(manifest_bytes).hexdigest()
    required_files = FULL_EVALUATION_REQUIRED_FILES
    document_entries = sorted(set(entries) & set(DOCUMENT_FILENAMES))
    if len(document_entries) != 1:
        raise RuntimeError("checksum manifest must bind exactly one documents.jsonl representation")
    physical_documents = [name for name in DOCUMENT_FILENAMES if (root / name).is_file()]
    if physical_documents != document_entries:
        raise RuntimeError(
            "dataset files do not match the single corpus representation in the manifest"
        )
    missing = sorted(required_files - entries.keys())
    if missing:
        raise RuntimeError(f"checksum manifest omits required files: {missing}")
    try:
        summary = json.loads(captured["dataset_summary.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("dataset_summary.json is invalid JSON") from exc
    document_name = document_entries[0]
    return VerifiedDatasetSnapshot(
        approval_status=str(summary.get("selection", {}).get("approval_status", "unknown")),
        verified_files=tuple(sorted(entries.items())),
        documents_filename=document_name,
        documents_sha256=entries[document_name],
        documents_bytes=captured[document_name],
        serving_dataset_checksum=serving_dataset_checksum,
    )


def require_dataset_snapshot(
    dataset_dir: Path,
    allow_unreviewed: bool,
) -> VerifiedDatasetSnapshot:
    snapshot = capture_dataset_snapshot(dataset_dir)
    status = snapshot.approval_status
    if status != "approved_frozen" and not allow_unreviewed:
        raise RuntimeError(
            f"CRAG dataset is not human-approved/frozen. Review "
            f"{dataset_dir / 'REVIEW_SHEET.md'} first, or set "
            "ALLOW_UNREVIEWED_DATASET=1 for explicitly non-final local checks."
        )
    return snapshot


def _documents_from_rows(rows: Iterable[dict]) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    seen_ids: set[str] = set()
    for row in rows:
        forbidden = FORBIDDEN_DOCUMENT_KEYS & row.keys()
        if forbidden:
            raise ValueError(f"forbidden label/query keys in retrievable row: {sorted(forbidden)}")
        doc_id = str(row["doc_id"])
        if doc_id in seen_ids:
            raise ValueError(f"duplicate doc_id: {doc_id}")
        seen_ids.add(doc_id)
        text = str(row.get("text") or "").strip()
        expected = str(row.get("content_sha256") or "")
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if expected != actual:
            raise ValueError(f"content checksum mismatch for {doc_id}")
        documents.append(
            SourceDocument(
                doc_id=doc_id,
                title=str(row.get("title") or "Untitled"),
                url=str(row.get("url") or ""),
                text=text,
                snippet=str(row.get("snippet") or ""),
                domain=str(row.get("domain") or "unknown"),
                query_time=str(row.get("snapshot_query_time") or ""),
                content_sha256=expected,
            )
        )
    if not documents:
        raise ValueError("document corpus is empty")
    return documents


def load_snapshot_documents(snapshot: VerifiedDatasetSnapshot) -> list[SourceDocument]:
    """Parse only the corpus bytes that were verified in ``snapshot``."""
    content = snapshot.documents_bytes
    if snapshot.documents_filename.endswith(".bz2"):
        try:
            content = bz2.decompress(content)
        except OSError as exc:
            raise ValueError("verified document corpus is invalid bzip2 data") from exc
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("verified document corpus is not valid UTF-8") from exc

    def rows() -> Iterable[dict]:
        for line_number, line in enumerate(lines, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {snapshot.documents_filename}:{line_number}"
                ) from exc

    return _documents_from_rows(rows())


def deduplicate_documents(documents: Iterable[SourceDocument]) -> list[SourceDocument]:
    unique: dict[str, SourceDocument] = {}
    for document in documents:
        unique.setdefault(document.content_sha256, document)
    return sorted(unique.values(), key=lambda item: item.doc_id)


def chunk_documents(
    documents: Iterable[SourceDocument],
    chunk_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    if chunk_tokens <= 0:
        raise ValueError("chunk size must be positive")
    if not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("overlap must be non-negative and smaller than chunk size")
    encoding = tiktoken.get_encoding("cl100k_base")
    step = chunk_tokens - overlap_tokens
    chunks: list[Chunk] = []
    for document in documents:
        payload = f"{document.title}\n{document.text}".strip()
        token_ids = encoding.encode(payload)
        for offset in range(0, len(token_ids), step):
            piece_ids = token_ids[offset : offset + chunk_tokens]
            if not piece_ids:
                continue
            text = encoding.decode(piece_ids).strip()
            if not text:
                continue
            ordinal = offset // step
            chunk_id = f"{document.doc_id}::c{ordinal:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    title=document.title,
                    url=document.url,
                    domain=document.domain,
                    text=text,
                    token_count=len(piece_ids),
                    content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
            if offset + chunk_tokens >= len(token_ids):
                break
    if not chunks:
        raise ValueError("chunker produced no chunks")
    return chunks
