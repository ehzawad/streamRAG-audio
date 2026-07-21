from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import numpy as np
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from shared import fingerprints
from shared.config import Settings
from shared.data.index_state import IndexStateRepository
from shared.models import Chunk, Hit, SearchResult
from shared.rag.embeddings import Embedder

POINT_NAMESPACE = uuid.UUID("5ae4ade0-5d95-4f2c-85c3-822aa0af4753")


@dataclass(frozen=True)
class IndexSyncReport:
    collection: str
    desired_chunks: int
    embedded_chunks: int
    unchanged_chunks: int
    deleted_chunks: int
    embedding_tokens: int
    index_version: int
    index_checksum: str
    index_source_sha256: str
    elapsed_ms: float


@dataclass(frozen=True)
class _CachedSearchPayload:
    """Reusable search data without metrics belonging to the original call."""

    hits: tuple[Hit, ...]


class IndexNotReadyError(RuntimeError):
    """Index access was rejected because durable or physical state is not current."""


class QdrantVectorStore:
    """Incremental dense index with bounded query/result caches."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        state: IndexStateRepository,
        client: AsyncQdrantClient | QdrantClient | None = None,
    ):
        self.settings = settings
        self.embedder = embedder
        self.state = state
        self._local_executor: ThreadPoolExecutor | None = None
        if client is not None:
            self.client = client
        elif settings.qdrant_url:
            self.client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=15,
            )
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(
                path=str(settings.qdrant_path),
                force_disable_check_same_thread=True,
            )
            self._local_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="embedded-qdrant",
            )
        self._query_cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._search_cache: OrderedDict[
            tuple[str, int, str, int], tuple[float, _CachedSearchPayload]
        ] = OrderedDict()
        self._cache_lock = asyncio.Lock()
        self._version_lock = asyncio.Lock()
        self._sync_lock = asyncio.Lock()
        self._index_version: int | None = None
        self._index_ready: bool | None = None

    async def _client_call(self, method: str, /, **kwargs: Any) -> Any:
        call = partial(getattr(self.client, method), **kwargs)
        if self._local_executor is not None:
            return await asyncio.get_running_loop().run_in_executor(
                self._local_executor,
                call,
            )
        result = call()
        return await result if inspect.isawaitable(result) else result

    async def get_collection(self) -> models.CollectionInfo:
        return await self._client_call(
            "get_collection",
            collection_name=self.settings.qdrant_collection,
        )

    async def setup(self) -> None:
        await self.state.setup()
        self._index_version = await self.state.version(self.settings.qdrant_collection)
        self._index_ready = await self.state.ready(self.settings.qdrant_collection)
        if not await self._client_call(
            "collection_exists",
            collection_name=self.settings.qdrant_collection,
        ):
            await self._client_call(
                "create_collection",
                collection_name=self.settings.qdrant_collection,
                vectors_config=models.VectorParams(
                    size=self.settings.embedding_dimensions,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                ),
                on_disk_payload=True,
            )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))

    def _index_hash(self, chunk: Chunk) -> str:
        value = {
            "chunk": asdict(chunk),
            "embedding_dimensions": self.settings.embedding_dimensions,
            "embedding_model": self.embedder.model,
            "index_pipeline_version": fingerprints.INDEX_PIPELINE_VERSION,
        }
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _desired_index_checksum(self, desired: dict[str, Chunk]) -> str:
        digest = hashlib.sha256()
        digest.update(self.settings.qdrant_collection.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(self.settings.embedding_dimensions).encode("ascii"))
        digest.update(b"\n")
        for chunk_id, chunk in sorted(desired.items()):
            digest.update(chunk_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(self._index_hash(chunk).encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()

    async def _existing(self) -> dict[str, dict[str, Any]]:
        existing: dict[str, dict[str, Any]] = {}
        offset = None
        while True:
            records, offset = await self._client_call(
                "scroll",
                collection_name=self.settings.qdrant_collection,
                limit=256,
                offset=offset,
                with_payload=["chunk_id", "index_hash"],
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                chunk_id = str(payload.get("chunk_id") or "")
                if chunk_id:
                    existing[chunk_id] = {
                        "point_id": str(record.id),
                        "index_hash": payload.get("index_hash"),
                    }
            if offset is None:
                return existing

    async def index_ready(self) -> bool:
        if self._index_ready is not None:
            return self._index_ready
        async with self._version_lock:
            if self._index_ready is None:
                self._index_ready = await self.state.ready(self.settings.qdrant_collection)
            return self._index_ready

    async def assert_ready(self, expected_index_source: str) -> None:
        """Fail closed unless durable metadata and physical points match the corpus."""
        metadata = await self.state.metadata(self.settings.qdrant_collection)
        collection = await self.get_collection()
        desired_chunks = int(metadata["desired_chunks"] or 0)
        points_count = int(collection.points_count or 0)
        async with self._version_lock:
            ready = (
                self._index_ready is True
                and metadata["ready"] is True
                and self._index_version == metadata["version"]
                and metadata["index_source_sha256"] == expected_index_source
                and desired_chunks > 0
                and points_count == desired_chunks
            )
        if not ready:
            raise IndexNotReadyError(
                "index is absent, incomplete, or stale for the current corpus; run data sync"
            )

    async def sync(
        self,
        chunks: list[Chunk],
        *,
        index_source: str,
        batch_size: int = 64,
    ) -> IndexSyncReport:
        if batch_size <= 0:
            raise ValueError("sync batch size must be positive")
        if not index_source:
            raise ValueError("index source fingerprint is required")
        async with self._sync_lock:
            return await self._sync(chunks, batch_size, index_source)

    async def _sync(
        self,
        chunks: list[Chunk],
        batch_size: int,
        index_source: str,
    ) -> IndexSyncReport:
        started = time.perf_counter()
        desired = {chunk.chunk_id: chunk for chunk in chunks}
        index_checksum = self._desired_index_checksum(desired)
        existing = await self._existing()
        changed = [
            chunk
            for chunk_id, chunk in desired.items()
            if chunk_id not in existing
            or existing[chunk_id]["index_hash"] != self._index_hash(chunk)
        ]
        removed = sorted(set(existing) - set(desired))
        embedding_tokens = 0
        async with self._version_lock:
            self._index_ready = False
        await self.state.mark_sync_started(self.settings.qdrant_collection)
        async with self._cache_lock:
            self._search_cache.clear()
        for start in range(0, len(changed), batch_size):
            batch = changed[start : start + batch_size]
            vectors, tokens = await self.embedder.embed(
                [f"{chunk.title}\n{chunk.text}" for chunk in batch]
            )
            embedding_tokens += tokens
            points = []
            for chunk, vector in zip(batch, vectors, strict=True):
                payload = asdict(chunk)
                payload["index_hash"] = self._index_hash(chunk)
                payload["embedding_model"] = self.embedder.model
                points.append(
                    models.PointStruct(
                        id=self._point_id(chunk.chunk_id),
                        vector=vector.tolist(),
                        payload=payload,
                    )
                )
            await self._client_call(
                "upsert",
                collection_name=self.settings.qdrant_collection,
                points=points,
                wait=True,
            )
        if removed:
            await self._client_call(
                "delete",
                collection_name=self.settings.qdrant_collection,
                points_selector=models.PointIdsList(
                    points=[existing[chunk_id]["point_id"] for chunk_id in removed]
                ),
                wait=True,
            )
        content_changed = bool(changed or removed)
        current_version = await self.state.record_sync(
            self.settings.qdrant_collection,
            index_checksum=index_checksum,
            index_source_sha256=index_source,
            desired_chunks=len(desired),
            content_changed=content_changed,
        )
        async with self._version_lock:
            self._index_version = current_version
            self._index_ready = True
        return IndexSyncReport(
            collection=self.settings.qdrant_collection,
            desired_chunks=len(desired),
            embedded_chunks=len(changed),
            unchanged_chunks=len(desired) - len(changed),
            deleted_chunks=len(removed),
            embedding_tokens=embedding_tokens,
            index_version=current_version,
            index_checksum=index_checksum,
            index_source_sha256=index_source,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(query.casefold().split())

    @staticmethod
    def _normalize_cache_scope(cache_scope: str) -> str:
        normalized = cache_scope.strip()
        if not normalized:
            raise ValueError("cache scope is empty")
        return normalized

    async def _query_vector(self, query: str, cache_scope: str) -> tuple[np.ndarray, int]:
        key = (
            cache_scope,
            f"{self.embedder.model}\0{self._normalize_query(query)}",
        )
        async with self._cache_lock:
            cached = self._query_cache.get(key)
            if cached is not None:
                self._query_cache.move_to_end(key)
                return cached.copy(), 0
        matrix, tokens = await self.embedder.embed([query])
        vector = matrix[0]
        async with self._cache_lock:
            self._query_cache[key] = vector.copy()
            self._query_cache.move_to_end(key)
            while len(self._query_cache) > self.settings.query_cache_size:
                self._query_cache.popitem(last=False)
        return vector, tokens

    async def _current_index_version(self) -> int:
        if self._index_version is not None:
            return self._index_version
        async with self._version_lock:
            if self._index_version is None:
                self._index_version = await self.state.version(self.settings.qdrant_collection)
            return self._index_version

    async def search(
        self,
        query: str,
        k: int | None = None,
        *,
        cache_scope: str = "default",
    ) -> SearchResult:
        started = time.perf_counter()
        if not await self.index_ready():
            raise IndexNotReadyError(
                f"index {self.settings.qdrant_collection!r} is not ready; run data sync"
            )
        query = query.strip()
        if not query:
            raise ValueError("retrieval query is empty")
        cache_scope = self._normalize_cache_scope(cache_scope)
        limit = k or self.settings.top_k
        version = await self._current_index_version()
        cache_key = (cache_scope, version, self._normalize_query(query), limit)
        now = time.monotonic()
        async with self._version_lock:
            if not self._index_ready or self._index_version != version:
                raise IndexNotReadyError("index changed while search was starting")
            async with self._cache_lock:
                cached = self._search_cache.get(cache_key)
                if cached and now - cached[0] <= self.settings.search_cache_ttl_s:
                    self._search_cache.move_to_end(cache_key)
                    return SearchResult(
                        query=query,
                        hits=list(cached[1].hits),
                        embedding_tokens=0,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        cache_scope=cache_scope,
                        cache_hit=True,
                    )
        query_vector_started = time.perf_counter()
        vector, tokens = await self._query_vector(query, cache_scope)
        query_vector_ms = (time.perf_counter() - query_vector_started) * 1000
        ann_started = time.perf_counter()
        response = await self._client_call(
            "query_points",
            collection_name=self.settings.qdrant_collection,
            query=vector.tolist(),
            limit=self.settings.retrieve_candidates,
            with_payload=True,
            with_vectors=False,
        )
        ann_ms = (time.perf_counter() - ann_started) * 1000
        hits: list[Hit] = []
        per_document: dict[str, int] = {}
        for point in response.points:
            payload = dict(point.payload or {})
            chunk = Chunk(
                chunk_id=str(payload["chunk_id"]),
                doc_id=str(payload["doc_id"]),
                title=str(payload["title"]),
                url=str(payload["url"]),
                domain=str(payload["domain"]),
                text=str(payload["text"]),
                token_count=int(payload["token_count"]),
                content_sha256=str(payload["content_sha256"]),
            )
            if per_document.get(chunk.doc_id, 0) >= 2:
                continue
            per_document[chunk.doc_id] = per_document.get(chunk.doc_id, 0) + 1
            hits.append(Hit(chunk=chunk, score=float(point.score), rank=len(hits) + 1))
            if len(hits) >= limit:
                break
        result = SearchResult(
            query=query,
            hits=hits,
            embedding_tokens=tokens,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            cache_scope=cache_scope,
            cache_hit=False,
            query_vector_ms=query_vector_ms,
            ann_ms=ann_ms,
        )
        async with self._version_lock:
            if not self._index_ready or self._index_version != version:
                raise IndexNotReadyError("index changed while search was in progress")
            async with self._cache_lock:
                self._search_cache[cache_key] = (
                    time.monotonic(),
                    _CachedSearchPayload(hits=tuple(hits)),
                )
                self._search_cache.move_to_end(cache_key)
                while len(self._search_cache) > self.settings.search_cache_size:
                    self._search_cache.popitem(last=False)
        return result

    async def close(self) -> None:
        await self._client_call("close")
        if self._local_executor is not None:
            executor = self._local_executor
            self._local_executor = None
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
