from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite
from pydantic_ai import ModelMessage, ModelMessagesTypeAdapter


@dataclass
class SessionMemory:
    messages: list[ModelMessage] = field(default_factory=list)
    summary: str = ""
    compression_calls: int = 0


@dataclass
class _LockEntry:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class SessionStore:
    """Server-owned conversation state; the browser never supplies model history."""

    def __init__(self, path: Path):
        self.path = path
        self._locks: dict[str, _LockEntry] = {}
        self._locks_guard = asyncio.Lock()

    async def setup(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                "CREATE TABLE IF NOT EXISTS sessions ("
                "session_key TEXT PRIMARY KEY, messages_json BLOB NOT NULL, "
                "summary TEXT NOT NULL DEFAULT '', compression_calls INTEGER NOT NULL DEFAULT 0, "
                "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            await db.commit()

    @asynccontextmanager
    async def lease(self, session_key: str) -> AsyncIterator[None]:
        async with self._locks_guard:
            entry = self._locks.setdefault(session_key, _LockEntry())
            entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            async with self._locks_guard:
                entry.users -= 1
                if entry.users == 0 and self._locks.get(session_key) is entry:
                    self._locks.pop(session_key)

    async def load(self, session_key: str) -> SessionMemory:
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "SELECT messages_json, summary, compression_calls FROM sessions "
                "WHERE session_key = ?",
                (session_key,),
            )
            row = await cursor.fetchone()
        if row is None:
            return SessionMemory()
        messages = ModelMessagesTypeAdapter.validate_json(bytes(row[0]))
        return SessionMemory(
            messages=list(messages),
            summary=str(row[1]),
            compression_calls=int(row[2]),
        )

    async def save(self, session_key: str, memory: SessionMemory) -> None:
        payload = ModelMessagesTypeAdapter.dump_json(memory.messages)
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO sessions(session_key, messages_json, summary, compression_calls) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(session_key) DO UPDATE SET "
                "messages_json=excluded.messages_json, summary=excluded.summary, "
                "compression_calls=excluded.compression_calls, updated_at=CURRENT_TIMESTAMP",
                (session_key, payload, memory.summary, memory.compression_calls),
            )
            await db.commit()

    async def prune(self, retention_hours: float) -> int:
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "DELETE FROM sessions WHERE updated_at < datetime('now', ?)",
                (f"-{retention_hours:g} hours",),
            )
            await db.commit()
            return max(0, int(cursor.rowcount))
