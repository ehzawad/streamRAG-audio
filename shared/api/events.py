from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator


class EventChannel:
    """Buffered SSE channel so fast backend events are not lost before subscribe."""

    def __init__(self):
        self.events: list[dict] = []
        self.closed = False
        self._condition = asyncio.Condition()

    async def publish(self, payload: dict) -> None:
        async with self._condition:
            event = {**payload, "sequence": len(self.events) + 1}
            self.events.append(event)
            self._condition.notify_all()

    async def close(self) -> None:
        async with self._condition:
            self.closed = True
            self._condition.notify_all()

    async def subscribe(self, after: int = 0) -> AsyncIterator[dict[str, str]]:
        cursor = max(0, after)
        while True:
            async with self._condition:
                while cursor >= len(self.events) and not self.closed:
                    await self._condition.wait()
                batch = self.events[cursor:]
                is_closed = self.closed
            for event in batch:
                cursor += 1
                yield {
                    "id": str(event["sequence"]),
                    "event": str(event.get("type", "message")),
                    "data": json.dumps(event, ensure_ascii=False),
                }
            if is_closed and cursor >= len(self.events):
                return


class EventRegistry:
    def __init__(self):
        self._channels: dict[str, EventChannel] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> EventChannel:
        async with self._lock:
            return self._channels.setdefault(key, EventChannel())

    async def existing(self, key: str) -> EventChannel | None:
        async with self._lock:
            return self._channels.get(key)

    async def close_and_remove(self, key: str) -> None:
        """Close a channel and release its buffered events from the registry."""
        async with self._lock:
            channel = self._channels.pop(key, None)
        if channel is not None:
            await channel.close()

    async def size(self) -> int:
        async with self._lock:
            return len(self._channels)
