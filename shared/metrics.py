from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

# Fully-local pipeline: token usage is logged, but there is no provider bill, so
# no dollar-cost accounting is emitted (the hosted-price model was removed).
COMMON_EVALUATION_METRICS = (
    "timing.submit_to_first_token_ms",
    "timing.total_response_ms",
    "retrieval.calls",
    "retrieval.cache_hit",
    "retrieval.query_vector_ms",
    "retrieval.ann_ms",
    "usage.input_tokens",
    "usage.output_tokens",
    "persistence.status",
)


class JsonlMetricLogger:
    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()

    async def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
