from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.config import Settings
from shared.models import Usage

COMMON_EVALUATION_METRICS = (
    "timing.submit_to_first_token_ms",
    "timing.total_response_ms",
    "retrieval.calls",
    "retrieval.cache_hit",
    "retrieval.query_vector_ms",
    "retrieval.ann_ms",
    "usage.input_tokens",
    "usage.output_tokens",
    "estimated_cost_usd.total",
    "estimated_cost_usd.accounting_complete",
    "persistence.status",
)


@dataclass(frozen=True)
class CostBreakdown:
    input_usd: float
    cache_write_usd: float
    cached_input_usd: float
    output_usd: float
    total_usd: float


def model_cost(usage: Usage, settings: Settings) -> CostBreakdown:
    uncached = max(
        0,
        usage.input_tokens - usage.cache_write_tokens - usage.cached_input_tokens,
    )
    input_usd = uncached * settings.sol_input_per_million / 1_000_000
    cache_write_usd = usage.cache_write_tokens * settings.sol_cache_write_per_million / 1_000_000
    cached_usd = usage.cached_input_tokens * settings.sol_cached_input_per_million / 1_000_000
    output_usd = usage.output_tokens * settings.sol_output_per_million / 1_000_000
    return CostBreakdown(
        input_usd=input_usd,
        cache_write_usd=cache_write_usd,
        cached_input_usd=cached_usd,
        output_usd=output_usd,
        total_usd=input_usd + cache_write_usd + cached_usd + output_usd,
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
