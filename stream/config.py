from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from shared.config import ROOT, Settings, env_bool


@dataclass(frozen=True)
class StreamSettings(Settings):
    trigger_reasoning_effort: str = os.getenv("TRIGGER_REASONING_EFFORT", "low")
    trigger_min_tokens: int = int(os.getenv("TRIGGER_MIN_TOKENS", "5"))
    trigger_min_new_tokens: int = int(os.getenv("TRIGGER_MIN_NEW_TOKENS", "3"))
    trigger_interval_ms: int = int(os.getenv("TRIGGER_INTERVAL_MS", "500"))
    trigger_max_presubmit_calls: int = int(os.getenv("TRIGGER_MAX_PRESUBMIT_CALLS", "4"))
    parallel_raw_retrieval: bool = env_bool("PARALLEL_RAW_RETRIEVAL", True)
    settled_draft_delay_ms: int = int(os.getenv("SETTLED_DRAFT_DELAY_MS", "500"))
    trigger_timeout_s: float = float(os.getenv("TRIGGER_TIMEOUT_S", "4.0"))

    def validate(self) -> None:
        super().validate()
        if self.trigger_reasoning_effort != "low":
            raise ValueError("the locked streaming trigger requires reasoning effort 'low'")
        if self.settled_draft_delay_ms != 500:
            raise ValueError("the locked StreamRAG configuration requires 500 ms draft settling")
        if self.trigger_timeout_s <= 0:
            raise ValueError("TRIGGER_TIMEOUT_S must be positive")

    def public_metadata(self) -> dict[str, object]:
        return {
            "trigger_reasoning_effort": self.trigger_reasoning_effort,
            "trigger_min_tokens": self.trigger_min_tokens,
            "trigger_min_new_tokens": self.trigger_min_new_tokens,
            "trigger_interval_ms": self.trigger_interval_ms,
            "trigger_max_presubmit_calls": self.trigger_max_presubmit_calls,
            "parallel_raw_retrieval": self.parallel_raw_retrieval,
            "settled_draft_delay_ms": self.settled_draft_delay_ms,
            "trigger_timeout_s": self.trigger_timeout_s,
        }


settings = StreamSettings(
    qdrant_path=Path(os.getenv("QDRANT_PATH", ROOT / "var" / "stream" / "qdrant")),
    runtime_db=Path(os.getenv("RUNTIME_DB", ROOT / "var" / "stream" / "runtime.sqlite3")),
    metrics_log=Path(os.getenv("METRICS_LOG", ROOT / "var" / "stream" / "requests.jsonl")),
)
settings.validate()
