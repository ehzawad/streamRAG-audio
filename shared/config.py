from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    dataset_dir: Path = Path(os.getenv("DATASET_DIR", ROOT / "data" / "crag_eval"))
    qdrant_path: Path = Path(os.getenv("QDRANT_PATH", ROOT / "data" / "qdrant"))
    runtime_db: Path = Path(os.getenv("RUNTIME_DB", ROOT / "data" / "runtime.sqlite3"))
    metrics_log: Path = Path(os.getenv("METRICS_LOG", ROOT / "var" / "requests.jsonl"))

    # Local-serving mode: fully on-box, no hosted API. When true the answer
    # model is an OpenAI-compatible local server (llama.cpp serving Qwen3.5-9B)
    # and embeddings come from a local embedder; the hosted-model locks in
    # validate() are relaxed. Set LOCAL_MODE=0 to restore the hosted baseline.
    local_mode: bool = env_bool("LOCAL_MODE", True)
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8400/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "local")
    # Qwen3.5 is a reasoning ("thinking") model; disabling thinking keeps the
    # answer inside the token budget and preserves clean tool-call transcripts.
    disable_thinking: bool = env_bool("DISABLE_THINKING", True)
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:8401/v1")

    openai_model: str = os.getenv("OPENAI_MODEL", "qwen3.5-9b-local")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "bge-large-en-v1.5")
    # Medium is the quality/latency balance for grounded answers. History
    # compression remains deliberately low effort. (Hosted-mode only; the local
    # chat model ignores reasoning-effort settings.)
    reasoning_effort: str = os.getenv("REASONING_EFFORT", "medium")
    summary_reasoning_effort: str = os.getenv("SUMMARY_REASONING_EFFORT", "low")
    openai_service_tier: str = os.getenv("OPENAI_SERVICE_TIER", "default")
    qdrant_url: str | None = os.getenv("QDRANT_URL") or None
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "crag_chunks")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
    openai_embedding_timeout_s: float = float(os.getenv("OPENAI_EMBEDDING_TIMEOUT_S", "45"))
    openai_embedding_max_retries: int = int(os.getenv("OPENAI_EMBEDDING_MAX_RETRIES", "0"))
    allow_unreviewed_dataset: bool = env_bool("ALLOW_UNREVIEWED_DATASET", False)

    chunk_tokens: int = int(os.getenv("CHUNK_TOKENS", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    retrieve_candidates: int = int(os.getenv("RETRIEVE_CANDIDATES", "8"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    context_token_budget: int = int(os.getenv("CONTEXT_TOKEN_BUDGET", "2600"))
    history_token_budget: int = int(os.getenv("HISTORY_TOKEN_BUDGET", "2200"))
    history_keep_turns: int = int(os.getenv("HISTORY_KEEP_TURNS", "4"))
    # Generation caps. Local reasoning ("thinking") models spend tokens on a
    # reasoning trace before the visible answer, so the answer cap must cover
    # both. Configurable; defaults raised from the original 600/320 (which were
    # sized for a non-thinking hosted model) so a local thinker can finish.
    answer_max_tokens: int = int(os.getenv("ANSWER_MAX_TOKENS", "2048"))
    summary_max_tokens: int = int(os.getenv("SUMMARY_MAX_TOKENS", "512"))

    retrieval_timeout_s: float = float(os.getenv("RETRIEVAL_TIMEOUT_S", "6.0"))
    answer_timeout_s: float = float(os.getenv("ANSWER_TIMEOUT_S", "30.0"))
    summary_timeout_s: float = float(os.getenv("SUMMARY_TIMEOUT_S", "8.0"))
    post_answer_persistence_timeout_s: float = float(
        os.getenv("POST_ANSWER_PERSISTENCE_TIMEOUT_S", "10.0")
    )
    turn_idle_timeout_s: float = float(os.getenv("TURN_IDLE_TIMEOUT_S", "120"))
    session_retention_hours: float = float(os.getenv("SESSION_RETENTION_HOURS", "24"))
    query_cache_size: int = int(os.getenv("QUERY_CACHE_SIZE", "512"))
    search_cache_size: int = int(os.getenv("SEARCH_CACHE_SIZE", "256"))
    search_cache_ttl_s: float = float(os.getenv("SEARCH_CACHE_TTL_S", "120"))
    cors_origins: tuple[str, ...] = tuple(
        item.strip()
        for item in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if item.strip()
    )

    # Standard API prices checked 2026-07-17. Returned usage is logged; explicit
    # unpriced counters identify interrupted calls whose final usage is unavailable.
    sol_input_per_million: float = 5.0
    sol_cache_write_per_million: float = 6.25
    sol_cached_input_per_million: float = 0.5
    sol_output_per_million: float = 30.0
    embedding_input_per_million: float = 0.13

    def validate(self) -> None:
        if self.local_mode:
            # Fully-local mode: the hosted-model identity locks do not apply.
            # Validate the local serving contract instead.
            if not self.llm_base_url:
                raise ValueError("LOCAL_MODE requires LLM_BASE_URL (local OpenAI-compatible LLM)")
            if not self.embedding_base_url:
                raise ValueError("LOCAL_MODE requires EMBEDDING_BASE_URL (local embedding server)")
            if self.embedding_dimensions <= 0:
                raise ValueError("EMBEDDING_DIMENSIONS must be positive")
            if self.answer_max_tokens <= 0 or self.summary_max_tokens <= 0:
                raise ValueError("generation token caps must be positive")
        else:
            if self.openai_model != "gpt-5.6-sol":
                raise ValueError("the locked benchmark model is 'gpt-5.6-sol'")
            if self.embedding_model != "text-embedding-3-large":
                raise ValueError("the locked benchmark embedding model is 'text-embedding-3-large'")
            if self.reasoning_effort != "medium":
                raise ValueError(
                    "the locked grounded-answer configuration requires reasoning effort 'medium'"
                )
            if self.summary_reasoning_effort != "low":
                raise ValueError("the locked summary role requires reasoning effort 'low'")
            if self.embedding_dimensions != 3072:
                raise ValueError(
                    "the locked text-embedding-3-large configuration requires 3072 dimensions"
                )
            if self.openai_service_tier != "default":
                raise ValueError(
                    "the locked benchmark configuration requires service tier 'default'"
                )
        if self.chunk_tokens <= 0:
            raise ValueError("chunk size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_tokens:
            raise ValueError("chunk overlap must be non-negative and smaller than chunk size")
        if not 1 <= self.top_k <= self.retrieve_candidates:
            raise ValueError("TOP_K must be between 1 and RETRIEVE_CANDIDATES")
        if self.openai_embedding_timeout_s <= 0:
            raise ValueError("OPENAI_EMBEDDING_TIMEOUT_S must be positive")
        if self.openai_embedding_max_retries != 0:
            raise ValueError("the locked benchmark disables embedding SDK retries")
        if (
            min(
                self.retrieval_timeout_s,
                self.answer_timeout_s,
                self.summary_timeout_s,
                self.post_answer_persistence_timeout_s,
                self.turn_idle_timeout_s,
                self.session_retention_hours,
            )
            <= 0
        ):
            raise ValueError("runtime deadlines and retention windows must be positive")
        if self.post_answer_persistence_timeout_s <= self.summary_timeout_s:
            raise ValueError("POST_ANSWER_PERSISTENCE_TIMEOUT_S must exceed SUMMARY_TIMEOUT_S")


settings = Settings()
settings.validate()
