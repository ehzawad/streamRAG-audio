"""Audio-layer configuration for streamRAG-audio.

Deliberately standalone (not a StreamSettings subclass): the RAG core keeps its
own locked `stream.config.settings` (trigger effort 'low', 500 ms settling), and
this module only carries the concrete host paths + audio parameters the cascade
front-end needs. Everything here is a real path on this box, verified to exist —
no placeholders. The TTS/ASR stacks run in a SEPARATE venv (see
requirements-audio.txt); they are never co-resident with the answer LLM on the
A5000.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- host paths (A5000 box), verified present -------------------------------
LLAMA_SERVER = Path(
    os.getenv(
        "LLAMA_SERVER",
        "/mnt/sdb/arafat/ehz/llm-stuff/qwen35-gguf-bench/llama.cpp/build/bin/llama-server",
    )
)
MODELS_DIR = Path(
    os.getenv("MODELS_DIR", "/mnt/sdb/arafat/ehz/llm-stuff/qwen35-gguf-bench/models")
)
BRAIN_GGUF = Path(os.getenv("BRAIN_GGUF", str(MODELS_DIR / "q9b" / "Qwen3.5-9B-Q4_K_M.gguf")))
# bge-large is served as a GGUF embedding model on :8401 (streamrag-local convention).
EMBED_GGUF = Path(
    os.getenv(
        "EMBED_GGUF",
        "/mnt/sdb/arafat/ehz/llm/qwen3-lora-gguf-bench/models/embed/bge-large-en-v1.5-f16.gguf",
    )
)

# The TTS (Chatterbox) + ASR (faster-whisper) live here; both are CPU-runnable for
# ASR, GPU-sole-tenant for TTS synthesis. Kept out of the main py3.14 RAG venv.
AUDIO_VENV = Path(os.getenv("AUDIO_VENV", "/mnt/sdb/arafat/ehz/hervoice/.venv-modular"))

# --- A5000 pinning ----------------------------------------------------------
# Always the A5000 (GPU0 by PCI order); the A6000 co-tenant must be left alone.
CUDA_ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0"}


@dataclass(frozen=True)
class AudioConfig:
    # streaming ASR cadence — mirrors the paper's 500 ms block and StreamSettings.
    block_ms: int = int(os.getenv("BLOCK_MS", "500"))
    sample_rate: int = 16_000
    # LocalAgreement-N confirmed-prefix stabilizer (the ASR-churn firewall).
    agreement_n: int = int(os.getenv("AGREEMENT_N", "2"))
    # commit only after the final transcript is prefix-stable for this many extra
    # windows (mitigation R1: never let a post-endpoint ASR revision poison commit).
    commit_stable_windows: int = int(os.getenv("COMMIT_STABLE_WINDOWS", "1"))
    # realistic post-speech dwell before the endpoint fires: real VAD waits for
    # ~min_silence before declaring end-of-turn, and the coordinator needs its
    # settled_draft_delay (500 ms) to let speculation finish. With zero dwell,
    # speculation never lands before commit (measured: 0/12). This models the
    # silence gap; the streaming latency result is reported at BOTH 0 ms and this.
    endpoint_dwell_ms: int = int(os.getenv("ENDPOINT_DWELL_MS", "700"))
    # faster-whisper model + decode policy (CPU, int8, greedy, no cross-chunk prior).
    whisper_model: str = os.getenv("WHISPER_MODEL", "base.en")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute: str = os.getenv("WHISPER_COMPUTE", "int8")
    # honesty gate: the streaming latency claim is forbidden until audio_quality.py
    # shows stabilized correction rate lets speculation survive to commit.
    speedup_claimable: bool = False
    cuda_env: dict[str, str] = field(default_factory=lambda: dict(CUDA_ENV))


audio = AudioConfig()
