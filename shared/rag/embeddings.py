from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from openai import AsyncOpenAI


class Embedder(Protocol):
    model: str

    async def embed(self, texts: Sequence[str]) -> tuple[np.ndarray, int]: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


class OpenAIEmbedder:
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        client: AsyncOpenAI | None = None,
        *,
        timeout_s: float = 45.0,
        max_retries: int = 0,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        if client is not None:
            self.client = client
        elif base_url:
            # Local OpenAI-compatible embedding server (e.g. llama.cpp --embedding).
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key or "local",
                timeout=timeout_s,
                max_retries=max_retries,
            )
        else:
            self.client = AsyncOpenAI(timeout=timeout_s, max_retries=max_retries)

    async def embed(self, texts: Sequence[str]) -> tuple[np.ndarray, int]:
        if not texts:
            raise ValueError("cannot embed an empty batch")
        response = await self.client.embeddings.create(
            model=self.model,
            input=list(texts),
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        matrix = l2_normalize(np.asarray([item.embedding for item in ordered], dtype=np.float32))
        tokens = int(getattr(response.usage, "total_tokens", 0) or 0)
        return matrix, tokens
