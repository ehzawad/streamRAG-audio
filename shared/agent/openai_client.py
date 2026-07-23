from __future__ import annotations

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from shared.config import Settings


def chat_model(
    settings: Settings,
    *,
    timeout_s: float,
) -> tuple[OpenAIChatModel, AsyncOpenAI]:
    """Build an OpenAI-compatible *Chat Completions* model.

    This targets the local llama.cpp server (Qwen3.5-9B) through
    OpenAI-compatible `/v1/chat/completions`, which supports streaming and tool
    calls. Application-owned hard deadlines with ``max_retries=0`` keep every
    measured call to a single transport attempt.
    """
    client = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=timeout_s,
        max_retries=0,
    )
    model = OpenAIChatModel(
        settings.openai_model,
        provider=OpenAIProvider(openai_client=client),
    )
    return model, client
