from __future__ import annotations

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from shared.config import Settings


def chat_model(
    settings: Settings,
    *,
    timeout_s: float,
) -> tuple[OpenAIChatModel, AsyncOpenAI]:
    """Build an OpenAI-compatible *Chat Completions* model.

    In LOCAL_MODE this targets a local llama.cpp server (Qwen3.5-9B) through
    OpenAI-compatible `/v1/chat/completions`, which supports streaming and tool
    calls. In hosted mode it targets OpenAI directly (chat completions, not the
    Responses API). Application-owned hard deadlines with ``max_retries=0`` keep
    every measured call to a single transport attempt.
    """
    if settings.local_mode:
        client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=timeout_s,
            max_retries=0,
        )
    else:
        client = AsyncOpenAI(timeout=timeout_s, max_retries=0)
    model = OpenAIChatModel(
        settings.openai_model,
        provider=OpenAIProvider(openai_client=client),
    )
    return model, client


def responses_model(
    settings: Settings,
    *,
    timeout_s: float,
) -> tuple[OpenAIResponsesModel, AsyncOpenAI]:
    """Hosted-only OpenAI Responses API model (retained for the frozen baseline)."""
    client = AsyncOpenAI(
        timeout=timeout_s,
        # Application-owned hard deadlines make hidden SDK retries impossible to
        # attribute precisely, so each measured call is one transport attempt.
        max_retries=0,
    )
    model = OpenAIResponsesModel(
        settings.openai_model,
        provider=OpenAIProvider(openai_client=client),
    )
    return model, client
