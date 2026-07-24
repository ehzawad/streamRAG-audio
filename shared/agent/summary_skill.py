from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic_ai import Agent, ModelMessage, UsageLimits
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModelSettings

from shared.agent.context import token_count
from shared.agent.openai_client import chat_model
from shared.config import Settings
from shared.data.session_store import SessionMemory
from shared.models import Usage


def message_text(message: ModelMessage) -> str:
    pieces: list[str] = []
    for part in message.parts:
        if isinstance(part, UserPromptPart):
            pieces.append(str(part.content))
        elif isinstance(part, TextPart):
            pieces.append(part.content)
    return " ".join(pieces)


def pydantic_usage(raw: object, name: str) -> Usage:
    return Usage(
        input_tokens=int(getattr(raw, "input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(raw, "cache_write_tokens", 0) or 0),
        cached_input_tokens=int(getattr(raw, "cache_read_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
        calls=int(getattr(raw, "requests", 0) or 0),
        names=[name],
    )


@dataclass(frozen=True)
class CompressionResult:
    memory: SessionMemory
    usage: Usage
    compressed: bool
    usage_complete: bool = True
    timeout_calls_without_usage: int = 0


class ConversationSummarySkill:
    """Explicit reusable skill invoked only when durable history exceeds budget."""

    def __init__(self, settings: Settings):
        model, self._client = chat_model(
            settings,
            timeout_s=settings.summary_timeout_s,
        )
        model_settings = OpenAIChatModelSettings(
            temperature=0.0,
            max_tokens=settings.summary_max_tokens,
            extra_body=(
                {"chat_template_kwargs": {"enable_thinking": False}}
                if settings.disable_thinking
                else {}
            ),
        )
        self.agent = Agent(
            model,
            name="conversation_summary_skill",
            instructions=(
                "Compress conversation memory into at most 220 words. Preserve user goals, "
                "constraints, decisions, unresolved questions, and named entities. Never preserve "
                "retrieved document passages, citations, or tool output. Return only the summary."
            ),
            model_settings=model_settings,
        )
        self.settings = settings

    async def close(self) -> None:
        await self._client.close()

    async def compact(self, memory: SessionMemory) -> CompressionResult:
        rendered = "\n".join(message_text(message) for message in memory.messages)
        if token_count(rendered) <= self.settings.history_token_budget:
            return CompressionResult(memory, Usage(), False)
        keep_count = self.settings.history_keep_turns * 2
        if len(memory.messages) <= keep_count:
            return CompressionResult(memory, Usage(), False)
        old_messages = memory.messages[:-keep_count]
        recent_messages = memory.messages[-keep_count:]
        transcript = "\n".join(
            f"{type(message).__name__}: {message_text(message)}"
            for message in old_messages
            if message_text(message)
        )
        prompt = f"Previous summary:\n{memory.summary or '(none)'}\n\nOlder turns:\n{transcript}"
        try:
            async with asyncio.timeout(self.settings.summary_timeout_s):
                result = await self.agent.run(
                    prompt,
                    usage_limits=UsageLimits(
                        request_limit=1, output_tokens_limit=self.settings.summary_max_tokens
                    ),
                )
        except TimeoutError:
            return CompressionResult(
                memory,
                Usage(),
                False,
                usage_complete=False,
                timeout_calls_without_usage=1,
            )
        compacted = SessionMemory(
            messages=recent_messages,
            summary=str(result.output).strip(),
            compression_calls=memory.compression_calls + 1,
        )
        return CompressionResult(
            memory=compacted,
            usage=pydantic_usage(result.usage, "summary_skill"),
            compressed=True,
        )


def append_conversation_turn(memory: SessionMemory, question: str, answer: str) -> None:
    # Only conversational text becomes durable memory. Retrieval/tool evidence is turn-local.
    memory.messages.extend(
        [
            ModelRequest(parts=[UserPromptPart(content=question)]),
            ModelResponse(parts=[TextPart(content=answer)]),
        ]
    )
