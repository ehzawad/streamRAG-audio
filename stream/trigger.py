from __future__ import annotations

import time
from dataclasses import dataclass

from pydantic_ai import Agent, ToolOutput, UsageLimits
from pydantic_ai.models.openai import OpenAIChatModelSettings, OpenAIResponsesModelSettings

from shared.agent.openai_client import chat_model, responses_model
from shared.agent.service import thinking_extra_body
from shared.agent.summary_skill import pydantic_usage
from shared.models import TriggerDecision, Usage
from stream.config import StreamSettings


@dataclass(frozen=True)
class TriggerResult:
    decision: TriggerDecision
    usage: Usage
    elapsed_ms: float


TRIGGER_POLICY = """You control speculative retrieval for a typed streaming RAG system.
Input is a cumulative draft while the user is typing.

Choose exactly one action:
- wait: intent/entities are still too incomplete or ambiguous for a precise search.
- retrieve: enough stable intent exists; emit a short standalone factual retrieval query.
- keep_previous: the reusable candidate query still covers this draft.

Also set candidate_query_compatible=true whenever the reusable candidate can
retrieve answer-bearing evidence for the current draft, even if you can phrase a
cleaner standalone query. Set it false when there is no candidate or when a new
entity, relation, date, location, number, negation, or comparison makes it unsafe.

Precision rules:
- Optimize recall while typing: some discarded retrieval is acceptable. Wait only
  while the entity/object or requested property/relation is too ambiguous to target
  answer-bearing evidence.
- Never invent missing entities, dates, constraints, or user intent.
- Retrieve once the object/entity and requested property or relation are clear.
- A reusable candidate may be a raw completed prefix whose retrieval is already in
  flight. Prefer keep_previous when it targets the same answer-bearing evidence;
  rewrite it only when the added text materially changes or disambiguates retrieval.
- For comparisons, wait until every compared target needed for retrieval is present.
- Treat output-format or explanation instructions as non-retrieval-changing when
  the existing query already targets the same answer-bearing evidence.
- If the draft materially corrects the entity or intent, retrieve a corrected query.
"""


class ModelTrigger:
    def __init__(self, settings: StreamSettings):
        self.settings = settings
        # LOCAL_MODE targets llama.cpp, which speaks Chat Completions (with tool
        # calls) but NOT the Responses API — so the trigger must use chat_model()
        # locally, mirroring the answer agent. responses_model() is hosted-only.
        if settings.local_mode:
            model, self._client = chat_model(settings, timeout_s=settings.trigger_timeout_s)
            model_settings = OpenAIChatModelSettings(
                parallel_tool_calls=False,
                temperature=0.0,
                max_tokens=120,
                extra_body=thinking_extra_body(settings),
            )
        else:
            model, self._client = responses_model(settings, timeout_s=settings.trigger_timeout_s)
            model_settings = OpenAIResponsesModelSettings(
                openai_reasoning_effort=settings.trigger_reasoning_effort,
                openai_reasoning_mode="standard",
                openai_service_tier=settings.openai_service_tier,
                openai_store=False,
                openai_text_verbosity="low",
                max_tokens=120,
            )
        self.agent = Agent(
            model,
            name="streamrag_trigger",
            output_type=ToolOutput(TriggerDecision, strict=True),
            instructions=TRIGGER_POLICY,
            model_settings=model_settings,
        )

    async def close(self) -> None:
        await self._client.close()

    async def decide(
        self,
        *,
        draft: str,
        previous_query: str | None,
        conversation_context: str,
    ) -> TriggerResult:
        started = time.perf_counter()
        result = await self.agent.run(
            "Cumulative draft:\n"
            f"{draft}\n\nReusable candidate query: {previous_query or '(none)'}\n"
            f"Recent conversation (may resolve follow-up references):\n"
            f"{conversation_context or '(none)'}",
            usage_limits=UsageLimits(request_limit=1, output_tokens_limit=120),
        )
        decision = result.output
        return TriggerResult(
            decision=decision,
            usage=pydantic_usage(result.usage, "trigger"),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
