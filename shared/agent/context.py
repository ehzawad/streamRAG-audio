from __future__ import annotations

import json
from collections.abc import Iterable

import tiktoken

from shared.models import Hit

_ENCODING = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


def fit_hits_to_budget(hits: Iterable[Hit], token_budget: int) -> list[Hit]:
    selected: list[Hit] = []
    used = 0
    for hit in hits:
        required = hit.chunk.token_count + token_count(hit.chunk.title) + 24
        if selected and used + required > token_budget:
            break
        if required > token_budget:
            continue
        selected.append(hit)
        used += required
    return selected


def evidence_block(hits: Iterable[Hit], token_budget: int) -> str:
    selected = fit_hits_to_budget(hits, token_budget)
    if not selected:
        return "No retrieved evidence is available."
    return "\n\n".join(
        f"[{hit.chunk.chunk_id}]\n"
        f"Title: {hit.chunk.title}\n"
        f"URL: {hit.chunk.url}\n"
        f"Text: {hit.chunk.text}"
        for hit in selected
    )


def tool_result_json(hits: Iterable[Hit], token_budget: int) -> str:
    selected = fit_hits_to_budget(hits, token_budget)
    return json.dumps(
        {
            "results": [
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "title": hit.chunk.title,
                    "url": hit.chunk.url,
                    "text": hit.chunk.text,
                    "score": round(hit.score, 6),
                }
                for hit in selected
            ]
        },
        ensure_ascii=False,
    )
