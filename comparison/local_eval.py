"""Lightweight local error-gate: run the fully-local pipeline over the CRAG test
questions and report answer-match + citation quality. Decides whether the base
local generator warrants fine-tuning (P3), before investing in LoRA.

Scoring mirrors the offline scorer's ``normalize``/``_contains_phrase`` for answer
match; it is a proxy, not the full support+citation gate. Usage::

    python -m comparison.local_eval --base http://127.0.0.1:8001
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path

import httpx

from comparison.benchmark.score import _contains_phrase, normalize

CITE = re.compile(r"\[([A-Za-z0-9_.-]+::c\d+)\]")
ROOT = Path(__file__).resolve().parents[1]


def answer_matches(answer: str, gold: dict) -> bool:
    normalized = normalize(answer)
    candidates = [gold.get("answer", ""), *gold.get("alt_answers", [])]
    return any(_contains_phrase(normalized, normalize(c)) for c in candidates if c)


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def drive(base: str, question: str, query_time: str) -> str:
    turn_id = str(uuid.uuid4())
    session = "eval-" + uuid.uuid4().hex[:8]
    with httpx.Client(timeout=120) as client:
        started = client.post(
            f"{base}/v1/turns/{turn_id}/commit",
            json={"session_id": session, "revision": 1, "text": question, "query_time": query_time},
        )
        started.raise_for_status()
        run_id = started.json()["run_id"]
        parts: list[str] = []
        with client.stream("GET", f"{base}/v1/runs/{run_id}/events") as stream:
            for line in stream.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "answer.delta":
                    parts.append(event.get("text", ""))
                elif event.get("type") in ("agent.persisted", "run.completed"):
                    break
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--queries", default=str(ROOT / "data/crag_eval/test_queries.jsonl"))
    parser.add_argument("--gold", default=str(ROOT / "data/crag_eval/test_gold.jsonl"))
    args = parser.parse_args()

    queries = _read_jsonl(args.queries)
    gold = {row["id"]: row for row in _read_jsonl(args.gold)}
    total = matched = cited = placeholder = 0
    for query in queries:
        qid = query.get("id")
        question = query.get("question") or query.get("query")
        answer = drive(args.base, question, query.get("query_time", ""))
        is_match = answer_matches(answer, gold.get(qid, {}))
        citations = CITE.findall(answer)
        has_placeholder = any(c.startswith("doc-id::") for c in citations)
        total += 1
        matched += is_match
        cited += bool(citations)
        placeholder += has_placeholder
        print(f"[{qid}] match={'Y' if is_match else 'N'} cites={citations or '-'} :: {answer[:88]}")

    print(f"\n== ERROR-GATE (n={total}) ==")
    print(f"answer-match:       {matched}/{total} = {matched / total:.0%}")
    print(f"emitted a citation: {cited}/{total} = {cited / total:.0%}")
    print(f"placeholder cite:   {placeholder}/{total}  (literal 'doc-id::' = grounding failure)")


if __name__ == "__main__":
    main()
