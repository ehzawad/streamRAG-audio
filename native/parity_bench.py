"""Parity + microbenchmark for the native SnapshotAnalyzer backend.

Proves the Rust `analyze_delta` produces byte-identical `SnapshotDelta` values
to the pure-Python fallback, then times both over the same draft-pair corpus and
prints microseconds per call. Kept out of any `tests/` directory so it survives
the test-removal phase as reproducible evidence for the write-up and video.

Run:  uv run python native/parity_bench.py
Exits non-zero if the two backends ever disagree.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from stream.snapshot import SnapshotAnalyzer

ROOT = Path(__file__).resolve().parents[1]
DEV_QUERIES = ROOT / "data" / "crag_eval" / "dev_queries.jsonl"

try:
    from streamrag_snapshot import analyze_delta as native_analyze_delta
except ImportError:
    native_analyze_delta = None


def _incremental_pairs(text: str) -> list[tuple[str, str]]:
    """Simulate append-only typing: growing prefixes of one query."""
    pairs: list[tuple[str, str]] = []
    previous = ""
    step = max(1, len(text) // 12)
    for end in range(step, len(text) + 1, step):
        current = text[:end]
        pairs.append((previous, current))
        previous = current
    if previous != text:
        pairs.append((previous, text))
    return pairs


def _correction_pairs(text: str) -> list[tuple[str, str]]:
    """Simulate mid-draft corrections that break the common prefix."""
    if len(text) < 6:
        return []
    mid = len(text) // 2
    edited = text[:mid] + "X" + text[mid + 1 :]
    return [(text, edited), (edited, text)]


def build_corpus() -> list[tuple[str, str]]:
    queries: list[str] = []
    if DEV_QUERIES.is_file():
        for line in DEV_QUERIES.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                query = str(row.get("query") or "").strip()
                if query:
                    queries.append(query)
    if not queries:
        queries = [
            "who is the current ceo of openai",
            "which dune movie has better music 1984 or 2021",
            "how long must a stock be held for long term capital gains",
        ]

    pairs: list[tuple[str, str]] = []
    for query in queries:
        pairs.extend(_incremental_pairs(query))
        pairs.extend(_correction_pairs(query))

    # Adversarial edge cases exercising the whitespace / Unicode contract.
    pairs.extend(
        [
            ("", ""),
            ("", "a"),
            ("hello", "hello"),
            ("hello world", "hello  world"),  # collapsed vs doubled space
            ("tab\tsep", "tab\tsep more"),
            ("nbsp word", "nbsp word two"),  # non-breaking space
            ("ideographic　space", "ideographic　space x"),  # U+3000
            ("unitsep", "unitsep y"),  # U+001F information separator
            ("emoji \U0001f680 rocket", "emoji \U0001f680 rocket ship"),  # 4-byte char
            ("café", "café latte"),  # combining/accented
            ("日本語 の テスト", "日本語 の テスト です"),  # CJK
            ("trailing space ", "trailing space  "),
            ("   ", "    "),  # only whitespace
        ]
    )
    return pairs


def to_tuple(delta) -> tuple[str, int, int, int, bool]:
    return (
        delta.fingerprint,
        delta.common_prefix_chars,
        delta.word_count,
        delta.new_words,
        delta.append_only,
    )


def check_parity(corpus: list[tuple[str, str]]) -> None:
    mismatches = 0
    for previous, current in corpus:
        py = to_tuple(SnapshotAnalyzer._analyze_python(previous, current))
        rs = native_analyze_delta(previous, current)
        if py != rs:
            mismatches += 1
            print(f"  MISMATCH prev={previous!r} cur={current!r}\n    python={py}\n    rust  ={rs}")
    if mismatches:
        print(f"PARITY FAILED: {mismatches}/{len(corpus)} pairs differ")
        sys.exit(1)
    print(f"PARITY OK: {len(corpus)} draft pairs produce identical SnapshotDelta")


def bench(corpus: list[tuple[str, str]], iterations: int = 200_000) -> None:
    n = len(corpus)

    def run_python() -> None:
        analyze = SnapshotAnalyzer._analyze_python
        for i in range(iterations):
            previous, current = corpus[i % n]
            analyze(previous, current)

    def run_rust() -> None:
        for i in range(iterations):
            previous, current = corpus[i % n]
            native_analyze_delta(previous, current)

    # Warm up so we time steady-state, not first-call setup.
    run_python()
    run_rust()

    start = time.perf_counter_ns()
    run_python()
    python_ns = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    run_rust()
    rust_ns = time.perf_counter_ns() - start

    py_us = python_ns / iterations / 1000
    rs_us = rust_ns / iterations / 1000
    speedup = python_ns / rust_ns if rust_ns else float("inf")
    print(f"\nMicrobenchmark ({iterations:,} calls over {n} draft pairs):")
    print(f"  python fallback: {py_us:8.3f} us/call")
    print(f"  rust  native   : {rs_us:8.3f} us/call")
    print(f"  speedup        : {speedup:8.2f}x")


def main() -> None:
    print(f"active SnapshotAnalyzer backend: {SnapshotAnalyzer.backend}")
    if native_analyze_delta is None:
        print("native streamrag_snapshot wheel not installed; run `make native` first.")
        sys.exit(1)
    corpus = build_corpus()
    check_parity(corpus)
    bench(corpus)


if __name__ == "__main__":
    main()
