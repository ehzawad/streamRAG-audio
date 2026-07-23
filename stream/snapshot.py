from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class SnapshotDelta:
    fingerprint: str
    common_prefix_chars: int
    word_count: int
    new_words: int
    append_only: bool


class SnapshotAnalyzer:
    """Compute deterministic append-versus-correction deltas between typed drafts.

    The hot ``analyze`` call runs on every draft update using the pure-Python
    implementation below.
    """

    backend = "python"

    def analyze(self, previous: str, current: str) -> SnapshotDelta:
        return self._analyze_python(previous, current)

    @staticmethod
    def _analyze_python(previous: str, current: str) -> SnapshotDelta:
        common = 0
        for before, after in zip(previous, current, strict=False):
            if before != after:
                break
            common += 1
        previous_words = len(previous.split())
        current_words = len(current.split())
        return SnapshotDelta(
            fingerprint=hashlib.blake2s(current.encode("utf-8"), digest_size=12).hexdigest(),
            common_prefix_chars=common,
            word_count=current_words,
            new_words=max(0, current_words - previous_words),
            append_only=common == len(previous),
        )
