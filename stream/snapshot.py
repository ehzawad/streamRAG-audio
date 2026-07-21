from __future__ import annotations

import hashlib
from dataclasses import dataclass

try:  # Prefer the native backend; fall back to pure Python if the wheel is absent.
    from streamrag_snapshot import analyze_delta as _native_analyze_delta

    _BACKEND = "rust"
except ImportError:  # pragma: no cover - exercised whenever the wheel is not installed
    _native_analyze_delta = None
    _BACKEND = "python"


@dataclass(frozen=True)
class SnapshotDelta:
    fingerprint: str
    common_prefix_chars: int
    word_count: int
    new_words: int
    append_only: bool


class SnapshotAnalyzer:
    """Compute deterministic append-versus-correction deltas between typed drafts.

    The hot ``analyze`` call runs on every draft update. When the native
    ``streamrag_snapshot`` wheel is installed it delegates to Rust; otherwise it
    uses the pure-Python implementation below. Both produce identical
    ``SnapshotDelta`` values, so ``coordinator``/``path`` need no changes.
    """

    backend = _BACKEND

    def analyze(self, previous: str, current: str) -> SnapshotDelta:
        if _native_analyze_delta is not None:
            fingerprint, common, word_count, new_words, append_only = _native_analyze_delta(
                previous, current
            )
            return SnapshotDelta(
                fingerprint=fingerprint,
                common_prefix_chars=common,
                word_count=word_count,
                new_words=new_words,
                append_only=append_only,
            )
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
