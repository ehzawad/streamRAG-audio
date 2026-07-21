"""LocalAgreement-N confirmed-prefix stabilizer — the ASR-churn firewall (R2).

Raw cumulative ASR revises almost every block; fed straight in, every revision is
a non-`append_only` delta that fires `StreamCoordinator._cancel_for_correction()`
and collapses streaming (Arm 3) down to endpoint-only (Arm 2). LocalAgreement-N
only releases the word-prefix that has been stable across the last N decodes, so
downstream sees mostly append-only growth.

`append_only` here is computed EXACTLY as `stream/snapshot.py::SnapshotAnalyzer`
does — char-level `current.startswith(previous)` — so `stab_corrections` counts
precisely the deltas that would still trip the coordinator. That number is the
gate on the streaming-latency claim (audio_quality.py); it is measured, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

from audio.asr_partials import AsrSnapshot


@dataclass
class StabilizeMetrics:
    raw_transitions: int = 0
    raw_corrections: int = 0          # raw cumulative deltas that are NOT append_only
    stab_transitions: int = 0
    stab_corrections: int = 0         # stabilized emitted deltas that are still NOT append_only
    emitted: int = 0

    def as_dict(self) -> dict:
        def rate(n, d):
            return round(n / d, 4) if d else 0.0
        return {
            "raw_transitions": self.raw_transitions,
            "raw_correction_rate": rate(self.raw_corrections, self.raw_transitions),
            "stab_transitions": self.stab_transitions,
            "stab_correction_rate": rate(self.stab_corrections, self.stab_transitions),
            "emitted_snapshots": self.emitted,
        }


def _common_word_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        out.append(x)
    return out


def stabilize_trace(
    raw: list[AsrSnapshot], agreement_n: int = 2
) -> tuple[list[AsrSnapshot], StabilizeMetrics]:
    """Return (stabilized TypedSnapshot-compatible trace, metrics)."""
    m = StabilizeMetrics()

    # raw correction rate (what the coordinator would see with NO stabilizer)
    prev_raw = ""
    for s in raw:
        if prev_raw:
            m.raw_transitions += 1
            if not s.text.startswith(prev_raw):
                m.raw_corrections += 1
        prev_raw = s.text

    hist: list[list[str]] = []
    emitted_text = ""
    out: list[AsrSnapshot] = []
    for s in raw:
        words = s.text.split()
        if s.is_final:
            # endpoint transcript is authoritative — freeze it verbatim (mitigation R1)
            confirmed = s.text
        else:
            hist.append(words)
            if len(hist) > agreement_n:
                hist.pop(0)
            if len(hist) < agreement_n:
                continue
            cw = hist[0]
            for h in hist[1:]:
                cw = _common_word_prefix(cw, h)
            confirmed = " ".join(cw)
        if not confirmed or (confirmed == emitted_text and not s.is_final):
            continue
        # only release genuinely new confirmed content
        if confirmed == emitted_text:
            continue
        if emitted_text:
            m.stab_transitions += 1
            if not confirmed.startswith(emitted_text):
                m.stab_corrections += 1  # inside-prefix change still trips the coordinator
        out.append(AsrSnapshot(s.planned_offset_ms, confirmed, len(confirmed.split()), s.is_final))
        emitted_text = confirmed
        m.emitted += 1

    # guarantee a final snapshot exists (endpoint) even if dedup skipped it
    if not out or not out[-1].is_final:
        if raw:
            f = raw[-1]
            out.append(AsrSnapshot(f.planned_offset_ms, f.text, len(f.text.split()), True))
            m.emitted += 1
    return out, m
