"""Streaming ASR trace: cumulative faster-whisper hypotheses at 500 ms blocks.

faster-whisper is not natively streaming, so we emulate the online setting the
paper assumes by re-decoding the GROWING audio prefix at each 500 ms boundary.
Unlike typing (append-only prefixes), a fresh decode REVISES earlier words — the
exact behavior that stresses the coordinator's correction path. The final block
(full audio) is the endpoint transcript; nothing is posted after it (mitigation R1).

The emitted objects are TypedSnapshot-compatible (`planned_offset_ms`, `text`,
`word_count`, `is_final`) so the existing replay harness consumes them unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from faster_whisper.audio import decode_audio

SR = 16_000


@dataclass(frozen=True)
class AsrSnapshot:
    planned_offset_ms: float
    text: str
    word_count: int
    is_final: bool


def load_wav_16k(wav_path: str) -> np.ndarray:
    # faster-whisper's own loader: 16 kHz mono float32, deterministic (av/ffmpeg).
    return decode_audio(wav_path, sampling_rate=SR)


def _decode(model, chunk: np.ndarray) -> str:
    # greedy, no cross-chunk prior — reproducible and honest about revision behavior
    segs, _ = model.transcribe(
        chunk, language="en", beam_size=1, condition_on_previous_text=False,
        vad_filter=False, temperature=0.0,
    )
    return " ".join(s.text for s in segs).strip()


def cumulative_asr_trace(model, wav_path: str, block_ms: int = 500) -> list[AsrSnapshot]:
    """Re-decode audio[0:t] at t = block_ms, 2*block_ms, ... duration."""
    audio = load_wav_16k(wav_path)
    dur_ms = len(audio) / SR * 1000
    bounds = list(range(block_ms, int(dur_ms) + block_ms, block_ms))
    if not bounds or bounds[-1] < dur_ms:
        bounds.append(int(round(dur_ms)))
    snaps: list[AsrSnapshot] = []
    last = ""
    for i, t in enumerate(bounds):
        chunk = audio[: int(SR * t / 1000)]
        if len(chunk) < SR * 0.1:  # <100ms: nothing to decode yet
            continue
        text = _decode(model, chunk)
        is_final = i == len(bounds) - 1
        if not text and not is_final:
            continue
        if text == last and not is_final:
            continue
        snaps.append(AsrSnapshot(float(t), text, len(text.split()), is_final))
        last = text
    return snaps


def endpoint_transcript(model, wav_path: str) -> str:
    return _decode(model, load_wav_16k(wav_path))
