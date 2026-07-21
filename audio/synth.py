"""CRAG-TTS-local: synthesize spoken CRAG queries with Chatterbox, offline + seeded.

Paper-shaped construction (arXiv:2510.02044 §9): synthesize every question with a
frozen voice/seed, then filter with an INDEPENDENT ASR toward zero full-utterance
WER. We do not silently drop items — every item is kept in the manifest with its
measured WER and a `keep` flag, so the eval can report exactly what was excluded.

Runs in .venv-modular (Chatterbox 0.1.7 + faster-whisper 1.2.1, torch 2.6/cu124),
GPU sole-tenant on the A5000, NEVER co-resident with the answer LLM.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  /mnt/sdb/arafat/ehz/hervoice/.venv-modular/bin/python -m audio.synth
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch  # noqa: E402
import torchaudio  # noqa: E402
from chatterbox.tts import ChatterboxTTS  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CRAG = ROOT / "data" / "crag_eval"
OUT = ROOT / "data" / "audio_crag"
WAV = OUT / "wav"
SEED = 1234
# frozen voice params (moderate temperature for clean, ASR-friendly speech)
VOICE = dict(exaggeration=0.4, cfg_weight=0.5, temperature=0.6, min_p=0.05, repetition_penalty=1.2)


def _norm(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()


def wer(ref: str, hyp: str) -> float:
    """Word-level Levenshtein / len(ref) — dependency-free, matches jiwer on words."""
    r, h = _norm(ref), _norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / len(r)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_queries() -> list[dict]:
    items = []
    for split, fn in (("test", "test_queries.jsonl"), ("dev", "dev_queries.jsonl")):
        for line in (CRAG / fn).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            items.append({"id": r["id"], "split": split, "query": r.get("query") or r.get("question")})
    return items


def main() -> int:
    WAV.mkdir(parents=True, exist_ok=True)
    items = load_queries()
    print(f"[synth] {len(items)} CRAG queries -> {WAV}")

    torch.manual_seed(SEED)
    tts = ChatterboxTTS.from_pretrained(device="cuda")
    asr = WhisperModel("base.en", device="cpu", compute_type="int8")

    manifest = []
    for it in items:
        torch.manual_seed(SEED)  # per-item determinism
        wav = tts.generate(it["query"], audio_prompt_path=None, **VOICE)
        wav_path = WAV / f"{it['id']}.wav"
        torchaudio.save(str(wav_path), wav, tts.sr)
        raw = wav_path.read_bytes()
        dur = wav.shape[-1] / tts.sr

        segs, _ = asr.transcribe(str(wav_path), language="en")
        hyp = " ".join(s.text for s in segs).strip()
        w = wer(it["query"], hyp)
        keep = w <= 0.10
        manifest.append({
            "id": it["id"], "split": it["split"], "wav": f"wav/{it['id']}.wav",
            "wav_sha256": sha256(raw), "duration_s": round(dur, 3), "sr": tts.sr,
            "seed": SEED, "voice": VOICE, "query_sha256": sha256(it["query"].encode()),
            "asr_selfcheck_hyp": hyp, "asr_selfcheck_wer": round(w, 4), "keep": keep,
        })
        flag = "" if keep else "  <-- HIGH WER, flagged"
        print(f"  {it['id']:22s} {dur:5.1f}s  wer={w:.3f}{flag}")

    (OUT / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
    lines = [f"{m['wav_sha256']}  {m['wav']}" for m in manifest]
    (OUT / "checksums.sha256").write_text("\n".join(lines) + "\n")

    kept = sum(m["keep"] for m in manifest)
    mean_wer = sum(m["asr_selfcheck_wer"] for m in manifest) / len(manifest)
    print(f"[synth] kept {kept}/{len(manifest)} at WER<=0.10; mean self-check WER={mean_wer:.3f}")
    print(f"[synth] manifest -> {OUT/'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
