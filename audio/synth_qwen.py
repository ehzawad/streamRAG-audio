"""Multi-voice robustness set: synthesize the CRAG queries in Qwen3-TTS's 9 timbres.

An ADDITIVE robustness appendix (codex round 4: GO) — it does NOT replace CRAG-TTS-local
(Chatterbox) or the shipped numbers. Full-crossed: every one of the 12 kept queries is
rendered in ALL 9 CustomVoice timbres with a FIXED neutral style, so timbre is never
confounded with question. Frozen: seed, style, speaker set, source(24k)+resampled(16k)
hashes. Earned claim ceiling: "the retrieval advantage persisted across nine Qwen3-TTS
synthetic timbres" — NOT human robustness or demographic coverage.

Qwen3-TTS is Apache-2.0. The rendered audio is a CRAG derivative -> still CC BY-NC
(non-commercial), gitignored, regenerate locally.

Runs in .venv-qwen-audio (qwen-tts 0.1.1), GPU sole-tenant on the A5000:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  /mnt/sdb/arafat/ehz/hervoice/.venv-qwen-audio/bin/python audio/synth_qwen.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "data" / "audio_crag" / "asr_traces.json"   # the 12 kept queries + gold
OUT = ROOT / "data" / "audio_crag_qwen"
W24, W16 = OUT / "wav24", OUT / "wav16"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SEED = 1234
NEUTRAL = "Speak in a neutral, clear, natural tone at a normal pace."


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def resample_24_to_16(wav: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == 16000:
        return wav.astype("float32")
    n = int(round(len(wav) * 16000 / sr_in))
    x = np.linspace(0, len(wav), n, endpoint=False)
    return np.interp(x, np.arange(len(wav)), wav).astype("float32")


def main() -> int:
    from qwen_tts import Qwen3TTSModel

    W24.mkdir(parents=True, exist_ok=True)
    W16.mkdir(parents=True, exist_ok=True)
    queries = json.loads(TRACES.read_text())
    print(f"[qwen-tts] {len(queries)} queries; loading {MODEL_ID} ...")
    model = Qwen3TTSModel.from_pretrained(MODEL_ID)
    speakers = model.get_supported_speakers()
    langs = model.get_supported_languages()
    english = next((c for c in langs if str(c).lower().startswith("en")), langs[0])
    print(f"[qwen-tts] {len(speakers)} speakers: {speakers}")
    print(f"[qwen-tts] language={english}")

    manifest = []
    for q in queries:
        for spk in speakers:
            torch.manual_seed(SEED)
            wavs, sr = model.generate_custom_voice(
                q["gold_query"], language=english, speaker=spk, instruct=NEUTRAL,
            )
            wav = np.asarray(wavs[0], dtype="float32")
            base = f"{q['id']}__{spk}"
            p24 = W24 / f"{base}.wav"
            sf.write(str(p24), wav, sr)
            wav16 = resample_24_to_16(wav, sr)
            p16 = W16 / f"{base}.wav"
            sf.write(str(p16), wav16, 16000)
            manifest.append({
                "id": q["id"], "voice": spk, "base": base,
                "gold_query": q["gold_query"], "gold_answer": q["gold_answer"],
                "alt_answers": q["alt_answers"],
                "wav24": f"wav24/{base}.wav", "wav16": f"wav16/{base}.wav",
                "sr24": int(sr), "duration_s": round(len(wav) / sr, 3),
                "wav24_sha256": sha(p24.read_bytes()), "wav16_sha256": sha(p16.read_bytes()),
                "seed": SEED, "style": NEUTRAL,
            })
        print(f"  {q['id']:20s} rendered in {len(speakers)} voices")

    (OUT / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
    (OUT / "checksums.sha256").write_text(
        "\n".join(f"{m['wav16_sha256']}  {m['wav16']}" for m in manifest) + "\n")
    print(f"[qwen-tts] {len(manifest)} clips ({len(queries)}x{len(speakers)}) -> {OUT/'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
