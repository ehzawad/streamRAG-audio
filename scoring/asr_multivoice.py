"""ASR + WER over the Qwen3-TTS multi-voice set -> data/audio_crag_qwen/asr.jsonl.

For each (query, voice) 16 kHz clip: faster-whisper transcript (the endpoint text the
RAG pipeline would see) + WER vs the gold query. No filtering here — every item is kept
with its WER so the eval can report per-voice WER and truthfulness on the full crossing.

  CUDA_VISIBLE_DEVICES="" /mnt/sdb/arafat/ehz/hervoice/.venv-modular/bin/python \
      scoring/asr_multivoice.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "audio_crag_qwen"


def _norm(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = _norm(ref), _norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / len(r)


def main() -> int:
    from faster_whisper import WhisperModel

    manifest = [json.loads(x) for x in (OUT / "manifest.jsonl").read_text().splitlines() if x.strip()]
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    rows = []
    for m in manifest:
        segs, _ = model.transcribe(str(OUT / m["wav16"]), language="en", beam_size=1,
                                   condition_on_previous_text=False, vad_filter=False, temperature=0.0)
        hyp = " ".join(s.text for s in segs).strip()
        w = round(wer(m["gold_query"], hyp), 4)
        rows.append({"id": m["id"], "voice": m["voice"], "gold_query": m["gold_query"],
                     "gold_answer": m["gold_answer"], "alt_answers": m["alt_answers"],
                     "endpoint_text": hyp, "wer": w})
    (OUT / "asr.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    by_voice = {}
    for r in rows:
        by_voice.setdefault(r["voice"], []).append(r["wer"])
    print(f"[asr] {len(rows)} clips; per-voice mean WER:")
    for v, ws in sorted(by_voice.items()):
        kept = sum(1 for x in ws if x <= 0.10)
        print(f"  {v:10s} mean_wer={sum(ws)/len(ws):.3f}  kept(<=0.10)={kept}/{len(ws)}")
    print(f"[asr] -> {OUT/'asr.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
