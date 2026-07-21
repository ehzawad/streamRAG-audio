"""Precompute stabilized ASR traces + endpoint transcripts for the 3-arm harness.

Runs in .venv-modular (faster-whisper); dumps data/audio_crag/asr_traces.json so
the RAG harness (py3.14 venv, no whisper) can replay without importing ASR. The
endpoint transcript is the SAME final ASR hypothesis for both the naive and
streaming arms (fairness: arms differ only in WHEN retrieval fires).

  CUDA_VISIBLE_DEVICES="" /mnt/sdb/arafat/ehz/hervoice/.venv-modular/bin/python \
      audio/build_traces.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.asr_partials import cumulative_asr_trace  # noqa: E402
from audio.stabilizer import stabilize_trace  # noqa: E402

CRAG = ROOT / "data" / "crag_eval"
OUT = ROOT / "data" / "audio_crag"


def load_queries() -> dict[str, dict]:
    q = {}
    for fn in ("test_queries.jsonl", "dev_queries.jsonl"):
        for line in (CRAG / fn).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                q[r["id"]] = r
    return q


def load_gold() -> dict[str, dict]:
    gold = {}
    for line in (CRAG / "test_gold.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            gold[r["id"]] = r
    return gold


def main() -> int:
    from faster_whisper import WhisperModel

    queries = load_queries()
    gold = load_gold()
    manifest = [json.loads(l) for l in (OUT / "manifest.jsonl").read_text().splitlines() if l.strip()]
    kept = [m for m in manifest if m["keep"]]
    model = WhisperModel("base.en", device="cpu", compute_type="int8")

    out = []
    for m in kept:
        qid = m["id"]
        raw = cumulative_asr_trace(model, str(OUT / m["wav"]), block_ms=500)
        stab, met = stabilize_trace(raw, agreement_n=2)
        endpoint = stab[-1].text if stab else ""
        q = queries[qid]
        g = gold.get(qid, {})
        out.append({
            "id": qid,
            "gold_query": q.get("query") or q.get("question"),
            "query_time": q.get("query_time", ""),
            "gold_answer": (g.get("answer") or q.get("answer") or ""),
            "alt_answers": (g.get("alt_answers") or q.get("alt_answers") or []),
            "question_type": q.get("question_type"),
            "dynamism": q.get("dynamism"),
            "endpoint_text": endpoint,
            "stabilized_trace": [
                {"offset_ms": s.planned_offset_ms, "text": s.text, "is_final": s.is_final}
                for s in stab
            ],
            "correction_metrics": met.as_dict(),
        })
        print(f"  {qid:22s} snaps={len(stab):2d} endpoint={endpoint[:60]!r}")

    (OUT / "asr_traces.json").write_text(json.dumps(out, indent=2))
    print(f"[traces] {len(out)} -> {OUT/'asr_traces.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
