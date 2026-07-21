"""The honesty gate: does streaming speculation survive noisy partial ASR?

Re-decodes each synthesized wav at 500 ms blocks and measures, per utterance:
  - raw_correction_rate: cumulative-ASR deltas that are NOT append_only (each one
    would fire StreamCoordinator._cancel_for_correction and kill speculation).
  - stab_correction_rate: same, AFTER the LocalAgreement-2 stabilizer.
  - emitted snapshots (stabilized) and endpoint WER vs the gold query.

If stab_correction_rate stays high, streaming collapses to endpoint-only and the
latency claim is NOT earned. We print the verdict and write runs/audio_quality.json;
audio/config.py::speedup_claimable stays False until this shows otherwise.

  CUDA_VISIBLE_DEVICES="" /mnt/sdb/arafat/ehz/hervoice/.venv-modular/bin/python \
      scoring/audio_quality.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.asr_partials import cumulative_asr_trace  # noqa: E402
from audio.stabilizer import stabilize_trace  # noqa: E402

CRAG = ROOT / "data" / "crag_eval"
OUT = ROOT / "data" / "audio_crag"
RUNS = ROOT / "runs"


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


def load_queries() -> dict[str, str]:
    q = {}
    for fn in ("test_queries.jsonl", "dev_queries.jsonl"):
        for line in (CRAG / fn).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                q[r["id"]] = r.get("query") or r.get("question")
    return q


def main() -> int:
    from faster_whisper import WhisperModel

    queries = load_queries()
    manifest = [json.loads(l) for l in (OUT / "manifest.jsonl").read_text().splitlines() if l.strip()]
    kept = [m for m in manifest if m["keep"]]
    print(f"[quality] measuring {len(kept)} kept utterances (block=500ms, agreement=2)")

    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    rows = []
    for m in kept:
        wav = str(OUT / m["wav"])
        raw = cumulative_asr_trace(model, wav, block_ms=500)
        stab, met = stabilize_trace(raw, agreement_n=2)
        final_hyp = stab[-1].text if stab else ""
        row = {
            "id": m["id"], "raw_snapshots": len(raw), **met.as_dict(),
            "endpoint_wer": round(wer(queries[m["id"]], final_hyp), 4),
        }
        rows.append(row)
        print(f"  {m['id']:22s} raw_snaps={row['raw_snapshots']:2d} "
              f"raw_corr={row['raw_correction_rate']:.2f} "
              f"stab_corr={row['stab_correction_rate']:.2f} "
              f"emitted={row['emitted_snapshots']:2d} ep_wer={row['endpoint_wer']:.3f}")

    n = len(rows)
    agg = {
        "n": n,
        "mean_raw_correction_rate": round(sum(r["raw_correction_rate"] for r in rows) / n, 4),
        "mean_stab_correction_rate": round(sum(r["stab_correction_rate"] for r in rows) / n, 4),
        "mean_raw_snapshots": round(sum(r["raw_snapshots"] for r in rows) / n, 2),
        "mean_emitted_snapshots": round(sum(r["emitted_snapshots"] for r in rows) / n, 2),
        "mean_endpoint_wer": round(sum(r["endpoint_wer"] for r in rows) / n, 4),
    }
    # honest verdict: speculation can plausibly survive only if the stabilized
    # correction rate is low enough that some pre-endpoint snapshot stays append-only.
    agg["speculation_can_survive"] = agg["mean_stab_correction_rate"] < 0.5
    agg["note"] = (
        "Stabilizer cuts correction rate from {raw} to {stab}; streaming speedup is "
        "only claimable if the 3-arm run then shows accepted_retrieval_lead_at_commit>0."
    ).format(raw=agg["mean_raw_correction_rate"], stab=agg["mean_stab_correction_rate"])

    RUNS.mkdir(exist_ok=True)
    (RUNS / "audio_quality.json").write_text(json.dumps({"aggregate": agg, "rows": rows}, indent=2))
    print("\n[quality] AGGREGATE:")
    for k, v in agg.items():
        print(f"    {k}: {v}")
    print(f"[quality] -> {RUNS/'audio_quality.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
