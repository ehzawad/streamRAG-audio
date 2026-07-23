"""NEW HEADLINE eval on the Qwen3-TTS 9-voice CRAG set (replaces single-voice Chatterbox;
the Chatterbox result is kept as a documented prior baseline).

Full-crossed: every query in every one of the 9 timbres (108 clips = 12 question CLUSTERS x
9 voices). Two arms — closed-book (parametric baseline, :8400 direct) vs naive-RAG (:8001).
The streaming-latency arm is NOT re-run here and is NOT voice-independent (whether speculation
lands depends on voice-conditioned ASR timing); it was measured only on the prior Chatterbox
run (runs/three_arm.json). Reports BOTH all-108 and WER<=0.10 (sensitivity) with a
QUESTION-CLUSTERED paired-gap bootstrap (12 clusters, not n=108 iid) + per-question rows for
auditability. Earned ceiling: "advantage observed positive on every tested timbre (per the
reused automatic judge)" — consistency across tested synthetic voices, NOT unseen-voice/human
robustness or a retrieval-only causal effect. Reuses run_three_arm isolation + arms + scorer.

  PYTHONPATH=. /mnt/sdb/arafat/ehz/llm/streamRAG/.venv/bin/python harness/run_multivoice.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comparison.crag_task1_eval import judge  # noqa: E402
from harness.run_three_arm import (  # noqa: E402
    RUNS,
    arm_closed_book,
    arm_naive,
    port_bound,
    start_service,
    sync_corpus,
    truthfulness,
)

ASR = ROOT / "data" / "audio_crag_qwen" / "asr.jsonl"
NAIVE = "http://127.0.0.1:8001"


def main() -> int:
    rows_in = [json.loads(x) for x in ASR.read_text().splitlines() if x.strip()]
    # full crossing; evaluate every clip (report per-voice WER + truthfulness on all)
    for p in (8400, 8401):
        if not port_bound(p):
            raise RuntimeError(f"llama-server :{p} not up — run harness/serve_local.sh")

    run_tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    naive_state = RUNS / "services" / f"mv-{run_tag}" / "naive"
    naive_p = start_service("naive", "naive.api:app", 8001, naive_state)
    rows = []
    try:
        sync_corpus(NAIVE)
        with httpx.Client(timeout=180) as client, \
             httpx.Client(base_url="http://127.0.0.1:8400/v1", timeout=120) as jc:
            for r in rows_in:
                cb = arm_closed_book(jc, r["endpoint_text"])
                nv = arm_naive(client, r["endpoint_text"], "")
                labels = {
                    "closed_book": judge(jc, r["gold_query"], r["gold_answer"], r["alt_answers"], cb["answer"]),
                    "naive": judge(jc, r["gold_query"], r["gold_answer"], r["alt_answers"], nv["answer"]),
                }
                rows.append({"id": r["id"], "voice": r["voice"], "wer": r["wer"], "labels": labels})
                print(f"  {r['voice']:10s} {r['id']:20s} wer={r['wer']:.2f} "
                      f"cb={labels['closed_book'][:4]} nv={labels['naive'][:4]}")
    finally:
        naive_p.terminate()
        try:
            naive_p.wait(timeout=10)
        except Exception:
            naive_p.kill()

    voices = sorted({r["voice"] for r in rows})

    def score(lbl: str) -> int:
        return 1 if lbl == "correct" else (-1 if lbl == "incorrect" else 0)

    def _pct(sorted_vals, q):
        return round(sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))], 4)

    def qcluster_gap_ci(subset, iters=2000, seed=0):
        # macro paired gap = mean over QUESTION clusters of (mean voice-paired gap); bootstrap the
        # 12 question clusters as intact units (NOT 108 iid clips). Returns (point, [lo,hi], n_clusters).
        by_q = {}
        for r in subset:
            by_q.setdefault(r["id"], []).append(r)
        qids = list(by_q)

        def macro(sample):
            gs = [sum(score(x["labels"]["naive"]) - score(x["labels"]["closed_book"]) for x in by_q[q])
                  / len(by_q[q]) for q in sample]
            return sum(gs) / len(gs)

        rng = random.Random(seed)
        boots = sorted(macro([rng.choice(qids) for _ in qids]) for _ in range(iters))
        return round(macro(qids), 4), [_pct(boots, 0.025), _pct(boots, 0.975)], len(qids)

    def voice_gap_ci(vr, iters=2000, seed=0):
        # paired per-question gap for one voice; bootstrap the (<=12) questions.
        diffs = [score(r["labels"]["naive"]) - score(r["labels"]["closed_book"]) for r in vr]
        rng = random.Random(seed)
        boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(iters))
        return round(sum(diffs) / len(diffs), 4), [_pct(boots, 0.025), _pct(boots, 0.975)]

    def summarize(subset: list[dict]) -> dict:
        pv = {}
        for v in voices:
            vr = [r for r in subset if r["voice"] == v]
            if not vr:
                pv[v] = {"n": 0}
                continue
            gap, gap_ci = voice_gap_ci(vr)
            pv[v] = {
                "n": len(vr), "mean_wer": round(sum(r["wer"] for r in vr) / len(vr), 4),
                "closed_book_truth": truthfulness([r["labels"]["closed_book"] for r in vr])["truthfulness"],
                "naive_truth": truthfulness([r["labels"]["naive"] for r in vr])["truthfulness"],
                "paired_gap": gap, "paired_gap_ci95_qbootstrap": gap_ci,
            }
        present = [v for v in voices if pv[v].get("n")]
        mgap, mgap_ci, nq = qcluster_gap_ci(subset)
        macro = {
            "n_clips": len(subset), "n_question_clusters": nq, "voices_present": len(present),
            "closed_book_truth": round(sum(pv[v]["closed_book_truth"] for v in present) / len(present), 4),
            "naive_truth": round(sum(pv[v]["naive_truth"] for v in present) / len(present), 4),
            "macro_paired_gap": mgap, "macro_gap_ci95_qcluster_bootstrap": mgap_ci,
            "gap_positive_on": f"{sum(1 for v in present if pv[v]['paired_gap'] > 0)}/{len(present)}",
        }
        return {"macro": macro, "per_voice": pv}

    all_res = summarize(rows)
    kept_res = summarize([r for r in rows if r["wer"] <= 0.10])

    def all_voices_positive(res):
        m = res["macro"]
        return m["gap_positive_on"] == f"{m['voices_present']}/{m['voices_present']}"

    persisted = (all_res["macro"]["macro_paired_gap"] > 0 and all_voices_positive(all_res)
                 and kept_res["macro"]["macro_paired_gap"] > 0 and all_voices_positive(kept_res))
    claim = (
        "The retrieval-enabled system's advantage over the illustrative closed-book baseline was "
        "OBSERVED positive on every one of the nine TESTED Qwen3-TTS synthetic voices (per the reused "
        "automatic judge). This is consistency across these tested timbres over 12 question clusters "
        "— NOT n=108 independent samples, unseen-voice/human/demographic robustness, or a "
        "retrieval-only causal effect."
        if persisted else
        "Advantage did NOT stay positive on all tested voices — do not promote as a headline.")
    out = {
        "dataset": "Qwen3-TTS 9-voice CRAG set: 108 clips = 12 CRAG question CLUSTERS x 9 tested "
                   "timbres, fixed neutral style. NEW HEADLINE (replaces single-voice Chatterbox "
                   "CRAG-TTS-local, kept as prior baseline: runs/three_arm.json). CRAG derivative "
                   "(CC BY-NC). Truthfulness per the REUSED automatic judge (same :8400 model answers "
                   "AND judges). 12 INDEPENDENT question clusters, 108 repeated observations; CIs are "
                   "question-clustered bootstraps, NOT n=108 iid.",
        "voices": voices, "all_108": all_res, "wer_filtered": kept_res,
        "note_latency": "Streaming-latency findings are NOT re-run here and are NOT voice-independent: "
                        "the speculative-retrieval null (0/24, paired median +898.5ms) was measured "
                        "ONLY on the prior Chatterbox run (runs/three_arm.json); the prefill no-KV-reuse "
                        "null is a Qwen3.5/llama.cpp property (runs/prefill_warm.json).",
        "rows": [{"id": r["id"], "voice": r["voice"], "wer": r["wer"],
                  "closed_book": r["labels"]["closed_book"], "naive": r["labels"]["naive"]} for r in rows],
        "claim": claim,
    }
    (RUNS / "multivoice.json").write_text(json.dumps(out, indent=2))
    for label, res in (("ALL-108", all_res), ("WER<=0.10 (sensitivity)", kept_res)):
        m = res["macro"]
        print(f"\n=== {label} — paired retrieval gap (question-clustered) ===")
        for v in voices:
            pv = res["per_voice"][v]
            if not pv.get("n"):
                continue
            print(f"  {v:10s} n={pv['n']:2} wer={pv['mean_wer']:.3f} cb={pv['closed_book_truth']:+.3f} "
                  f"nv={pv['naive_truth']:+.3f} gap={pv['paired_gap']:+.3f} CI{pv['paired_gap_ci95_qbootstrap']}")
        print(f"  MACRO cb={m['closed_book_truth']:+.3f} nv={m['naive_truth']:+.3f} "
              f"gap={m['macro_paired_gap']:+.3f} CI{m['macro_gap_ci95_qcluster_bootstrap']} "
              f"(gap>0 on {m['gap_positive_on']}, {m['n_question_clusters']} clusters, {m['n_clips']} clips)")
    print(f"\nCLAIM: {claim}\n[multivoice] -> {RUNS/'multivoice.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
