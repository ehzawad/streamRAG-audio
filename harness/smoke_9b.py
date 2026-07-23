"""One-clip end-to-end smoke test for the CLEAN local 9B pipeline.

Proves, on the single clip data/audio_crag_qwen/wav16/crag-text-test-002__aiden.wav
("Who stepped down as Apple's CEO in August 2011?" → "Steve Jobs"), that every stage
of the retained pipeline is wired and local — with NO dependency on any removed module
(no synth.py / stabilizer / asr_partials / audio_quality / run_three_arm / 27B / Nemotron).

It is a TWO-VENV smoke (the Makefile `smoke-9b` target chains them):

  asr  — run under the faster-whisper venv (.venv-modular). Transcribes the clip with
         base.en int8 (CPU) and asserts WER ≤ 0.10 vs the manifest gold. Writes the
         transcript + gold to a handoff JSON.
  rag  — run under the streamRAG py3.14 venv. Requires :8400 (qwen3.5-9b-local) and
         :8401 (bge-large-en-v1.5) already serving; starts an isolated stream service,
         and asserts: /v1/health advertises both local aliases; the service accepts a
         snapshot + a commit (the pre-Send snapshot fires the ModelTrigger, proving the
         trigger runs on local Chat Completions); retrieval returns ≥1 source; and the
         final grounded answer is nonempty and contains "Jobs".

Any failed assertion exits nonzero.

  # under the two venvs, from the repo root with PYTHONPATH=. :
  <fw-python>  harness/smoke_9b.py asr <handoff.json>
  <rag-python> harness/smoke_9b.py rag <handoff.json>
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLIP_ID = "crag-text-test-002"
CLIP_VOICE = "aiden"
AUDIO_DIR = ROOT / "data" / "audio_crag_qwen"
WAV16 = AUDIO_DIR / "wav16" / f"{CLIP_ID}__{CLIP_VOICE}.wav"
MANIFEST = AUDIO_DIR / "manifest.jsonl"
WER_GATE = 0.10
STREAM_PORT = 8002
STREAM_BASE = f"http://127.0.0.1:{STREAM_PORT}"
DEFAULT_HANDOFF = ROOT / "runs" / "smoke_9b_transcript.json"


def _gold() -> dict:
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] == CLIP_ID and row["voice"] == CLIP_VOICE:
            return row
    raise SystemExit(f"[smoke] gold not found for {CLIP_ID}__{CLIP_VOICE} in {MANIFEST}")


# --------------------------------------------------------------------------- asr
def run_asr(handoff: Path) -> int:
    """faster-whisper base.en transcribe + WER gate. Runs under .venv-modular."""
    from faster_whisper import WhisperModel  # local to keep the rag venv import-clean

    from scoring.asr_multivoice import wer  # reused scorer (no removed-module deps)

    if not WAV16.exists():
        raise SystemExit(f"[smoke] missing clip {WAV16}")
    gold = _gold()
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(
        str(WAV16), language="en", beam_size=1,
        condition_on_previous_text=False, vad_filter=False, temperature=0.0,
    )
    hyp = " ".join(s.text for s in segs).strip()
    w = round(wer(gold["gold_query"], hyp), 4)
    print(f"[smoke:asr] gold='{gold['gold_query']}'")
    print(f"[smoke:asr] hyp ='{hyp}'  wer={w:.4f}")
    if w > WER_GATE:
        raise SystemExit(f"[smoke:asr] FAIL: base.en WER {w:.4f} > gate {WER_GATE}")
    handoff.parent.mkdir(parents=True, exist_ok=True)
    handoff.write_text(json.dumps({
        "id": CLIP_ID, "voice": CLIP_VOICE, "endpoint_text": hyp,
        "gold_query": gold["gold_query"], "gold_answer": gold["gold_answer"], "wer": w,
    }))
    print(f"[smoke:asr] PASS (WER ≤ {WER_GATE})  -> {handoff}")
    return 0


# --------------------------------------------------------------------------- rag
def _collect_run(client, run_id: str) -> tuple[str, list[dict]]:
    """Stream a run's SSE events; return (answer_text, sources)."""
    answer, sources = "", []
    with client.stream("GET", f"{STREAM_BASE}/v1/runs/{run_id}/events") as s:
        for line in s.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            etype = ev.get("type")
            if etype == "answer.started" and ev.get("sources"):
                sources = ev["sources"]
            elif etype == "answer.delta":
                answer += ev.get("text", "")
            elif etype in ("answer.ready", "answer.completed"):
                if ev.get("answer"):
                    answer = ev["answer"]
                if ev.get("sources"):
                    sources = ev["sources"]
                if etype == "answer.completed":
                    break
            elif etype == "run.error":
                raise SystemExit(f"[smoke:rag] run error: {ev.get('error')}: {ev.get('message')}")
            elif etype == "run.completed":
                break
    return answer.strip(), sources


def run_rag(handoff: Path) -> int:
    """Health aliases + snapshot/commit + retrieval + grounded answer. Runs under the rag venv."""
    import httpx

    from harness.eval_common import port_bound, start_service, sync_corpus

    if not handoff.exists():
        raise SystemExit(f"[smoke:rag] missing ASR handoff {handoff} — run the `asr` step first")
    payload = json.loads(handoff.read_text())
    endpoint_text = payload["endpoint_text"]
    gold_answer = payload["gold_answer"]

    for port in (8400, 8401):
        if not port_bound(port):
            raise SystemExit(f"[smoke:rag] FAIL: llama-server :{port} not up — run harness/serve_local.sh")

    state_dir = ROOT / "runs" / "services" / f"smoke-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}" / "stream"
    proc = start_service("stream", "stream.api:app", STREAM_PORT, state_dir)
    try:
        with httpx.Client(timeout=180) as client:
            # 1. health advertises the two local aliases
            health = client.get(f"{STREAM_BASE}/v1/health", timeout=15).json()
            if health.get("model") != "qwen3.5-9b-local":
                raise SystemExit(f"[smoke:rag] FAIL: health model={health.get('model')!r} != 'qwen3.5-9b-local'")
            if health.get("embedding_model") != "bge-large-en-v1.5":
                raise SystemExit(
                    f"[smoke:rag] FAIL: health embedding_model={health.get('embedding_model')!r} != 'bge-large-en-v1.5'")
            print(f"[smoke:rag] health OK: model={health['model']} embedding_model={health['embedding_model']}")

            # 2. index the local corpus (asserts >0 chunks internally)
            sync_corpus(STREAM_BASE)

            # 3. accept a pre-Send snapshot (fires the local-Chat-Completions ModelTrigger),
            #    then commit — proving the stream service's snapshot+commit path is live.
            turn_id = "smoke-" + uuid.uuid4().hex[:8]
            session_id = "smoke-" + uuid.uuid4().hex[:8]
            words = endpoint_text.split()
            partial = " ".join(words[: max(1, len(words) // 2)])
            r = client.post(f"{STREAM_BASE}/v1/turns/{turn_id}/snapshots",
                            json={"session_id": session_id, "revision": 1, "text": partial})
            if r.status_code != 202:
                raise SystemExit(f"[smoke:rag] FAIL: snapshot rejected ({r.status_code}): {r.text}")
            r = client.post(f"{STREAM_BASE}/v1/turns/{turn_id}/snapshots",
                            json={"session_id": session_id, "revision": 2, "text": endpoint_text})
            if r.status_code != 202:
                raise SystemExit(f"[smoke:rag] FAIL: second snapshot rejected ({r.status_code}): {r.text}")
            print("[smoke:rag] snapshots accepted (trigger ran on local Chat Completions)")

            r = client.post(f"{STREAM_BASE}/v1/turns/{turn_id}/commit",
                            json={"session_id": session_id, "revision": 3, "text": endpoint_text, "query_time": ""})
            if r.status_code != 202:
                raise SystemExit(f"[smoke:rag] FAIL: commit rejected ({r.status_code}): {r.text}")
            run_id = r.json()["run_id"]

            # 4. collect the grounded answer + retrieval sources
            answer, sources = _collect_run(client, run_id)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    if len(sources) < 1:
        raise SystemExit(f"[smoke:rag] FAIL: retrieval returned {len(sources)} sources (want ≥1)")
    if not answer:
        raise SystemExit("[smoke:rag] FAIL: empty final answer")
    if "jobs" not in answer.casefold():
        raise SystemExit(f"[smoke:rag] FAIL: answer does not contain 'Jobs' (gold={gold_answer!r}): {answer!r}")
    print(f"[smoke:rag] sources={len(sources)}  answer={answer!r}")
    print("[smoke:rag] PASS (≥1 source; nonempty answer contains 'Jobs')")
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    handoff = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_HANDOFF
    if mode == "asr":
        return run_asr(handoff)
    if mode == "rag":
        return run_rag(handoff)
    raise SystemExit(f"usage: smoke_9b.py {{asr|rag}} [handoff.json]  (got {mode!r})")


if __name__ == "__main__":
    sys.exit(main())
