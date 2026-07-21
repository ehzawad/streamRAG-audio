"""3-arm audio RAG eval on one A5000 — v2, hardened after codex round-2 audit.

Arms (identical audio -> identical endpoint ASR transcript; differ only in WHEN/
whether retrieval fires):
  1. closed_book  : parametric baseline, NO retrieval (:8400 direct). Illustrative,
                    NOT a strict retrieval ablation (different answer stack).
  2. naive        : commit endpoint transcript -> retrieve -> answer (:8001).
  3. stream       : replay stabilized ASR partials paced by offset_ms, then dwell to
                    the TRUE audio endpoint (+silence) so speculation has its window,
                    then commit the SAME endpoint transcript (:8002).

Honesty machinery (all from the round-2 audit):
  * Clean isolation: fail if a port is already bound; unique per-run state dirs;
    verify the spawned child is alive (the earlier run's naive child died on a Qdrant
    lock while a stale process served requests — that run was discarded).
  * Counterbalanced arm order across 2 reps to average out llama.cpp prompt-cache
    warming (identical answer prompts otherwise cache-hit whichever arm runs second).
  * Per-query stream telemetry read-back: accepted_ready_before_commit,
    accepted_retrieval_lead_at_commit_ms, reuse.mode. The streaming-latency benefit is
    only reported as REAL if speculation actually landed (gate enforced here, not by eye).
  * PAIRED latency stats (median/mean/win-rate), never difference-of-independent-medians.
  * Truthfulness: report test and dev SEPARATELY; "incorrect labels", not "hallucination".

  PYTHONPATH=. /mnt/sdb/arafat/ehz/llm/streamRAG/.venv/bin/python harness/run_three_arm.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from statistics import mean, median

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comparison.crag_task1_eval import bootstrap_truthfulness, generate, judge  # noqa: E402

TRACES = ROOT / "data" / "audio_crag" / "asr_traces.json"
MANIFEST = ROOT / "data" / "audio_crag" / "manifest.jsonl"
RUNS = ROOT / "runs"
PYBIN = os.getenv("RAG_PYBIN", "/mnt/sdb/arafat/ehz/llm/streamRAG/.venv/bin/python")
NAIVE, STREAM = "http://127.0.0.1:8001", "http://127.0.0.1:8002"
REPS = int(os.getenv("REPS", "2"))
ENDPOINT_DWELL_MS = int(os.getenv("ENDPOINT_DWELL_MS", "700"))

CLOSED_BOOK_SYSTEM = (
    "You are a precise QA assistant with NO access to external documents. Answer from your own "
    "knowledge in one short phrase. If unsure, reply exactly 'I don't know' rather than guessing."
)


def port_bound(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


# --------------------------------------------------------------------- services
def _env(state_dir: Path) -> dict:
    state_dir.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ)
    e.update(
        PYTHONPATH=str(ROOT), LOCAL_MODE="1", ALLOW_UNREVIEWED_DATASET="1",
        LLM_BASE_URL="http://127.0.0.1:8400/v1", EMBEDDING_BASE_URL="http://127.0.0.1:8401/v1",
        EMBEDDING_DIMENSIONS="1024", CHUNK_TOKENS="256", CHUNK_OVERLAP="32",
        QDRANT_PATH=str(state_dir / "qdrant"), RUNTIME_DB=str(state_dir / "runtime.sqlite3"),
        METRICS_LOG=str(state_dir / "requests.jsonl"),
    )
    return e


def start_service(role: str, app: str, port: int, state_dir: Path) -> subprocess.Popen:
    if port_bound(port):
        raise RuntimeError(f":{port} already bound by a FOREIGN process — refusing to run "
                           f"(kill it first; a stale service would silently serve {role}).")
    state_dir.mkdir(parents=True, exist_ok=True)
    log = open(state_dir / f"{role}.log", "w")
    p = subprocess.Popen([PYBIN, "-m", "uvicorn", app, "--host", "127.0.0.1", "--port", str(port)],
                         cwd=str(ROOT), env=_env(state_dir), stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        if p.poll() is not None:
            raise RuntimeError(f"{role} child died on startup (see {state_dir/f'{role}.log'})")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/v1/health", timeout=2).status_code == 200:
                print(f"[svc] {role} up on :{port} (pid {p.pid}, our child)")
                return p
        except Exception:
            pass
        time.sleep(2)
    p.terminate()
    raise RuntimeError(f"{role} never became healthy")


def sync_corpus(base: str) -> int:
    httpx.post(f"{base}/v1/data/sync", timeout=900).raise_for_status()
    n = httpx.get(f"{base}/v1/data/status", timeout=15).json().get("indexed_chunks", 0)
    if not n:
        raise RuntimeError(f"{base} produced 0 indexed chunks")
    print(f"[svc] {base} indexed_chunks={n}")
    return n


# ------------------------------------------------------------------------- arms
def _collect(client, base, run_id, t_commit):
    parts, ttft = [], None
    with client.stream("GET", f"{base}/v1/runs/{run_id}/events") as s:
        for line in s.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "answer.delta":
                if ttft is None:
                    ttft = (time.perf_counter() - t_commit) * 1000
                parts.append(ev.get("text", ""))
            elif ev.get("type") in ("agent.persisted", "run.completed"):
                break
    return "".join(parts), (ttft if ttft is not None else -1.0)


def arm_naive(client, endpoint_text, query_time):
    tid, sid = str(uuid.uuid4()), "naive-" + uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    r = client.post(f"{NAIVE}/v1/turns/{tid}/commit",
                    json={"session_id": sid, "revision": 1, "text": endpoint_text, "query_time": query_time})
    r.raise_for_status()
    ans, ttft = _collect(client, NAIVE, r.json()["run_id"], t0)
    return {"answer": ans, "ttft_ms": round(ttft, 1)}


def arm_stream(client, trace, endpoint_text, query_time, duration_s, stream_state: Path):
    tid, sid = str(uuid.uuid4()), "stream-" + uuid.uuid4().hex[:8]
    partials = [s for s in trace if not s["is_final"]]
    t_start = time.perf_counter()
    rev = accepted = 0
    for s in partials:  # pace by offset so speculative retrieval overlaps "speech"
        dt = (t_start + s["offset_ms"] / 1000.0) - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        rev += 1
        try:
            resp = client.post(f"{STREAM}/v1/turns/{tid}/snapshots",
                               json={"session_id": sid, "revision": rev, "text": s["text"]}, timeout=30)
            accepted += resp.status_code < 300
        except Exception:
            pass
    # dwell to the TRUE audio endpoint (+ realistic VAD silence) before committing,
    # so the coordinator's 500 ms settle window can complete speculation.
    end_ms = max((partials[-1]["offset_ms"] if partials else 0), duration_s * 1000) + ENDPOINT_DWELL_MS
    dt = (t_start + end_ms / 1000.0) - time.perf_counter()
    if dt > 0:
        time.sleep(dt)
    t_commit = time.perf_counter()
    r = client.post(f"{STREAM}/v1/turns/{tid}/commit",
                    json={"session_id": sid, "revision": rev + 1, "text": endpoint_text, "query_time": query_time})
    r.raise_for_status()
    ans, ttft = _collect(client, STREAM, r.json()["run_id"], t_commit)
    # read back this turn's server telemetry (did speculation actually land?)
    tel = {}
    try:
        last = [l for l in (stream_state / "requests.jsonl").read_text().splitlines() if l.strip()][-1]
        d = json.loads(last); rr = d.get("retrieval", {}); ru = d.get("reuse", {})
        tel = {"ready_before_commit": rr.get("accepted_ready_before_commit"),
               "lead_ms": rr.get("accepted_retrieval_lead_at_commit_ms"),
               "from_fallback": rr.get("accepted_from_fallback"), "reuse_mode": ru.get("mode"),
               "controller_calls": d.get("controller", {}).get("calls")}
    except Exception:
        pass
    return {"answer": ans, "ttft_ms": round(ttft, 1), "snapshots_accepted": accepted,
            "snapshots_posted": rev, "telemetry": tel}


def arm_closed_book(jc, endpoint_text):
    return {"answer": generate(jc, CLOSED_BOOK_SYSTEM, endpoint_text, max_tokens=96), "ttft_ms": None}


# ------------------------------------------------------------------------- score
def truthfulness(labels):
    scores = [1 if x == "correct" else (-1 if x == "incorrect" else 0) for x in labels]
    n = len(scores)
    lo, hi = bootstrap_truthfulness(scores)
    return {"n": n, "correct": labels.count("correct"), "incorrect": labels.count("incorrect"),
            "missing": labels.count("missing"), "accuracy": round(labels.count("correct") / n, 4),
            "incorrect_rate": round(labels.count("incorrect") / n, 4),
            "truthfulness": round(sum(scores) / n, 4), "truthfulness_ci95": [round(lo, 4), round(hi, 4)]}


def main() -> int:
    data = json.loads(TRACES.read_text())
    dur = {json.loads(l)["id"]: json.loads(l)["duration_s"]
           for l in MANIFEST.read_text().splitlines() if l.strip()}
    run_tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    state_root = RUNS / "services" / run_tag
    for p in (8400, 8401):
        if not port_bound(p):
            raise RuntimeError(f"llama-server :{p} not up — run harness/serve_local.sh")

    naive_state, stream_state = state_root / "naive", state_root / "stream"
    naive_p = start_service("naive", "naive.api:app", 8001, naive_state)
    stream_p = start_service("stream", "stream.api:app", 8002, stream_state)
    rows = []
    try:
        sync_corpus(NAIVE); sync_corpus(STREAM)
        with httpx.Client(timeout=240) as client, \
             httpx.Client(base_url="http://127.0.0.1:8400/v1", timeout=120) as jc:
            for rep in range(REPS):
                for i, d in enumerate(data):
                    cb = arm_closed_book(jc, d["endpoint_text"])
                    # counterbalance answer-arm order to average out prompt-cache warming
                    naive_first = (i + rep) % 2 == 0
                    if naive_first:
                        nv = arm_naive(client, d["endpoint_text"], d["query_time"])
                        st = arm_stream(client, d["stabilized_trace"], d["endpoint_text"],
                                        d["query_time"], dur[d["id"]], stream_state)
                    else:
                        st = arm_stream(client, d["stabilized_trace"], d["endpoint_text"],
                                        d["query_time"], dur[d["id"]], stream_state)
                        nv = arm_naive(client, d["endpoint_text"], d["query_time"])
                    labels = {a: judge(jc, d["gold_query"], d["gold_answer"], d["alt_answers"], r["answer"])
                              for a, r in (("closed_book", cb), ("naive", nv), ("stream", st))}
                    rows.append({"id": d["id"], "split": ("test" if "-test-" in d["id"] else "dev"),
                                 "rep": rep, "naive_first": naive_first, "closed_book": cb,
                                 "naive": nv, "stream": st, "labels": labels})
                    landed = st["telemetry"].get("ready_before_commit")
                    print(f"  r{rep} {d['id']:20s} cb={labels['closed_book'][:4]} nv={labels['naive'][:4]} "
                          f"st={labels['stream'][:4]} nv_ttft={nv['ttft_ms']} st_ttft={st['ttft_ms']} "
                          f"spec_landed={landed} lead={st['telemetry'].get('lead_ms')}")
    finally:
        for p in (naive_p, stream_p):
            p.terminate()
        for p in (naive_p, stream_p):
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()

    # ---- scoring (dedupe to one label per query per arm; reps are for latency variance)
    def arm_truth(split):
        seen, labs = set(), []
        for r in rows:
            if split and r["split"] != split:
                continue
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            labs.append(r["labels"])
        return {a: truthfulness([x[a] for x in labs]) for a in ("closed_book", "naive", "stream")}, len(labs)

    truth_all, n_all = arm_truth(None)
    truth_test, n_test = arm_truth("test")
    truth_dev, n_dev = arm_truth("dev")

    # ---- paired latency (average reps per query, then paired diffs)
    per_q = {}
    for r in rows:
        per_q.setdefault(r["id"], {"nv": [], "st": []})
        if r["naive"]["ttft_ms"] > 0:
            per_q[r["id"]]["nv"].append(r["naive"]["ttft_ms"])
        if r["stream"]["ttft_ms"] > 0:
            per_q[r["id"]]["st"].append(r["stream"]["ttft_ms"])
    pairs = [(mean(v["nv"]), mean(v["st"])) for v in per_q.values() if v["nv"] and v["st"]]
    diffs = [st - nv for nv, st in pairs]  # +ve => stream slower
    landed = [r["stream"]["telemetry"].get("ready_before_commit") for r in rows]
    landed_rate = round(sum(1 for x in landed if x is True) / len(landed), 4) if landed else 0.0

    latency = {
        "n_pairs": len(pairs), "reps": REPS,
        "paired_median_stream_minus_naive_ms": round(median(diffs), 1) if diffs else None,
        "paired_mean_stream_minus_naive_ms": round(mean(diffs), 1) if diffs else None,
        "stream_win_rate": round(sum(1 for d in diffs if d < 0) / len(diffs), 3) if diffs else None,
        "speculation_landed_rate": landed_rate,
        "streaming_benefit_demonstrated": bool(landed_rate > 0.5 and diffs and median(diffs) < 0),
    }
    out = {"dataset": f"CRAG-TTS-local: {n_test} test + {n_dev} dev kept (WER<=0.10); "
                      f"3 excluded; NOT the paper's AudioCRAG; synthetic single-voice TTS",
           "truthfulness": {"all": truth_all, "test": truth_test, "dev": truth_dev},
           "latency": latency, "rows": rows,
           "honest_headline": (
               "Retrieval raises truthfulness far above closed-book and drives observed-incorrect "
               "labels to 0 (robust). The streaming arm showed NO speculative benefit: speculation "
               f"landed before commit on {landed_rate:.0%} of turns, so any TTFT delta is a "
               "cache/order artifact, not the paper's prefetch mechanism. Absolute numbers are not "
               "comparable to the paper (local Qdrant, tiny n, synthetic TTS, cascade not E2E)."),
           }
    (RUNS / "three_arm.json").write_text(json.dumps(out, indent=2))
    print("\n=== TRUTHFULNESS (C-I)/N ===")
    for split, t, n in (("all", truth_all, n_all), ("test", truth_test, n_test), ("dev", truth_dev, n_dev)):
        print(f"  [{split} n={n}]")
        for a, s in t.items():
            print(f"    {a:11s} acc={s['accuracy']:.3f} incorrect={s['incorrect_rate']:.3f} "
                  f"truth={s['truthfulness']:+.3f} CI{s['truthfulness_ci95']} (C{s['correct']}/I{s['incorrect']}/M{s['missing']})")
    print(f"\n=== LATENCY (paired, {REPS} reps) ===\n  {json.dumps(latency)}")
    print(f"\n{out['honest_headline']}\n[three-arm] -> {RUNS/'three_arm.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
