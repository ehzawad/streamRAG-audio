"""Shared eval harness helpers (extracted from the retired run_three_arm.py).

Clean isolation + arm + scorer primitives reused by the multi-voice headline eval:
  * port_bound / start_service : fail if a port is already bound; unique per-run state
    dirs; verify the spawned child is alive (a stale service would silently serve).
  * sync_corpus                : index the corpus into a fresh Qdrant and assert n>0.
  * arm_closed_book / arm_naive: the two answer arms (parametric baseline vs naive-RAG).
  * truthfulness               : (correct-incorrect)/n with a bootstrap CI; reports
                                 "incorrect labels", not "hallucination".

The streaming arm and the 3-arm driver were retired with the Chatterbox pipeline; the
frozen latency evidence lives in runs/three_arm.json (historical, not re-run).
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

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comparison.crag_task1_eval import bootstrap_truthfulness, generate  # noqa: E402

RUNS = ROOT / "runs"
PYBIN = os.getenv("RAG_PYBIN", "/mnt/sdb/arafat/ehz/llm/streamRAG/.venv/bin/python")
NAIVE = "http://127.0.0.1:8001"

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
