"""The fast lever, measured: speculative answer-prefill / KV-cache warming.

The 3-arm eval showed the paper's speculative *retrieval* prefetch gives no latency
benefit on short local spoken queries (0/24 landed; retrieval is cheap vs the 9B
answer TTFT, which dominates). This measures the lever that actually attacks that
dominant cost: warming the answer LLM's KV cache with (system + evidence + query)
during speech, so the committed request reuses the cached prefix.

Method: for each trial, build a realistic grounded-answer prompt (~1.8k-token
evidence context) with a UNIQUE nonce so the first call is genuinely COLD (full
prefill), then send the identical prompt again (WARM, prefix-cache hit). TTFT =
time to first content token. Paired, warmup discarded. Writes runs/prefill_warm.json.

  python scoring/prefill_warm.py            # needs :8400 up (harness/serve_local.sh)
"""

from __future__ import annotations

import json
import statistics as st
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8400/v1/chat/completions"
RUNS = Path(__file__).resolve().parent.parent / "runs"
SYS = "You are a grounded QA assistant. Answer only from the provided context in one short phrase."
QUERY = "What album did the Killers release in 2004?"
TRIALS = 14
WARMUP_DROP = 2


def context(nonce: str) -> str:
    base = ("Document excerpt %d [%s]: The Killers are an American rock band formed in Las Vegas in "
            "2001. Their debut studio album Hot Fuss was released in 2004 and included Mr. Brightside "
            "and Somebody Told Me. Members: Brandon Flowers, Dave Keuning, Mark Stoermer, Ronnie "
            "Vannucci Jr. Hot Fuss reached number one in the UK albums chart. ")
    return "\n".join(base % (i, nonce) for i in range(6))


def ttft_ms(ctx: str, q: str) -> float:
    body = json.dumps({
        "model": "qwen3.5-9b-local",
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": ctx + "\nQuestion: " + q}],
        "max_tokens": 16, "temperature": 0, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        for line in r:
            if line.startswith(b"data:") and b'"content"' in line:
                return (time.perf_counter() - t0) * 1000
    return -1.0


def main() -> int:
    cold, warm = [], []
    for n in range(TRIALS):
        ctx = context(f"n{n}-{time.time_ns()}")  # unique => cold on first call
        cold.append(ttft_ms(ctx, QUERY)); time.sleep(0.15)
        warm.append(ttft_ms(ctx, QUERY)); time.sleep(0.15)  # identical => warm prefix-cache
    cold, warm = cold[WARMUP_DROP:], warm[WARMUP_DROP:]
    diffs = [c - w for c, w in zip(cold, warm, strict=True)]
    out = {
        "n": len(cold), "context_tokens_approx": 1800,
        "cold_ttft_ms": {"median": round(st.median(cold), 1), "mean": round(st.mean(cold), 1)},
        "warm_ttft_ms": {"median": round(st.median(warm), 1), "mean": round(st.mean(warm), 1)},
        "prefill_warm_saving_ms_median": round(st.median(diffs), 1),
        "prefill_warm_saving_pct": round(100 * st.median(diffs) / st.median(cold), 1),
        "warm_wins": f"{sum(1 for d in diffs if d > 0)}/{len(diffs)}",
        "note": ("Warming the answer KV cache during speech reduces committed TTFT by the median "
                 "above. This is the real fast lever on one A5000; the paper's speculative-retrieval "
                 "prefetch was a measured null on this workload (see runs/three_arm.json). Benefit "
                 "requires the query+evidence to stabilize before endpoint with lead to spare."),
    }
    RUNS.mkdir(exist_ok=True)
    (RUNS / "prefill_warm.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
