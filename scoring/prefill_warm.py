"""Answer-prefill / KV-cache reuse probe — an HONEST NEGATIVE.

An earlier version of this file reported a "-52% TTFT from KV-cache warming" and
called it the fast lever. Instrumenting the server (cache_n / prompt_n) — after a
codex-council round-3 audit — showed that was WRONG: the KV prefix cache does not
reuse for Qwen3.5-9B on this llama.cpp build, so there is no prefill-warming benefit.

Why: Qwen3.5-9B is a HYBRID/recurrent model (GatedDeltaNet + gated attention). Its
per-sequence recurrent state cannot be partially reused across requests, so llama.cpp
rolls the reusable prefix back to ~0 (only the short chat-template head survives).
Even an IDENTICAL prompt re-prefills every token. The wall-clock TTFT variation seen
earlier was GPU scheduling / first-after-idle warmup (the prefill compute is identical),
not cache reuse — and it is not a deployable optimization.

This probe now measures the truth: for a realistic grounded-answer prompt, send it
COLD (fresh) then IMMEDIATELY REPEAT (identical), and read the server's own
`timings.cache_n` (cached tokens) and `timings.prompt_n` (tokens actually evaluated).
A working prefix cache would show cache_n ~= prompt_n-1 on the repeat; here it does not.

  python scoring/prefill_warm.py      # needs :8400 up (harness/serve_local.sh)
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
TRIALS = 10


def evidence(nonce: str) -> str:
    base = ("Document %d [%s]: The Killers are an American rock band formed in Las Vegas in 2001. "
            "Their debut album Hot Fuss (2004) included Mr. Brightside and Somebody Told Me. Members: "
            "Brandon Flowers, Dave Keuning, Mark Stoermer, Ronnie Vannucci Jr. It reached number one "
            "in the UK albums chart. ")
    return "\n".join(base % (i, nonce) for i in range(10))


def call(prompt: str, stream: bool):
    body = {"model": "qwen3.5-9b-local", "messages": [{"role": "system", "content": SYS},
            {"role": "user", "content": prompt}], "max_tokens": 8, "temperature": 0,
            "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
    if stream:
        body["stream"] = True
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if not stream:
        r = json.load(urllib.request.urlopen(req))
        t = r["timings"]
        return {"cache_n": t["cache_n"], "prompt_n": t["prompt_n"], "prompt_ms": round(t["prompt_ms"], 1)}
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        for line in r:
            if line.startswith(b"data:") and b'"content"' in line:
                return {"ttft_ms": round((time.perf_counter() - t0) * 1000, 1)}


def main() -> int:
    cold_ttft, repeat_ttft, cold_tok, repeat_tok = [], [], [], []
    for n in range(TRIALS):
        time.sleep(0.3)
        p = evidence(f"n{n}-{time.time_ns()}") + "\nQuestion: " + QUERY
        cold_ttft.append(call(p, True)["ttft_ms"])
        cold_tok.append(call(p, False))          # this is already a repeat -> would show cache if any
        repeat_ttft.append(call(p, True)["ttft_ms"])
        repeat_tok.append(call(p, False))
    cold_ttft, repeat_ttft = cold_ttft[1:], repeat_ttft[1:]
    repeat_tok = repeat_tok[1:]
    med_prompt_n = st.median([x["prompt_n"] for x in repeat_tok])
    med_cache_n = st.median([x["cache_n"] for x in repeat_tok])
    out = {
        "verdict": "NO KV prefix reuse for Qwen3.5-9B on this llama.cpp build -> no prefill-warming lever",
        "n": len(cold_ttft),
        "repeat_identical_prompt": {
            "median_cache_n_cached_tokens": med_cache_n,
            "median_prompt_n_evaluated_tokens": med_prompt_n,
            "reuse_fraction": round(med_cache_n / (med_cache_n + med_prompt_n), 3),
        },
        "wall_clock_ttft_ms": {
            "median_cold": round(st.median(cold_ttft), 1),
            "median_repeat": round(st.median(repeat_ttft), 1),
            "note": ("Any cold-vs-repeat TTFT gap is a GPU-scheduling / first-after-idle artifact: "
                     "the repeat still evaluates ~all prompt tokens (prompt_n above), i.e. the prefill "
                     "compute is unchanged. It is NOT cache reuse and NOT a deployable speedup."),
        },
        "why": ("Qwen3.5-9B is hybrid/recurrent (GatedDeltaNet); its per-sequence recurrent state "
                "cannot be partially reused, so llama.cpp cannot reuse the KV prefix across requests. "
                "Corrected after codex-council round 3 caught the original over-attributed claim."),
    }
    RUNS.mkdir(exist_ok=True)
    (RUNS / "prefill_warm.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
