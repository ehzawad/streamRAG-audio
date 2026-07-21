# Naive RAG vs StreamRAG benchmark

**Status:** local development evidence on a candidate dataset; a small-scale,
local, single-machine measurement, not a production-scale or final accuracy
claim.

## Comparison contract

The benchmark measures one product difference: Naive starts retrieval after
**Send**; StreamRAG may prepare evidence during typed input. Both paths start
answer generation only after **Send**.

Both services use the same 250 complete documents, 1,000 chunks, embeddings,
search policy, answer model and prompt, local tool, memory policy, and offline
scorer. They run in different processes with separate Qdrant, SQLite, metrics,
sessions, and caches.

The external runner validates role, capabilities, configuration, source and
dataset fingerprints, index health, and distinct service identity. It measures
the paths sequentially and counterbalances question order. Gold is unavailable
until predictions and their manifest are finalized.

## Benchmark run (10 held-out test questions)

Reproduce with two isolated services (`make benchmark-services-serve`) then
`make benchmark && make score`, which writes
`comparison/benchmark/results/{predictions.jsonl,summary.json,summary.md}`.

- 10 held-out test questions × 2 paths (20 outputs), 1 measured repetition;
- deterministic 70 WPM input with 400 ms changed-draft snapshots and a 5-second
  pre-Send dwell;
- real `gpt-5.6-sol` and `text-embedding-3-large` calls against two isolated
  Docker services;
- clean run: 20/20 completed, 0 failures, `run_integrity_gate: complete`.

| Path | Answer proxy | Support + citation | Median TTFT | p95 TTFT | Median total | Model calls/output | Cost/output |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive | 100% | 100% | 2,274 ms | 6,818 ms | 2,860 ms | 1.2 | $0.0125 complete |
| StreamRAG | 100% | 100% | 1,185 ms | 6,479 ms | 1,814 ms | 4.1 | ≥ $0.0239 lower bound |

Paired (10 A/B pairs):

- StreamRAG TTFT wins: **10/10**;
- median TTFT delta: **-782 ms (-42.1%)**;
- median total-time delta: **-868 ms**;
- exact speculative evidence reuse (`presubmit_reuse`): **10/10**;
- automatic-accuracy delta: **0** (both correct on all 10; 10/10 same outcome).

Honest trade-off: StreamRAG cuts perceived latency (median TTFT down 42.1%) at
higher cost. It issues about 4.1 model calls per output versus Naive's 1.2 (a
trigger decision plus speculative retrieval while typing), so its per-output cost
is roughly double and is reported as a lower bound because cancelled speculative
calls do not always return provider usage. Accuracy is identical on this set, so
the benefit is latency paid for in tokens and calls. Automatic answer/citation
checks are proxies, not human semantic judgments.

An earlier full run recorded one StreamRAG failure on the "black swan" question:
a transient `RemoteProtocolError` (client-side transport disconnect at 18 s, well
under the 45 s deadline, with container `RestartCount=0`). The scorer counted it
honestly as a failure; the clean re-run above then completed 20/20, confirming the
blip was transient transport rather than a pipeline defect.

## Native SnapshotAnalyzer microbenchmark (Rust vs Python)

`SnapshotAnalyzer.analyze` runs on every typed draft and feeds the StreamRAG
trigger. It is implemented in Rust (`native/snapshot_delta/`, PyO3) behind an
import seam with a byte-identical pure-Python fallback. `make bench-native`
verifies parity and times both backends over 90 draft pairs, including
append-only, correction, empty-input, and Unicode/whitespace edge cases (matching
CPython's `str.split()`, including U+001C–U+001F):

- parity: **90/90 identical `SnapshotDelta`** (fingerprint, common-prefix chars,
  word count, new words, append flag);
- speed: **~1.44 µs/call (Python) vs ~0.35 µs/call (Rust) ≈ 4.1× faster**.

This is a per-call microbenchmark only. It is not attributed to the end-to-end
latency above: network, embedding, retrieval, and model-generation time dominate a
request, so the native speedup does not by itself explain the TTFT difference.

## Real browser acceptance

The final Docker topology was also exercised through Google Chrome. Playwright
handled navigation, Send, and DOM inspection; native macOS Computer Use entered
each character in the visible textbox. No paste, programmatic fill, or whole-text
injection was used.

The same deterministic human-like cadence and verbatim prompts were used for
both paths, followed by a fixed 2.7-second pre-Send dwell. The three fresh-chat
prompts were:

1. `how long does a stock need to be held to make capital gains long term?`
2. `which dune movie has better music, 1984 or 2021?`
3. `what is the name of the bad bunny album released before nadie sabe lo que va a pasar manana?`

The four-turn conversation used prompt 1 followed by:

1. `Does exactly one year qualify?`
2. `When does that holding period start?`
3. `Summarize both rules in one sentence.`

| Scenario | Work per path | Naive avg TTFT / total | StreamRAG avg TTFT / total | StreamRAG reduction | Correct |
|---|---:|---:|---:|---:|---:|
| Standalone, fresh chat | 3 | 3,749 / 4,673 ms | 1,641.333 / 3,159.667 ms | 56.219% / 32.385% | 3/3 both |
| Simultaneous `/compare` | 3 | 1,876.333 / 2,745.333 ms | 1,307.667 / 2,316.333 ms | 30.307% / 15.627% | 3/3 both |
| Multi-turn `/compare` | 4 | 1,898.25 / 2,966.75 ms | 1,156.75 / 1,785.25 ms | 39.062% / 39.825% | 4/4 both |
| Multi-turn solo routes | 4 | 2,201.25 / 3,272 ms | 1,547.5 / 2,242.25 ms | 29.699% / 31.472% | 4/4 both |

Raw timings:

| Scenario | Naive TTFT | StreamRAG TTFT | Naive total | StreamRAG total |
|---|---|---|---|---|
| Standalone | 4,297 / 3,234 / 3,716 | 1,957 / 1,664 / 1,303 | 5,393 / 4,622 / 4,004 | 3,069 / 4,800 / 1,610 |
| Simultaneous | 1,576 / 2,048 / 2,005 | 1,799 / 919 / 1,205 | 2,778 / 3,152 / 2,306 | 2,756 / 2,660 / 1,533 |
| Multi-turn Compare | 1,844 / 1,750 / 1,821 / 2,178 | 915 / 917 / 1,110 / 1,685 | 2,614 / 4,300 / 2,355 / 2,598 | 1,610 / 1,458 / 1,877 / 2,196 |
| Multi-turn solo | 1,775 / 1,881 / 2,952 / 2,197 | 1,128 / 2,161 / 991 / 1,910 | 2,564 / 2,789 / 3,743 / 3,992 | 1,766 / 2,960 / 1,738 / 2,505 |

Both paths produced correct, cited answers on all 14 browser turns. StreamRAG had
exact evidence ready before Send on all 14, never answered early, and won 12/14
individual first-token and total-time races. Provider variance still caused two
losses; the approach does not guarantee every request is faster.

This browser study is manually reproducible product evidence, not an automated or
gold-blind benchmark. Its answers were inspected rather than hash-bound human
adjudicated.

After the final Docker rebuild on 2026-07-19, Chrome was rechecked with the Black
Swan question entered one character at a time at 120 ms intervals. Before Send,
StreamRAG showed that candidate evidence was ready while both paths still showed
no answer. After Send, both returned Natalie Portman with the same support
citation; Naive reported 2,649 ms TTFT / 3,195 ms total and StreamRAG 1,810 ms /
2,307 ms. Playwright also opened `/`, `/naive`, `/stream`, and `/compare`; all
health/data requests returned 200 and the browser console had no errors. This is
a product spot check, not an additional benchmark row.

## Clean reproduction evidence

The recorded clean Docker run verified:

- 250 documents, 5 development questions, 10 held-out questions, and all 9
  checksummed files;
- 1,000 points in each isolated Qdrant service;
- real clean index syncs of 40.926 seconds for Naive and 39.055 seconds for
  StreamRAG, each embedding 366,142 tokens;
- persistence across container removal and recreation without re-embedding;
- linting and a production frontend build;
- five healthy containers, matching source/index hashes, zero path failures, and
  no application error or HTTP 4xx/5xx in the final logs.

Run the same development pipeline with the commands in [`RUN.md`](RUN.md).

## Artifact ownership

`comparison/benchmark/results/` contains the retained benchmark evidence:
`predictions.jsonl`, the content-addressed `predictions.manifest.json`, the
machine-readable `summary.json`, and the Markdown `summary.md`. They are kept as
evidence, not generated clutter.

There is no separate `final` artifact directory and no approval gate;
re-running `make benchmark` then `make score` regenerates
`comparison/benchmark/results/predictions.jsonl` and `summary.json`.

## Claim boundary

The supported result is narrow: on this ten-question test set and the
manual browser checks, StreamRAG preserved the measured correctness proxies and
reduced perceived latency when usable evidence stabilized before Send. The data
does not establish a universal speed, accuracy, or cost improvement, and results
from the StreamRAG paper are not treated as results of this implementation.
