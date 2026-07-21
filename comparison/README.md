# Comparison CLI

`comparison/` is the headless benchmark consumer. It provisions two isolated
services from the committed candidate corpus, Naive RAG and StreamRAG, replays
identical typed questions, records content-addressed predictions, and scores
them offline against gold. It has no GUI and communicates only over
HTTP/JSON/SSE.

The runner checks each service's role, contract, fingerprints, index health, and
state identity. Gold answers are unavailable during inference and reach only the
offline scorer after predictions are finalized.

## Benchmark run

The committed corpus is a candidate
(`approval_status = candidate_pending_human_review`), so each service loads it
with `ALLOW_UNREVIEWED_DATASET=1`.

```bash
make benchmark-services-check
make benchmark-services-sync

# Terminal A
make benchmark-services-serve

# Terminal B
make benchmark
make score
```

Everything is bound to `data/crag_eval` (`test_queries.jsonl` /
`test_gold.jsonl`, verified via `checksums.sha256`). `make benchmark` writes
predictions to `comparison/benchmark/results/predictions.jsonl`; `make score`
writes the report to `summary.json`. The report compares common latency,
correctness, reliability, usage, and cost metrics; StreamRAG scheduling
diagnostics remain path-specific.
