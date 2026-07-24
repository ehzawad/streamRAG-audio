# Naive RAG

`naive/` is the baseline path. It waits for Send, then retrieves from the exact
committed question. It has no draft endpoint, trigger, or speculative work.

The service uses `shared/` but does not import `stream/`, `frontend/`, or
`comparison/`. It remains runnable when those directories are absent.

## Run

```bash
ALLOW_UNREVIEWED_DATASET=1 \
QDRANT_PATH=./var/naive/qdrant \
RUNTIME_DB=./var/naive/runtime.sqlite3 \
METRICS_LOG=./var/naive/requests.jsonl \
uv run uvicorn naive.api:app --host 127.0.0.1 --port 8001
```

Build the local index once for fresh state:

```bash
curl --fail --request POST http://127.0.0.1:8001/v1/data/sync
```

The root returns service metadata, and the API schema is at
<http://127.0.0.1:8001/docs>. `ALLOW_UNREVIEWED_DATASET=1` is for development
before the dataset review is complete. Use the separate `frontend/` for chat.

Naive RAG reports the common answer, citation, latency, reliability, usage,
retrieval, and usage-accounting metrics. It does not emit StreamRAG
scheduling diagnostics.
