# StreamRAG

`stream/` is the typed-input path. It can prepare evidence before Send, but it
never generates an answer from an uncommitted draft. Prepared evidence is reused
only when its draft exactly matches the committed question; otherwise the path
retrieves again from the committed text.

The service uses `shared/` but does not import `naive/`, `frontend/`, or
`comparison/`. It remains runnable when those directories are absent.

## Run

```bash
ALLOW_UNREVIEWED_DATASET=1 \
QDRANT_PATH=./var/stream/qdrant \
RUNTIME_DB=./var/stream/runtime.sqlite3 \
METRICS_LOG=./var/stream/requests.jsonl \
uv run uvicorn stream.api:app --host 127.0.0.1 --port 8002
```

Build the local index once for fresh state:

```bash
curl --fail --request POST http://127.0.0.1:8002/v1/data/sync
```

The root returns service metadata, and the API schema is at
<http://127.0.0.1:8002/docs>. `ALLOW_UNREVIEWED_DATASET=1` is for development
before the dataset review is complete. Use the separate `frontend/` for chat.

StreamRAG reports the common comparison metrics plus speculation, evidence
lead/reuse, stale-work, cancellation, and fallback diagnostics.
