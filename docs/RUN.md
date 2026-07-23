# Run and reproduce

Everything here runs **fully on one box** — no hosted API. The answer/trigger/judge
model and the embedder are local llama.cpp servers (see [`LOCAL.md`](LOCAL.md)).

## Prerequisites

- `uv`, Python 3.14, and (for the GUI) Node.js/npm;
- `make` and `curl`;
- the two llama.cpp servers up on the A5000 (`:8400` `qwen3.5-9b-local`, `:8401`
  `bge-large-en-v1.5`) — `bash harness/serve_local.sh`.

## Configure

```bash
cp .env.example .env
```

The defaults already point at the local servers. The one value to set for a run is:

```dotenv
ALLOW_UNREVIEWED_DATASET=1
```

`ALLOW_UNREVIEWED_DATASET=1` lets the services load the committed candidate
corpus. There is no sealed test run; this is the normal setting for every run.

## Run the two services

Each service embeds its own Qdrant + SQLite under `./var/<role>/`.

```bash
# terminal A — Naive RAG on :8001
make dev-naive
# terminal B — StreamRAG on :8002
make dev-stream
```

Build each index once it is up (reads the committed 250-document corpus, writes
its chunks to that service's Qdrant):

```bash
make sync-naive
make sync-stream
```

Then, for the browser comparison UI:

```bash
make dev-frontend      # http://127.0.0.1:5173/
```

| URL | Experience |
|---|---|
| <http://127.0.0.1:5173/naive> | Naive RAG only |
| <http://127.0.0.1:5173/stream> | StreamRAG only |
| <http://127.0.0.1:5173/compare> | simultaneous comparison |

Ports 8001 and 8002 expose JSON APIs and generated OpenAPI docs. End users need
only the frontend URL.

## The typed benchmark (headless)

The headless provisioner creates two isolated embedded service stores, replays the
queries through both, and scores offline against sealed gold:

```bash
make benchmark-services-check
make benchmark-services-sync

# terminal A
make benchmark-services-serve

# terminal B
make benchmark
make score
```

`make benchmark` replays `test_queries.jsonl` through both services with
deterministic 70 WPM typing, changed-only 400 ms snapshots, and a 5-second
pre-Send dwell, writing predictions to
`comparison/benchmark/results/predictions.jsonl`. `make score` binds the gold set
via `checksums.sha256` and writes `summary.json`. This is the single benchmark
path, bound to `data/crag_eval`.

## The audio pipeline

The spoken-query evaluation (Qwen3-TTS synth → faster-whisper ASR → the streaming
RAG core) is documented in [`AUDIO.md`](AUDIO.md); its reproduce block and the
one-clip `make smoke-9b` end-to-end check live there.

## Verify the repository

```bash
make setup
make check
make smoke-9b   # needs the :8400/:8401 servers up (see AUDIO.md)
```

`make check` runs Python linting (ruff) and a production frontend build. The
committed corpus is checksum-bound; each service verifies those checksums when it
loads the dataset, so no separate verification step is required.

Useful live checks:

```bash
curl --fail http://127.0.0.1:8001/v1/health
curl --fail http://127.0.0.1:8002/v1/health
```

Both API health payloads should report `"ok": true`, `"index_ready": true`,
`"model": "qwen3.5-9b-local"`, and `"embedding_model": "bge-large-en-v1.5"`.

## Persistent state

Each standalone service keeps generated state under `./var/<role>/`:

- `var/naive/qdrant`, `var/naive/runtime.sqlite3`, `var/naive/requests.jsonl`
- `var/stream/qdrant`, `var/stream/runtime.sqlite3`, `var/stream/requests.jsonl`

These hold generated indexes, SQLite conversations, and logs; they are gitignored.
Deleting them removes generated state but not the committed dataset — re-run the
`sync-*` targets to rebuild.

## Troubleshooting

- **API health says the dataset is unapproved:** set
  `ALLOW_UNREVIEWED_DATASET=1` in `.env`, then restart the service.
- **Index is not ready:** run the matching `make sync-*` target and inspect the
  service log under `var/<role>/`.
- **Health reports the wrong model/embedding alias:** the llama.cpp servers were
  started with different `--alias` values; match them to `.env` (`OPENAI_MODEL`,
  `OPENAI_EMBEDDING_MODEL`).
- **Port already in use:** stop the conflicting process before starting.
- **A changed `.env` has no effect:** configuration is validated at startup;
  restart the affected service.

The supplied ports are loopback-only and the stack has no authentication. See
the deployment and security boundary in [`PIPELINE.md`](PIPELINE.md) before any
network exposure.
