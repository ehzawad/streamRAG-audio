# Deployment (Modal)

Public, scale-to-zero deployment of the StreamRAG stack on [Modal](https://modal.com).
The local project runs as five docker-compose containers (frontend + two FastAPI
paths + two Qdrant servers). Modal doesn't run compose, so this folder collapses
that topology into **one container behind one HTTPS URL** — without changing any
application code.

**Live URL:** printed by `modal deploy` and listed by `modal app list`. It is
intentionally not committed here — the endpoint is public and unauthenticated, so
share it yourself on your own terms.

```
browser ──► gateway (deployment/modal_app.py, a FastAPI app)
               ├─ mount /api/naive  ──► naive.api:app   ──► embedded Qdrant @ /app/var/naive
               ├─ mount /api/stream ──► stream.api:app  ──► embedded Qdrant @ /app/var/stream
               └─ static: the built React SPA (frontend/dist), same-origin, no CORS
           naive & stream ──► OpenAI API (text-embedding-3-large + gpt-5.6-sol)
```

## What changed vs. the local stack, and why

| Local (docker-compose) | Modal | Why |
|---|---|---|
| 2 standalone **Qdrant servers** | **embedded Qdrant** (`QDRANT_URL` unset) on a Modal Volume | The code already supports embedded local mode; drops two services and inter-container networking. |
| nginx proxies `/api/*` to two backends | one **FastAPI gateway** mounts both apps at `/api/naive` + `/api/stream` | Same-origin surface the frontend already expects (`frontend/src/api.ts`), so no CORS and one URL. |
| ports bound to `127.0.0.1`, no TLS | Modal-issued `*.modal.run` URL with **automatic HTTPS** | Modal terminates TLS; no domain or reverse proxy to run. |
| `make docker-sync` after boot | **idempotent index build on container startup** | First cold start embeds the corpus once (~80s, a few cents) into the Volume; later starts re-embed nothing. |
| Python from `uv:python3.14` base | same base image | The code uses Python 3.14-only grammar (`except A, B:`), so 3.14 is required. |

**State isolation is preserved.** `naive.config` and `stream.config` build separate
`Settings` with separate default paths (`var/naive/…` vs `var/stream/…`), and
`shared.api.factory.create_app` is fully parameterized — so both apps live in one
interpreter without ever sharing an index, a SQLite runtime DB, or a cache. The
native Rust snapshot module is omitted; `stream/snapshot.py` runs its identical
pure-Python fallback.

## Prerequisites

- Modal CLI, authenticated (`modal token set …` or `~/.modal.toml`).
- A Modal **Secret** named `streamrag-openai` holding `OPENAI_API_KEY`:

  ```bash
  modal secret create streamrag-openai OPENAI_API_KEY="sk-…"
  ```

- The built frontend at `frontend/dist/` (already committed; rebuild with
  `cd frontend && npm ci && npm run build` if you change the UI).

## Commands

```bash
# from the project root
modal run    deployment/modal_app.py     # sanity probe: image runs 3.14 + apps import
modal deploy deployment/modal_app.py     # build image, publish the public URL
modal app stop streamrag                 # take it offline (keeps the Volume + index)
modal volume delete streamrag-state      # optional: delete the built indexes ($0 storage)
```

The first request after a deploy (or after a scale-down) is a cold start: the
container boots, brings up both paths, and runs each path's idempotent
`/v1/data/sync`. On a fresh Volume that indexes 250 documents → 1000 chunks per
path (~80s). Once the vectors are committed to the Volume, later cold starts skip
re-embedding and are fast.

## Cost model

- **GPU:** none — this stack is OpenAI-API-based, so it runs on CPU only.
- **Compute:** scale-to-zero. You pay for CPU/RAM only while a request is being
  served (plus a 5-minute idle keep-warm, `scaledown_window=300`). Idle = \$0.
- **Storage:** one small Modal Volume (`streamrag-state`) holds the embeddings —
  a few MB. Delete it with `modal volume delete streamrag-state` for \$0.
- **OpenAI:** every answer calls `gpt-5.6-sol`; every new question embeds a query
  with `text-embedding-3-large`. **The endpoint is public and unauthenticated —
  anyone with the URL spends your OpenAI budget.** Set spend limits on your
  OpenAI key, or stop the app when you're done demoing.

## Configuration knobs (`deployment/modal_app.py`)

- `max_containers=1` — single writer for the embedded Qdrant/SQLite on the Volume.
- `@modal.concurrent(max_inputs=20)` — one container fans out over many
  concurrent requests because the work is OpenAI-I/O-bound.
- `scaledown_window=300`, `timeout=900` — idle keep-warm and a generous startup
  budget to cover first-boot indexing.

The locked benchmark config (model `gpt-5.6-sol`, `text-embedding-3-large`,
reasoning effort `medium`, 3072 dims) is enforced by `Settings.validate()` in the
app code; the deployment does not override it.

## Verified live

- `/api/naive/v1/health` and `/api/stream/v1/health` → `ok: true`,
  `index_ready: true`, `indexed_chunks: 1000` for both paths.
- `/compare`: "Who stepped down as Apple's CEO in August 2011?" → both paths
  answered **Steve Jobs** with a citation; StreamRAG's pre-Send prefetch gave
  ~1.2s lower time-to-first-token.
- Multi-turn: follow-up "Who replaced him as CEO?" → both paths resolved the
  pronoun from history and answered **Tim Cook**.
