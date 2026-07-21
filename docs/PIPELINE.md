# Pipeline and architecture

## Product contract

StreamRAG compares one scheduling choice over the same retrieval and answer
stack:

| Path | Before Send | At and after Send |
|---|---|---|
| Naive RAG | no retrieval | retrieve the exact committed text, then answer |
| StreamRAG | analyze changed drafts and prepare evidence | validate exact-text reuse or retrieve the committed text, then answer |

No path may generate or display an answer before **Send**. Speculative evidence
is accepted only when its recorded draft exactly equals the committed text.
Corrections, stale work, trigger failure, or retrieval failure fall back to the
same committed-text retrieval used by Naive.

This transfers StreamRAG's scheduling idea to typed input. It does not reproduce
the paper's speech stack, trained trigger, reranker, model training, or scale.

## Component ownership

| Component | Owns | Does not own |
|---|---|---|
| `naive/` | post-Send retrieval policy | StreamRAG, GUI, scoring |
| `stream/` | snapshots, trigger, speculative retrieval, reuse validation | Naive, GUI, scoring |
| `shared/` | corpus/index code, answer agent, tool, memory, API lifecycle, contracts | either path's scheduling policy |
| `frontend/` | route hub, chat UI, browser request lifecycle | Python implementations, gold, benchmark logic |
| `comparison/` | service provisioning, deterministic replay, artifacts, offline scoring | GUI and application imports |

The APIs are independently runnable. Neither imports or dispatches the other.
The frontend and comparison runner communicate only through HTTP/JSON/SSE. This
keeps the GUI removable, the headless evaluator removable, and each RAG path
testable by itself.

## Knowledge-base flow

1. `checksums.sha256` binds the committed dataset files.
2. `documents.jsonl.bz2` supplies 250 complete document rows.
3. The shared chunker creates 400-token chunks with 50-token overlap.
4. `text-embedding-3-large` creates 3,072-dimensional vectors.
5. Each path writes the same 1,000 chunks to its own Qdrant collection.
6. Index readiness requires matching checksums, source fingerprint, pipeline
   version, desired count, and physical point count.

Compose runs two private Qdrant servers. `qdrant-naive` is visible only to the
Naive API on `naive-data`; `qdrant-stream` is visible only to StreamRAG on
`stream-data`. Neither Qdrant port is published to the host.

## Typed request flow

The browser and deterministic replay deliver changed cumulative drafts every
400 ms. StreamRAG uses a bounded low-reasoning model trigger while text is
changing and may start exact-draft retrieval after the latest delivered draft
has remained unchanged for 500 ms.

At Send:

1. the server reserves and freezes the final revision;
2. Naive retrieves with the committed text;
3. StreamRAG checks whether prepared evidence belongs to that exact text;
4. incompatible work is cancelled or discarded and replaced by committed-text
   retrieval;
5. the common grounded answer agent receives only accepted evidence;
6. tokens and citations stream through SSE;
7. conversation memory, metrics, and usage are persisted after the visible
   answer under one bounded deadline.

The frontend holds at most one active snapshot request plus one replaceable
latest draft. Send cancels obsolete snapshot transport without waiting. Compare
starts both commits concurrently but preserves separate backend sessions.

Snapshot delta analysis runs through a Rust native `SnapshotAnalyzer` backend
(`native/snapshot_delta/`, PyO3 module `streamrag_snapshot`) with a pure-Python
fallback in `stream/snapshot.py`; build and benchmark it with `make native` and
`make bench-native`.

## Agent, tool, and memory

- Answer model: `gpt-5.6-sol`, medium reasoning.
- Trigger and memory summary: low reasoning.
- Primary retrieval: 8 candidates, 5 answer-context chunks, cosine search.
- Dynamic tool: strict read-only PydanticAI `search_local_crag`, limited to one
  model-issued call when primary evidence is insufficient.
- Memory: independent asynchronous SQLite history and rolling summary per path.

Questions, history, timestamps, and retrieved text are serialized as untrusted
user-role content. Privileged instructions remain static. Retrieval evidence is
turn-local; only conversation text becomes memory.

## Fairness and evaluation boundary

Both paths share the dataset, final committed text, chunker, embeddings, search,
answer model and prompt, tool, memory policy, common metrics, and scorer. The
runner also validates implementation role, service capability, configuration,
source and dataset fingerprints, index readiness, and distinct process/state
identity.

Common metrics cover first-token and total latency, completion/failure, retrieval,
answer/support/citation proxies, token usage, accounting coverage, and observed
cost. StreamRAG separately reports trigger calls, speculation, evidence lead,
reuse, stale work, cancellation, and fallback. Missing provider usage is marked
as a lower bound, never counted as zero.

Gold is not read by either service or by the single benchmark runner. The offline
scorer reads it only after predictions are finalized and hashed. Automatic answer
and citation checks are not presented as human semantic accuracy.

## State and concurrency

The Docker stack has four persistent volumes: one Qdrant index and one embedded
SQLite/runtime volume per path. SQLite is a library inside each API image, not a
separate server. Ordinary `docker compose down` removes containers and networks
but preserves all four volumes.

FastAPI, OpenAI, and networked Qdrant operations are asynchronous. Standalone and
headless embedded-Qdrant mode moves synchronous vector work off the event loop.
This supports a responsive local evaluation; it is not a production concurrency
or tenancy claim.

## Security and deployment boundary

The supplied stack is local-only and unauthenticated. Host ports bind to
`127.0.0.1`; the browser uses one frontend origin, and the OpenAI key remains in
the API environments. `.env`, generated indexes, SQLite, sessions, logs, and
metrics are excluded from Git. The fixed corpus and evaluation metadata are
intentionally committed.

Anyone who can reach an API can submit requests, consume OpenAI budget, and read
results. A public deployment therefore needs TLS, authentication and authorization,
principal-bound sessions, rate and spend limits, protected index administration,
secret management, backups, retention, abuse monitoring, and load/security tests.
Changing a bind address alone is not a deployment plan.

## References

- [StreamRAG paper](https://arxiv.org/abs/2510.02044)
- [CRAG repository and license](https://github.com/facebookresearch/CRAG)
