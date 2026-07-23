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
3. The shared chunker creates 256-token chunks with 32-token overlap (bge caps at
   512 tokens; 400-token chunks overflow its tokenizer).
4. Local **bge-large-en-v1.5** (llama.cpp `:8401`) creates 1,024-dimensional vectors.
5. Each path writes the same chunk set to its own Qdrant collection (the count is
   corpus- and chunker-specific).
6. Index readiness requires matching checksums, source fingerprint, pipeline
   version, desired count, and physical point count.

Each service runs its own embedded local Qdrant under `./var/<role>/qdrant`; the
two paths never share a collection.

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

Snapshot delta analysis runs through the pure-Python `SnapshotAnalyzer` in
`stream/snapshot.py`.

## Agent, tool, and memory

- Answer / trigger / judge model: local **Qwen3.5-9B** (llama.cpp `:8400`),
  reasoning ("thinking") disabled to keep answers in budget and tool transcripts clean.
- Trigger and memory summary run on the same local model.
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
answer/support/citation proxies, and token usage (no provider dollar cost — the
pipeline is fully local). StreamRAG separately reports trigger calls, speculation,
evidence lead, reuse, stale work, cancellation, and fallback.

Gold is not read by either service or by the single benchmark runner. The offline
scorer reads it only after predictions are finalized and hashed. Automatic answer
and citation checks are not presented as human semantic accuracy.

## State and concurrency

Each service keeps its generated state under `./var/<role>/`: one embedded Qdrant
index and one SQLite runtime DB per path. SQLite is a library inside each process,
not a separate server.

FastAPI and the calls out to the local model servers are asynchronous; embedded
Qdrant work is moved off the event loop. This supports a responsive local
evaluation; it is not a production concurrency or tenancy claim.

## Security and deployment boundary

The supplied stack is local-only and unauthenticated. Host ports bind to
`127.0.0.1`; the browser uses one frontend origin; the model servers are local and
loopback-only. `.env`, generated indexes, SQLite, sessions, logs, and metrics are
excluded from Git. The fixed corpus and evaluation metadata are intentionally
committed.

Anyone who can reach an API can submit requests, consume local GPU budget, and read
results. A public deployment therefore needs TLS, authentication and authorization,
principal-bound sessions, rate and spend limits, protected index administration,
secret management, backups, retention, abuse monitoring, and load/security tests.
Changing a bind address alone is not a deployment plan.

## References

- [StreamRAG paper](https://arxiv.org/abs/2510.02044)
- [CRAG repository and license](https://github.com/facebookresearch/CRAG)
