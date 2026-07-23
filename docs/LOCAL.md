# Fully-local serving (single RTX A5000, no hosted API)

This repo runs the entire StreamRAG stack on one machine with no OpenAI
dependency: the answer generator and the embedder are local OpenAI-compatible
servers (llama.cpp), and everything else (Qdrant, SQLite) is unchanged. There is
**one** pipeline here — local — with no hosted toggle. The OpenAI-compatible
protocol is still used (the `openai` SDK / `OpenAIChatModel` / `OpenAIEmbedder`),
because llama.cpp speaks that protocol; the base URLs point at the on-box servers.

This is a **local backend**, not a claim of parity with any hosted model.

## Serving choices

| Piece | This repo (fully local) |
|---|---|
| Generator | **Qwen3.5-9B** Q4_K_M via llama.cpp, OpenAI **Chat Completions** (`OpenAIChatModel`) — answer, trigger, and judge |
| Embeddings | **bge-large-en-v1.5** (1024-d) via llama.cpp `--embedding` |
| Thinking | Qwen3.5 is a reasoning model; disabled (`chat_template_kwargs.enable_thinking=false`) to keep answers in budget and tool transcripts clean |
| Token cap | configurable `ANSWER_MAX_TOKENS` (2048) / `SUMMARY_MAX_TOKENS` (512) — sized for the visible answer (plus the reasoning trace if thinking is enabled) |
| Chunking | 256 / 32 (bge caps at 512 tokens; 400-token chunks overflow its tokenizer) |
| Multi-turn | **contextual retrieval**: follow-up pronouns/ellipsis resolved from compact conversational state before retrieval (query text only; evidence stays turn-local), applied symmetrically to both paths |

## Serving

Two llama.cpp servers on the A5000:

```bash
# generator (tool-calling via --jinja; multiple slots for throughput)
llama-server -m qwen3.5-9b-Q4_K_M.gguf --port 8400 -ngl 99 -fa on \
  -np 8 -cb --ctx-size 24576 --jinja --alias qwen3.5-9b-local

# embedder
llama-server -m bge-large-en-v1.5-f16.gguf --port 8401 -ngl 99 \
  --embedding --pooling mean -c 512 --alias bge-large-en-v1.5
```

Then point the services at them (`.env`):

```
LLM_BASE_URL=http://127.0.0.1:8400/v1
EMBEDDING_BASE_URL=http://127.0.0.1:8401/v1
EMBEDDING_DIMENSIONS=1024
CHUNK_TOKENS=256
CHUNK_OVERLAP=32
```

Start a service and build its index with `POST /v1/data/sync`. The Qdrant
collection is dimension- and chunker-specific, so a fresh sync is required if the
embedding model or chunking ever changes.

Serving precision is currently `Q4_K_M`. Qwen3.5's recurrent (GatedDeltaNet)
layers are quantization-sensitive; a precision ladder (bf16 → Q8_0 → Q6_K → Q4)
is the recommended next step before treating any single quant as final.

## Evaluation: CRAG Task-1 local auto-eval

`comparison/crag_task1_eval.py` runs the **standard CRAG Task-1 protocol** — for
each question, retrieve only over its own ≤5 supplied archived pages (no global
corpus, no live web), generate once (deterministic, temperature 0), and score
with CRAG's truthfulness structure: **correct +1, missing/abstain 0, incorrect
−1**, reporting accuracy, hallucination, missing, and truthfulness `(C−I)/N` with
95 % bootstrap CIs, sliced by the eight question types and the four dynamism
classes. Citation coverage/validity is reported **separately** (a citation is not
proof of grounding).

Provenance: the 10 curated `test` items ship from CRAG **public test** (split 1),
so the **sealed** set is the **1,325 public-test questions excluding those 10**; a
secondary all-1,335 number is disclosed as containing the 10 seen items.

### Honest claim ceiling

This is a **CRAG-style local auto-eval**, not an official CRAG/KDD leaderboard
score (the judge is a pinned local model, not the paper's hosted GPT judge). It
does **not** claim: large-model quality parity; that fine-tuning is
ineffective/unnecessary in general; unseen generalization beyond this split;
"full Task 2" (the mock KG/API path is not implemented); or production readiness.

The base-model error-gate (`comparison/local_eval.py`) found the largest early
deficit — placeholder citations — was a **prompt bug**, fixed cheaply, so LoRA
fine-tuning was **deferred** as not-yet-justified rather than assumed. Cheaper
levers (citation few-shot, top-k/reranking, precision ladder) come first.

CRAG data is CC BY-NC 4.0 (non-commercial). The full upstream dataset is not
vendored here; a small CRAG-derived subset (`data/crag_eval/`) is committed for
non-commercial research use — see `NOTICE`.
