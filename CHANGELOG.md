# Changelog

## streamRAG-audio: spoken-query RAG on one A5000 (this repo)

New work layered on the streamrag-local base (imported unchanged):

- **CRAG-TTS-local**: synthesize spoken CRAG queries with Chatterbox, WER-filter with an
  independent ASR (kept 12/15 at WER≤0.10; 3 excluded, recorded). `audio/synth.py`.
- **Streaming ASR + LocalAgreement-2 stabilizer** with char-level `append_only` semantics
  matching `SnapshotAnalyzer`. Measured: raw ASR correction rate 0.774 → 0.084 after
  stabilization (`scoring/audio_quality.py`).
- **3-arm eval** (closed-book / naive / streaming) over the reused coordinator, scored with
  the CRAG Task-1 judge. Hardened after an adversarial audit: clean service isolation,
  true-endpoint commit, counterbalanced arm order, per-query speculation telemetry, paired
  latency stats. Result: retrieval lifts truthfulness +0.42 → +0.92 (incorrect → 0).
- **Honest latency finding (corrected after codex-council round 3):** the paper's
  speculative *retrieval* prefetch landed on 0/24 turns and is not a speedup. An interim
  commit claimed answer-prefill / KV-cache warming as a "−52 %" lever — **retracted**:
  server instrumentation (`cache_n`/`prompt_n`) shows Qwen3.5-9B (hybrid/recurrent
  GatedDeltaNet) gets **no KV prefix reuse** (`reuse_fraction ≈ 0.02`; identical prompts
  fully re-prefill), so there is no prefill-warming lever; the earlier TTFT gap was a
  GPU-scheduling artifact. Net: neither streaming trick reduces TTFT on this stack — a
  useful negative about hybrid/recurrent LLMs. `scoring/prefill_warm.py`, `docs/AUDIO.md`.
- **Fix**: `stream/trigger.py` used the OpenAI Responses API (not implemented by llama.cpp)
  → the stream service could not start in `LOCAL_MODE`; switched to Chat Completions +
  `enable_thinking=false`, mirroring the answer agent.

---

History of the fully-local re-architecture, squashed into this repo's initial
commit. Original per-step commits (from the streamRAG working branch):

## CRAG Task-1 sealed local auto-eval results (n=1325): truthfulness +0.144 [0.119,0.171]


## docs: fully-local mode + CRAG Task-1 eval protocol and claim ceiling


## Add CRAG Task-1 sealed local auto-eval harness

Standard CRAG Task-1 protocol per the reconcile: for each question retrieve only
over ITS OWN <=5 supplied archived pages (no global corpus, no live web), generate
a grounded answer with the local generator, and score with CRAG's truthfulness
structure (correct +1, missing/abstain 0, incorrect -1) -- accuracy, hallucination,
missing, truthfulness (C-I)/N with 95% bootstrap CIs, sliced by question type and
static/dynamic. Judge is automatic string match (scorer's normalize/_contains_phrase)
plus a pinned local LLM judge for the rest.

Deliberately a CRAG-STYLE LOCAL auto-eval, NOT an official CRAG/KDD score (local
judge, not the paper's hosted GPT judge). Runs against the local servers directly
(generator :8400, embedder :8401), deterministic (temp 0), one generation per item,
no retries. Sealed set = the 1,325 public-test questions NOT among the 10 curated
items already inspected (provenance from selection_manifest: those 10 are split=1).

## local_eval: satisfy ruff (imports, no one-line statements, naming)


## P3 error-gate: local eval harness + fix placeholder-citation prompt bug

Add comparison/local_eval.py: run the fully-local pipeline over the CRAG test
questions and report answer-match (mirroring the scorer's normalize/contains) +
citation quality. This is the base-model audit the reconcile requires before
committing to LoRA.

First run exposed a clean deficit: 7/10 answers cited the literal placeholder
[doc-id::c0001] -- the model was copying the instruction's EXAMPLE marker rather
than the real bracket id printed before each evidence passage. That is a prompt
bug, not a capability gap. Rewrote the citation instruction to say "copy the
bracketed id printed at the start of the evidence passage verbatim; never use a
placeholder like [doc-id::...]".

Result on the 10 test questions (local Qwen3.5-9B Q4_K_M + bge-large):
- placeholder citations 7/10 -> 0/10 (fixed)
- real chunk-id citations 1/10 -> 5/10
- answer-match steady at 7/10 (70%; hosted baseline was 100%)

Error-gate verdict: the base local model is already reasonable and the biggest
deficit was fixable cheaply, so fine-tuning becomes a targeted improvement
(citation coverage, grounded-answer consistency), not a rescue -- to be decided
against measured gains, not assumed.

## P1b: multi-turn contextual retrieval (resolve follow-up pronouns/ellipsis)

Follow-ups like "where did he work before?" embedded poorly because the entity
lived only in prior turns, so retrieval missed. Add contextual_retrieval_query:
lead with the current question, append a bounded tail of compact conversational
state so the embedder resolves pronouns/ellipsis. Query text only -- prior
retrieved passages stay turn-local (never carried forward), per the reconcile.

Plumbed symmetrically so the Naive vs StreamRAG comparison stays fair:
- shared/query.py: contextual_retrieval_query (first turns == bounded_retrieval_query).
- RagPath.commit protocol + naive/path.py + stream/path.py gain conversation_context.
- shared/api/runtime._run_path fetches agent.conversation_context and passes it.
- stream/coordinator.py: committed query (L288) and speculative candidate (L442)
  both use it with the coordinator's pre-Send context, so exact-reuse validation
  (evidence.query == committed query) still holds when the draft equals the commit.

Verified end-to-end on the fully-local stack: T1 "who founded salesforce?" ->
Benioff; T2 "where did HE work before?" -> Oracle (correct pronoun resolution),
whereas the bare follow-up would not retrieve the Oracle evidence.

## Local mode: swap gpt-5.6-sol -> local Qwen3.5-9B + bge embeddings (P1a+P2)

Make the pipeline run fully on one box (no hosted API), proven end-to-end.

Generator:
- openai_client.py: add chat_model() building an OpenAI-compatible Chat
  Completions model. In LOCAL_MODE it targets a local llama.cpp server
  (Qwen3.5-9B) at LLM_BASE_URL; hosted mode targets OpenAI chat completions.
  Keep responses_model() for the frozen baseline.
- service.py / summary_skill.py: migrate OpenAIResponsesModel(+Settings) ->
  OpenAIChatModel(+Settings). Drop Responses-only knobs (reasoning_effort,
  service_tier, prompt_cache_key, store, verbosity). Qwen3.5 is a reasoning
  model, so disable thinking via extra_body chat_template_kwargs
  {enable_thinking:false} to keep answers in budget and tool transcripts clean.

Token caps (the pipeline hardcoded 600/320, too small once a local thinker
spends tokens reasoning): new configurable ANSWER_MAX_TOKENS (2048) and
SUMMARY_MAX_TOKENS (512), wired through model settings and UsageLimits.

Embeddings:
- embeddings.py: OpenAIEmbedder accepts base_url/api_key for a local embedding
  server (llama.cpp --embedding serving bge-large-en-v1.5, 1024-dim).
- factory.py: pass the local embedding endpoint in LOCAL_MODE.

Config: add LOCAL_MODE, LLM_BASE_URL/API_KEY, EMBEDDING_BASE_URL,
DISABLE_THINKING; default openai_model=qwen3.5-9b-local, embedding=bge-large,
dims=1024. validate() relaxes the hosted-identity locks in local mode and
checks the local serving contract instead. .env.example documents it.

Verified: naive service against local models indexed 1,514 chunks (256-token
chunks, since bge caps at 512 tokens) and answered a CRAG test question
correctly and grounded ("...Marc Benioff previously worked at Oracle...") with
tool-calling wired. Known gap for the error-gate: base model sometimes cites
the literal example marker instead of the real chunk id.

