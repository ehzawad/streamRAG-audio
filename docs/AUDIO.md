# streamRAG-audio — spoken CRAG on one A5000

A single-A5000 **cascaded emulation of the *Model-Triggered Streaming RAG control
policy*** from *Stream RAG: Instant and Accurate Spoken Dialogue Systems with
Streaming Tool Usage* (Arora et al., Meta + CMU, [arXiv:2510.02044](https://arxiv.org/abs/2510.02044)),
applied to spoken CRAG queries. It reuses this author's own prior work:
[streamRAG](https://github.com/ehzawad/streamRAG) (typed streaming RAG),
[streamrag-local](https://github.com/ehzawad/streamrag-local) (its fully-local
variant; this repo's base), and the Silero-VAD turn/barge-in logic from
[omni-voice-lab](https://github.com/ehzawad/omni-voice-lab).

## What this is — and is NOT

**Is:** a cascade — spoken query → (VAD endpoint) → streaming ASR partials →
LocalAgreement stabilizer → the reused `StreamCoordinator` trigger fires
*speculative* retrieval over a local CRAG corpus → commit at endpoint → grounded
answer from local **Qwen3.5-9B (llama.cpp)** → TTS. The paper explicitly notes its
Model-Triggered method is **modality-agnostic and applies to typed input**; this
implements that control policy over an ASR front-end.

**Is NOT** (do not read these into any number here):

- NOT the paper's **end-to-end speech-in/speech-out** model, its **audio-conditioned**
  trigger, its **joint response post-training**, or its **web + knowledge-graph** tools.
- NOT the paper's **AudioCRAG** data. AudioCRAG is unreleased (0 hits on HF/GitHub as
  of this writing); we synthesize our own spoken queries (`CRAG-TTS-local`, below).
- NOT a **bit-exact** copy of the paper's control loop. The reused `StreamCoordinator`
  **revalidates evidence at commit** and falls back unless speculative evidence matches
  the committed text — *safer but different* from the paper, which trusts the latest
  tool result and drops the reflector.
- Absolute numbers are **not comparable** to the paper (local Qdrant over 250 curated
  pages ≠ the paper's 100K-document web+KG pipeline; tiny n; synthetic single-voice TTS).

## Pipeline & reuse

| Stage | Component | Reuse |
|---|---|---|
| Endpoint / barge-in | Silero VAD (`hervoice/live`) | copied for the **unscored** live demo only |
| Streaming ASR | faster-whisper `base.en` int8 (CPU), re-decode 500 ms prefixes | `audio/asr_partials.py` (new) |
| Stabilizer | LocalAgreement-2, char-level `append_only` matching `SnapshotAnalyzer` | `audio/stabilizer.py` (new) |
| Trigger / speculation | `stream/trigger.py::ModelTrigger` + `stream/coordinator.py` | **imported unchanged** |
| Retrieval | bge-large-en-v1.5 (GGUF, :8401) + Qdrant | imported |
| Answer | Qwen3.5-9B Q4_K_M (llama.cpp, :8400), grounded agent | imported |
| Query synthesis / TTS-out | Chatterbox-TTS (offline, GPU sole-tenant) | `audio/synth.py` (new) |

**Local-mode fix made here:** `stream/trigger.py` called `responses_model()` (OpenAI
*Responses* API), which llama.cpp does not implement — so the stream service could not
start locally. Patched to use `chat_model()` (Chat Completions + `enable_thinking=false`)
in `local_mode`, mirroring the answer agent. (The same bug exists upstream in
streamrag-local; left untouched there.)

## CRAG-TTS-local (the spoken set)

Paper-shaped construction (§9 of the paper): synthesize each CRAG Task-1 question with
a frozen voice/seed (Chatterbox), re-ASR with an independent recognizer (faster-whisper),
and keep only items at WER ≤ 0.10. **Kept 12/15** (9 test + 3 dev); the 3 excluded are
recorded in the manifest with their WER, not silently dropped. Synthesized wavs are a
CRAG derivative (CC BY-NC 4.0, **non-commercial**) — gitignored, regenerate locally with
`audio/synth.py`.

## Two measurements

### 1. The ASR-churn gate (`scoring/audio_quality.py`)

Raw cumulative ASR revises constantly: **mean raw correction rate 0.774** — most deltas
are non-`append_only` and would fire `_cancel_for_correction`, collapsing streaming to
endpoint-only. LocalAgreement-2 cuts this to **0.084** (endpoint WER 0.013). This proves
the stabilizer is *necessary* and makes partials mostly append-only. It does **NOT** prove
speculative evidence survives the coordinator's commit gate — that is measured separately.

### 2. Three-arm eval (`harness/run_three_arm.py`)

Identical audio → identical endpoint ASR transcript for all arms; they differ only in
*when/whether* retrieval fires. Scored with the reused CRAG Task-1 judge (truthfulness
`(C−I)/N`, bootstrap CI). Hardened after an adversarial audit: clean service isolation
(unique per-run state, verified child, refuse a busy port), commit at the **true audio
endpoint**, **counterbalanced** arm order across reps (llama.cpp prompt-cache warming
otherwise favors whichever arm runs second), per-query **speculation telemetry read-back**,
and **paired** latency statistics (never difference-of-independent-medians).

**Accuracy — the real, robust win** (`runs/three_arm.json`, 2 counterbalanced reps):

| Arm | truthfulness (C−I)/N | accuracy | observed-incorrect |
|---|---|---|---|
| closed-book (no retrieval) | **+0.417** | 0.667 | 0.250 |
| naive-audio-RAG | **+0.917** | 0.917 | 0.000 |
| streaming-audio-RAG | **+0.917** | 0.917 | 0.000 |

Retrieval lifts truthfulness +0.42 → +0.92 and drives observed-incorrect labels to 0
(test-only +0.44 → +0.89; dev +0.33 → +1.00). This holds up; it is the point.

**Latency — where "fast" actually is** (honest, after an adversarial audit):

- The paper's speculative **retrieval** prefetch is a **measured null here**:
  `accepted_ready_before_commit` on **0/24** turns; with counterbalanced arm order the
  streaming arm is in fact **+898 ms slower** (paired median) — trigger overhead with no
  prefetch payoff. Retrieval (~200 ms) is simply not the bottleneck.
- The dominant cost is the **9B answer prefill (TTFT)** — and it is **not reducible by
  cache-warming here.** An earlier version of this repo claimed a "−52 % TTFT from
  answer-prefill / KV-cache warming" fast lever; **that was wrong and is retracted.**
  Instrumenting the server (`cache_n` / `prompt_n`, `runs/prefill_warm.json`) shows the KV
  prefix cache **does not reuse** for Qwen3.5-9B: even an *identical* prompt re-evaluates
  ~all tokens (`reuse_fraction ≈ 0.02`). The cause is architectural — **Qwen3.5-9B is
  hybrid/recurrent (GatedDeltaNet); its per-sequence recurrent state can't be partially
  reused**, so llama.cpp rolls the reusable prefix back to ~0. The cold-vs-repeat wall-clock
  TTFT gap I first measured was **GPU scheduling / first-after-idle warmup** (identical
  prefill compute both times), not cache reuse, and not deployable.

**Architecture conclusion:** keep the local streaming-RAG for **accuracy** (that is the real
win). On latency, be honest: on this stack, **neither** speculative retrieval-prefetch
**nor** answer-prefill warming reduces TTFT — the system runs at the 9B's cold
prefill+decode latency. The genuinely useful (negative) finding: **a hybrid/recurrent LLM
defeats the standard KV-cache-warming latency trick.** The speculative-retrieval coordinator
is imported and exercised (its commit gate keeps it safe), but reported here as
latency-neutral-to-negative, not a speedup.

## Honest ceiling

- Tiny n (9 test + 3 dev), synthetic single-voice TTS — directional at best; wide CIs.
- The judge marks any answer containing a gold phrase correct without a contradiction
  check → report "observed-incorrect labels", not "hallucination".
- closed-book is an **illustrative parametric baseline**, not a strict retrieval ablation
  (different answer stack/token budget).
- Latency: report the **paired** distribution and the **speculation-landed rate**; claim a
  streaming benefit only if speculation actually landed before commit.

## Reproduce

```bash
# 0. servers on the A5000 (:8400 Qwen3.5-9B, :8401 bge-large)
bash harness/serve_local.sh
# 1. synth CRAG-TTS-local (GPU sole-tenant, .venv-modular)
CUDA_VISIBLE_DEVICES=0 .venv-modular/bin/python audio/synth.py
# 2. ASR-churn gate + traces (CPU)
CUDA_VISIBLE_DEVICES="" .venv-modular/bin/python scoring/audio_quality.py
CUDA_VISIBLE_DEVICES="" .venv-modular/bin/python audio/build_traces.py
# 3. 3-arm eval (py3.14 RAG venv)
PYTHONPATH=. .venv/bin/python harness/run_three_arm.py
```
