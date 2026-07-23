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

**Is:** a cascade — spoken CRAG query is synthesized offline (Qwen3-TTS) →
transcribed by faster-whisper `base.en` → the transcript feeds the reused typed
streaming-RAG core (`StreamCoordinator` trigger fires *speculative* retrieval over a
local CRAG corpus → commit → grounded answer from local **Qwen3.5-9B (llama.cpp)**).
The paper explicitly notes its Model-Triggered method is **modality-agnostic and
applies to typed input**; this implements that control policy behind an ASR
front-end. (An earlier Chatterbox-era live demo added a VAD endpoint + streaming
ASR-partials + a LocalAgreement stabilizer; that demo front-end has been retired —
see *Retired / rejected experiments* below — leaving one clean offline-synth → ASR →
RAG evaluation pipeline.)

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
  pages ≠ the paper's 100K-document web+KG pipeline; 12 question clusters; synthetic TTS —
  a single-voice Chatterbox baseline and a 9-voice Qwen3-TTS headline set).

## Pipeline & reuse

| Stage | Component | Reuse |
|---|---|---|
| Query synthesis | Qwen3-TTS CustomVoice, 9 voices (offline, GPU sole-tenant) | `audio/synth_qwen.py` (new) |
| ASR | faster-whisper `base.en` int8 (CPU) | `scoring/asr_multivoice.py` (new) |
| Trigger / speculation | `stream/trigger.py::ModelTrigger` + `stream/coordinator.py` | **imported unchanged** |
| Retrieval | bge-large-en-v1.5 (GGUF, :8401) + Qdrant | imported |
| Answer / judge | Qwen3.5-9B Q4_K_M (llama.cpp, :8400), grounded agent | imported |

**Local Chat-Completions fix made here:** `stream/trigger.py` originally targeted the
OpenAI *Responses* API, which llama.cpp does not implement — so the stream service could
not start locally. Patched to use `chat_model()` (Chat Completions +
`enable_thinking=false`), mirroring the answer agent, so the trigger runs entirely on the
local :8400 server. (The same issue exists upstream in streamrag-local; left untouched
there.) The one-clip `make smoke-9b` target exercises exactly this path end-to-end.

## Spoken sets: `CRAG-TTS-local` (prior baseline) → Qwen3-TTS 9-voice (headline)

**`CRAG-TTS-local` (prior baseline, Chatterbox — historical)** — paper-shaped
construction (§9): each CRAG Task-1 question was synthesized with a frozen voice/seed
(**Chatterbox**, one voice), re-ASR'd with an independent recognizer (faster-whisper), and
kept at WER ≤ 0.10 (12/15; the 3 excluded recorded with their WER, not silently dropped).
The Chatterbox synth script has been removed; this baseline survives only as its **frozen
result** `runs/three_arm.json` (not re-run).

**Qwen3-TTS 9-voice set (current headline, `audio/synth_qwen.py`)** — the same 12 kept
questions rendered in **all 9 Qwen3-TTS CustomVoice timbres** (`aiden, dylan, eric, ono_anna,
ryan, serena, sohee, uncle_fu, vivian`) with a fixed neutral style → **108 clips = 12 question
clusters × 9 voices**. Qwen3-TTS is Apache-2.0; the rendered audio is a CRAG derivative
(CC BY-NC, non-commercial, gitignored). Per-voice mean WER 0.017–0.094.

**Headline result** (`runs/multivoice.json`, closed-book vs naive-RAG, truthfulness `(C−I)/N`
per the *reused automatic judge*):

> Across 12 CRAG questions rendered in each of nine **tested** Qwen3-TTS synthetic voices
> (108 clips; **12 independent question clusters**), the retrieval-enabled system scored
> **+0.898** vs **+0.389** for an illustrative closed-book baseline — a descriptive paired gap
> of **+0.509** (question-clustered 95 % bootstrap CI **[0.20, 0.85]**). The observed gap was
> **positive for every tested voice** (+0.333 … +0.750); the WER ≤ 0.10 sensitivity subset was
> similar (**+0.465**, CI [0.14, 0.84], 89 clips). This shows **consistency across these tested
> synthetic timbres** — **not** 108 independent samples, unseen-voice or human/demographic
> robustness, or a retrieval-only causal effect (both arms get the same transcript, but the
> closed-book arm uses a different answer stack).

CIs are **question-clustered** bootstraps (resampling the 12 questions as intact units), never
n=108 iid; `multivoice.json` retains per-question rows so the clustering is auditable.

## Frozen 3-arm baseline (accuracy & latency)

The prior single-voice Chatterbox run's three-arm eval is retained as **frozen evidence**
(`runs/three_arm.json`); its driver has been retired, but its isolation / arm / scorer
primitives live on in `harness/eval_common.py` (reused by the 9-voice headline). Identical
audio → identical endpoint ASR transcript for all arms; they differ only in *when/whether*
retrieval fires. Scored with the reused CRAG Task-1 judge (truthfulness `(C−I)/N`, bootstrap
CI). It was hardened after an adversarial audit: clean service isolation (unique per-run
state, verified child, refuse a busy port), commit at the **true audio endpoint**,
**counterbalanced** arm order across reps (llama.cpp prompt-cache warming otherwise favors
whichever arm runs second), per-query **speculation telemetry read-back**, and **paired**
latency statistics (never difference-of-independent-medians).

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
latency-neutral-to-negative, not a speedup. **The speculative-retrieval null (0/24, paired
median +898.5 ms) was measured on the prior *Chatterbox* run only — it is NOT voice-independent
(whether speculation lands depends on voice-conditioned ASR timing) and was not re-run per Qwen
voice. The prefill no-KV-reuse null, by contrast, is a Qwen3.5/llama.cpp property.**

## Honest ceiling

- **12 independent question clusters** (× 9 voices = 108 clips) — small; the question-clustered
  95 % CI is wide ([0.20, 0.85]) though it excludes 0. Do **not** read 108 as an iid n, treat
  9/9 voices as independent replications, or infer to unseen voices / human / demographic robustness.
- Truthfulness is **per the reused automatic judge** (the same `:8400` model answers *and* judges;
  the scorer auto-accepts gold-phrase containment) — "observed-incorrect labels", not "hallucination".
- closed-book is an **illustrative parametric baseline**, not a strict retrieval ablation
  (different answer stack/token budget) → "retrieval-enabled *system* vs baseline", not causal.
- Latency: report the **paired** distribution and the **speculation-landed rate**; claim a
  streaming benefit only if speculation actually landed before commit (it did not).

## Retired / rejected experiments

One clean local pipeline is kept; the following were tried and cut, recorded here so the
history stays honest. None is part of the live pipeline or the retained result bundle
(`runs/{multivoice,three_arm,prefill_warm}.json`).

| Experiment | What was measured | Verdict |
|---|---|---|
| **Qwen3.6-27B** answer/judge upgrade | RAG truthfulness 0.907; retrieval gap **+0.269**, question-clustered 95 % CI **[0.00, 0.57]**, positive on **7/9** voices | **Reverted to 9B** — the larger model *weakened* the retrieval headline (9B: gap +0.51, 9/9, CI excludes 0) and the CI now touched 0. |
| **Nemotron-3.5** ASR (vs faster-whisper `base.en`) | WER **0.076** vs base.en **0.057**; `large-v3-turbo` tied base.en | **Rejected** — failed the WER gate; kept `base.en`. No ASR change. |
| **Pre-Send answering** (speculative answer before commit) | ready-before-commit rate below the **≥ 80 %** gate on audio | **Removed** (gated experiment) — it never cleared the readiness gate on spoken input. |
| Chatterbox streaming front-end (VAD + ASR-partials + LocalAgreement-2 stabilizer) + ASR-churn gate | stabilizer cut the raw ASR correction rate 0.774 → 0.084 (endpoint WER 0.013) | **Retired** with the Chatterbox live demo; the offline-synth → ASR → RAG evaluation pipeline does not need it. Its result JSON is not retained. |

## Reproduce

```bash
# 0. servers on the A5000 (:8400 Qwen3.5-9B, :8401 bge-large)
bash harness/serve_local.sh
# 1. Qwen3-TTS 9-voice synth (GPU sole-tenant; qwen-audio venv)
CUDA_VISIBLE_DEVICES=0 <qwen-audio-venv>/bin/python audio/synth_qwen.py
# 2. faster-whisper base.en ASR + WER (CPU; faster-whisper venv)
CUDA_VISIBLE_DEVICES="" <fw-venv>/bin/python scoring/asr_multivoice.py
# 3. 9-voice headline eval (py3.14 RAG venv)
PYTHONPATH=. <rag-venv>/bin/python harness/run_multivoice.py
# 4. KV-reuse probe (honest negative)
python scoring/prefill_warm.py
# 5. one-clip end-to-end smoke (chains the ASR venv + the RAG venv)
make smoke-9b
```
