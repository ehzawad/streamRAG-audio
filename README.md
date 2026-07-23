# streamRAG-audio

A single-**A5000** exploration of **spoken-query RAG**: a cascaded emulation of the
*Model-Triggered Streaming RAG control policy* from *Stream RAG* (Arora et al.,
Meta + CMU, [arXiv:2510.02044](https://arxiv.org/abs/2510.02044)), run fully locally
over CRAG questions with **Qwen3.5-9B (llama.cpp)** + **bge-large** + **Qdrant**, plus a
faster-whisper front-end and **Qwen3-TTS** (9 voices) / Chatterbox query synthesis.

Built on this author's own [streamrag-local](https://github.com/ehzawad/streamrag-local)
(local variant of [streamRAG](https://github.com/ehzawad/streamRAG)); the streaming
audio/turn logic is reused from [omni-voice-lab](https://github.com/ehzawad/omni-voice-lab).
Both source repos are left untouched.

## The honest headline

This started as a faithful port of the paper's speculative streaming and became an
**evidence-driven** exercise in finding where local latency actually is. Two findings,
both backed by committed result JSON under `runs/`:

- **Retrieval is the real win (accuracy), and it holds across voices.** On spoken CRAG
  queries the retrieval-enabled system beats an illustrative closed-book baseline on
  automatically-judged truthfulness `(C−I)/N`. Headline: the 12 questions rendered in **all 9
  Qwen3-TTS voices** (108 clips = **12 question clusters**) → **+0.90 vs +0.39, paired gap
  +0.51** (question-clustered 95 % CI **[0.20, 0.85]**), **positive on 9/9 voices**
  (`runs/multivoice.json`). The prior single-voice Chatterbox run agrees (+0.92 vs +0.42;
  `runs/three_arm.json`). Consistency across *tested* synthetic timbres — **not** unseen-voice
  or human/demographic robustness, and not a retrieval-only causal effect. Fully local.
- **Neither streaming trick makes it faster here, and I can prove why.** The paper's
  speculative *retrieval* prefetch landed on **0/24** turns (fairly measured, *slower*).
  And **answer-prefill / KV-cache warming does not work either**: instrumenting the server
  (`cache_n`) shows Qwen3.5-9B — a **hybrid/recurrent (GatedDeltaNet)** model — gets
  **no KV prefix reuse** (`reuse_fraction ≈ 0.02`; identical prompts fully re-prefill).
  An earlier commit wrongly claimed a "−52 % prefill-warming" lever; a codex-council audit +
  cache instrumentation caught it — **retracted**. The honest, useful negative:
  *a hybrid/recurrent LLM defeats the standard KV-cache-warming latency trick.*
  (`runs/prefill_warm.json`)

It is **not** the paper's end-to-end speech model, audio-conditioned trigger, joint
post-training, or web+KG tools; and the audio is **`CRAG-TTS-local`** (our own synthesis —
the paper's AudioCRAG is unreleased), not the paper's data. Absolute numbers are **not**
comparable to the paper. Full scope, mapping, and ceilings: **[docs/AUDIO.md](docs/AUDIO.md)**.

## Reproduce (one A5000)

```bash
bash harness/serve_local.sh                                  # :8400 Qwen3.5-9B, :8401 bge
CUDA_VISIBLE_DEVICES=0  .venv-modular/bin/python audio/synth.py          # CRAG-TTS-local
CUDA_VISIBLE_DEVICES="" .venv-modular/bin/python scoring/audio_quality.py # ASR-churn gate
CUDA_VISIBLE_DEVICES="" .venv-modular/bin/python audio/build_traces.py    # stabilized traces
PYTHONPATH=. .venv/bin/python harness/run_three_arm.py       # 3-arm (prior single-voice baseline)
CUDA_VISIBLE_DEVICES=0  .venv-qwen-audio/bin/python audio/synth_qwen.py    # Qwen3-TTS 9-voice set
CUDA_VISIBLE_DEVICES="" .venv-modular/bin/python scoring/asr_multivoice.py
PYTHONPATH=. .venv/bin/python harness/run_multivoice.py      # 9-voice headline (per-voice + macro)
python scoring/prefill_warm.py                               # KV-reuse probe (honest negative)
```

The underlying local RAG app (naive/stream services, frontend, CRAG Task-1 eval) is
inherited from streamrag-local; see `docs/LOCAL.md`. Note: the stream service's trigger
was fixed here to use Chat Completions in `LOCAL_MODE` (it called the OpenAI Responses
API, which llama.cpp does not implement).

## License & data

MIT (code). Evaluation uses **CRAG** (CC BY-NC 4.0, **non-commercial**); a small
CRAG-derived subset is included under `data/crag_eval/`, and synthesized audio is a
non-commercial CRAG derivative (gitignored, regenerate locally). See **NOTICE**.
Commits authored by ehzawad. No affiliation with the paper's authors, Meta, or CMU.
