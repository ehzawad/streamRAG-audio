#!/usr/bin/env bash
# Bring up the two llama.cpp servers on the A5000 for streamRAG-audio's 3-arm eval:
#   :8400  Qwen3.5-9B Q4_K_M  (answer + trigger LLM, tool-calling via --jinja)
#   :8401  bge-large-en-v1.5  (embeddings, mean pooling)
# Both on GPU0 (A5000) by PCI order; the A6000 co-tenant is left alone.
# Commands mirror docs/LOCAL.md exactly. Logs -> runs/serve/.
set -euo pipefail

LLAMA="${LLAMA_SERVER:-/mnt/sdb/arafat/ehz/llm-stuff/qwen35-gguf-bench/llama.cpp/build/bin/llama-server}"
MODELS="${MODELS_DIR:-/mnt/sdb/arafat/ehz/llm-stuff/qwen35-gguf-bench/models}"
BRAIN="${BRAIN_GGUF:-$MODELS/q9b/Qwen3.5-9B-Q4_K_M.gguf}"
EMBED="${EMBED_GGUF:-/mnt/sdb/arafat/ehz/llm/qwen3-lora-gguf-bench/models/embed/bge-large-en-v1.5-f16.gguf}"
LOGDIR="$(cd "$(dirname "$0")/.." && pwd)/runs/serve"
mkdir -p "$LOGDIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
# the binary's shared libs sit beside it but there is no $ORIGIN rpath
export LD_LIBRARY_PATH="$(dirname "$LLAMA")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[serve] LLM  :8400  $BRAIN"
nohup "$LLAMA" -m "$BRAIN" --port 8400 -ngl 99 -fa on \
  -np 8 -cb --ctx-size 24576 --jinja --alias qwen3.5-9b-local \
  > "$LOGDIR/llm-8400.log" 2>&1 &
echo $! > "$LOGDIR/llm.pid"

echo "[serve] EMB  :8401  $EMBED"
nohup "$LLAMA" -m "$EMBED" --port 8401 -ngl 99 \
  --embedding --pooling mean -c 512 --alias bge-large-en-v1.5 \
  > "$LOGDIR/emb-8401.log" 2>&1 &
echo $! > "$LOGDIR/emb.pid"

echo "[serve] waiting for health ..."
for port in 8400 8401; do
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "[serve] :$port healthy"; break
    fi
    sleep 2
    [ "$i" = 120 ] && { echo "[serve] :$port FAILED to come up"; exit 1; }
  done
done
echo "[serve] both servers up."
