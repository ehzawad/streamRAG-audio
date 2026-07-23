NAIVE_BASE_URL ?= http://localhost:8001
STREAM_BASE_URL ?= http://localhost:8002
APP_STATE_ROOT ?= var/dev-services
BENCH_EVALUATION_DIR ?= data/crag_eval
BENCH_STATE_ROOT ?= comparison/benchmark/results/services
BENCH_PREDICTIONS ?= comparison/benchmark/results/predictions.jsonl
BENCH_SUMMARY ?= comparison/benchmark/results/summary.json
BENCH_QUERY_LIMIT ?= 10
BENCH_CASE_TIMEOUT_S ?= 45
BENCH_POST_TYPING_DWELL_MS ?= 5000

# smoke-9b: two-venv end-to-end smoke over one clip against a RUNNING :8400/:8401.
# ASR needs faster-whisper (.venv-modular); the RAG client needs the py3.14 rag venv.
SMOKE_FW_PYBIN ?= /mnt/sdb/arafat/ehz/hervoice/.venv-modular/bin/python
SMOKE_RAG_PYBIN ?= /mnt/sdb/arafat/ehz/llm/streamRAG/.venv/bin/python
SMOKE_HANDOFF ?= runs/smoke_9b_transcript.json

.PHONY: setup setup-python setup-frontend dev-naive dev-stream dev-frontend \
	check check-shared check-naive check-stream check-comparison \
	check-frontend build sync-naive sync-stream \
	benchmark-services-check benchmark-services-sync benchmark-services-serve \
	benchmark score smoke-9b

setup: setup-python setup-frontend

setup-python:
	uv sync --frozen --python 3.14

setup-frontend:
	cd frontend && npm ci

dev-naive:
	mkdir -p $(APP_STATE_ROOT)/naive
	ALLOW_UNREVIEWED_DATASET=1 QDRANT_PATH=$(APP_STATE_ROOT)/naive/qdrant \
		RUNTIME_DB=$(APP_STATE_ROOT)/naive/runtime.sqlite3 \
		METRICS_LOG=$(APP_STATE_ROOT)/naive/requests.jsonl \
		uv run uvicorn naive.api:app --reload --host 127.0.0.1 --port 8001

dev-stream:
	mkdir -p $(APP_STATE_ROOT)/stream
	ALLOW_UNREVIEWED_DATASET=1 QDRANT_PATH=$(APP_STATE_ROOT)/stream/qdrant \
		RUNTIME_DB=$(APP_STATE_ROOT)/stream/runtime.sqlite3 \
		METRICS_LOG=$(APP_STATE_ROOT)/stream/requests.jsonl \
		uv run uvicorn stream.api:app --reload --host 127.0.0.1 --port 8002

dev-frontend:
	cd frontend && npm run dev -- --host 127.0.0.1

check-shared:
	uv run ruff check shared

check-naive:
	uv run ruff check shared naive

check-stream:
	uv run ruff check shared stream

check-comparison:
	uv run ruff check comparison

check-frontend:
	cd frontend && npm run build

check:
	uv run ruff check shared naive stream comparison
	cd frontend && npm run build

build:
	uv build
	cd frontend && npm run build

sync-naive:
	curl --fail --show-error --request POST $(NAIVE_BASE_URL)/v1/data/sync

sync-stream:
	curl --fail --show-error --request POST $(STREAM_BASE_URL)/v1/data/sync

benchmark-services-check:
	uv run python -m comparison.services check \
		--dataset-dir $(BENCH_EVALUATION_DIR) --state-root $(BENCH_STATE_ROOT)

benchmark-services-sync:
	uv run python -m comparison.services sync \
		--dataset-dir $(BENCH_EVALUATION_DIR) --state-root $(BENCH_STATE_ROOT)

benchmark-services-serve:
	uv run python -m comparison.services serve \
		--dataset-dir $(BENCH_EVALUATION_DIR) --state-root $(BENCH_STATE_ROOT)

benchmark:
	uv run python -m comparison.benchmark.run_benchmark \
		--repetitions 1 --query-limit $(BENCH_QUERY_LIMIT) \
		--wpm 70 --post-typing-dwell-ms $(BENCH_POST_TYPING_DWELL_MS) \
		--case-timeout-s $(BENCH_CASE_TIMEOUT_S) --max-typing-drift-ms 100 \
		--queries $(BENCH_EVALUATION_DIR)/test_queries.jsonl \
		--naive-base-url $(NAIVE_BASE_URL) --stream-base-url $(STREAM_BASE_URL) \
		--output $(BENCH_PREDICTIONS)

score:
	uv run python -m comparison.benchmark.score \
		--gold $(BENCH_EVALUATION_DIR)/test_gold.jsonl \
		--evaluation-manifest $(BENCH_EVALUATION_DIR)/checksums.sha256 \
		--predictions $(BENCH_PREDICTIONS) --output $(BENCH_SUMMARY)

# One-clip end-to-end smoke of the clean local 9B pipeline. Requires the two
# llama.cpp servers already up (bash harness/serve_local.sh: :8400 qwen3.5-9b-local,
# :8401 bge-large-en-v1.5). Chains the ASR venv and the RAG venv; exits nonzero on
# any failed stage. The `rag` step starts its own isolated stream service.
smoke-9b:
	CUDA_VISIBLE_DEVICES="" PYTHONPATH=. $(SMOKE_FW_PYBIN) harness/smoke_9b.py asr $(SMOKE_HANDOFF)
	PYTHONPATH=. $(SMOKE_RAG_PYBIN) harness/smoke_9b.py rag $(SMOKE_HANDOFF)
