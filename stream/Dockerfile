# --- Native stage: compile the Rust snapshot-delta wheel ---
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS native
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /build
COPY native/snapshot_delta ./native/snapshot_delta
RUN cd native/snapshot_delta && uvx maturin build --release --out /wheels

# --- Runtime stage ---
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    QDRANT_PATH=/app/var/qdrant \
    RUNTIME_DB=/app/var/runtime.sqlite3 \
    METRICS_LOG=/app/var/requests.jsonl

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY shared ./shared
COPY stream ./stream
COPY data/crag_eval ./data/crag_eval
RUN uv sync --frozen --no-dev

# Install the native snapshot backend into the app venv. If the wheel is missing
# or ABI-incompatible the stream service still runs on the pure-Python fallback.
COPY --from=native /wheels /wheels
RUN uv pip install --python /app/.venv/bin/python /wheels/*.whl \
    || echo "native snapshot wheel not installed; using Python fallback"

RUN mkdir -p /app/var
EXPOSE 8002
CMD ["uv", "run", "--no-sync", "uvicorn", "stream.api:app", "--host", "0.0.0.0", "--port", "8002"]
