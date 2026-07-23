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

RUN mkdir -p /app/var
EXPOSE 8002
CMD ["uv", "run", "--no-sync", "uvicorn", "stream.api:app", "--host", "0.0.0.0", "--port", "8002"]
