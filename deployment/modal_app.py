"""
Modal deployment for the StreamRAG assessment stack — one container, one URL.

The local project is a five-container docker-compose stack (frontend + two
FastAPI paths + two Qdrant servers). Modal does not run docker-compose, so this
file collapses that topology into ONE container / ONE public HTTPS URL while
keeping the two paths' state isolated exactly as before:

  browser ──► gateway (FastAPI, defined here)
                 ├─ mount /api/naive  ──► naive.api:app   ──► embedded Qdrant @ /app/var/naive
                 ├─ mount /api/stream ──► stream.api:app  ──► embedded Qdrant @ /app/var/stream
                 └─ static: the built React SPA (frontend/dist), same-origin, no CORS
             naive & stream ──► OpenAI API (embeddings + answer model)

Why isolation still holds in one process: naive.config and stream.config build
SEPARATE Settings objects whose DEFAULT paths differ (var/naive/... vs
var/stream/...), and shared.api.factory.create_app is fully parameterized — so
two app objects in one interpreter never share an index, a SQLite runtime DB, or
a cache. We deliberately do NOT set QDRANT_PATH/RUNTIME_DB env vars (both configs
read the same var name, which would collide); the per-path defaults do the split.

Indexing: embedded Qdrant (QDRANT_URL unset) writes to a Modal Volume mounted at
/app/var. On startup the gateway runs each path's /v1/data/sync — idempotent, so
the first cold start embeds ~1000 chunks/path (~80s, a few cents) and every later
cold start finds the vectors already present and re-embeds nothing.

Commands (run from the project root):
  modal run    deployment/modal_app.py            # import/health probe
  modal deploy deployment/modal_app.py            # go live
  modal app stop streamrag                        # take it down (keeps the Volume)

Requires a Modal Secret named 'streamrag-openai' holding OPENAI_API_KEY.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "streamrag"
SECRET_NAME = "streamrag-openai"
VOLUME_NAME = "streamrag-state"
VAR_DIR = "/app/var"

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"

# Runtime dependencies, verbatim from the project's pyproject.toml. The native
# Rust snapshot module is intentionally omitted: stream/snapshot.py falls back to
# an identical pure-Python implementation when the wheel is absent.
PY_DEPS = [
    "fastapi>=0.115,<1",
    "uvicorn[standard]>=0.34,<1",
    "openai>=2.20,<3",
    "pydantic-ai-slim[openai]>=1.0,<2",
    "qdrant-client>=1.15,<2",
    "numpy>=2.0,<3",
    "pydantic>=2.10,<3",
    "python-dotenv>=1.0,<2",
    "tiktoken>=0.9,<1",
    "aiosqlite>=0.20,<1",
    "sse-starlette>=3.0,<4",
    "httpx>=0.28,<1",
]

# The code uses Python 3.14-only grammar (bare multi-exception `except A, B:`),
# so the image must ship 3.14. Mirror the project's own Dockerfile base.
image = (
    modal.Image.from_registry(
        "ghcr.io/astral-sh/uv:python3.14-bookworm-slim",
        add_python=None,
    )
    .env(
        {
            "UV_SYSTEM_PYTHON": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/app",
            # The committed corpus is a review-gated candidate; this is the
            # project's normal "load it anyway" flag (see docs/RUN.md).
            "ALLOW_UNREVIEWED_DATASET": "1",
        }
    )
    .run_commands("uv pip install --system " + " ".join(f'"{d}"' for d in PY_DEPS))
    .add_local_dir(str(REPO_ROOT / "shared"), "/app/shared", copy=True)
    .add_local_dir(str(REPO_ROOT / "naive"), "/app/naive", copy=True)
    .add_local_dir(str(REPO_ROOT / "stream"), "/app/stream", copy=True)
    .add_local_dir(str(REPO_ROOT / "data" / "crag_eval"), "/app/data/crag_eval", copy=True)
    .add_local_dir(str(FRONTEND_DIST), "/app/frontend/dist", copy=True)
    .workdir("/app")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
openai_secret = modal.Secret.from_name(SECRET_NAME)


def _build_gateway():
    """Construct the same-origin gateway that fronts both RAG paths + the SPA.

    Runs inside the container. Imports the child apps (their config reads env at
    import time, which Modal has already populated), mounts them under the paths
    the frontend calls, drives their lifespans, indexes once, and serves the SPA.
    """
    from contextlib import AsyncExitStack, asynccontextmanager

    import httpx
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    # Imported here (not at module top) so this only happens in the container,
    # where the packages and PYTHONPATH=/app exist.
    from naive.api import app as naive_app
    from stream.api import app as stream_app

    dist = "/app/frontend/dist"

    @asynccontextmanager
    async def lifespan(_gateway: FastAPI):
        async with AsyncExitStack() as stack:
            # Mounted sub-apps do NOT get their lifespan run by Starlette, so we
            # drive each child's lifespan explicitly (index setup, DB, caches).
            await stack.enter_async_context(naive_app.router.lifespan_context(naive_app))
            await stack.enter_async_context(stream_app.router.lifespan_context(stream_app))
            # Idempotent index build straight into the mounted Volume. First cold
            # start embeds the corpus (~80s, a few cents); later starts re-embed
            # nothing because the vectors already sit on the Volume.
            for name, child in (("naive", naive_app), ("stream", stream_app)):
                transport = httpx.ASGITransport(app=child)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://svc", timeout=600
                ) as client:
                    resp = await client.post("/v1/data/sync")
                    resp.raise_for_status()
                    report = resp.json()
                    print(
                        f"[sync:{name}] indexed={report.get('desired_chunks')} "
                        f"embedded={report.get('embedded_chunks')} "
                        f"unchanged={report.get('unchanged_chunks')}"
                    )
            volume.commit()
            yield

    gateway = FastAPI(title="StreamRAG (Naive vs StreamRAG)", lifespan=lifespan)

    # Same-origin API surface the built frontend expects (frontend/src/api.ts).
    gateway.mount("/api/naive", naive_app)
    gateway.mount("/api/stream", stream_app)
    # Hashed JS/CSS bundles.
    gateway.mount("/assets", StaticFiles(directory=f"{dist}/assets"), name="assets")

    @gateway.get("/favicon.svg", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(f"{dist}/favicon.svg")

    # SPA fallback: everything else (/, /naive, /stream, /compare) serves the
    # app shell so client-side routing can take over. Added last, so the mounts
    # above win for their prefixes.
    @gateway.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str) -> FileResponse:
        return FileResponse(f"{dist}/index.html")

    return gateway


@app.function(
    image=image,
    volumes={VAR_DIR: volume},
    secrets=[openai_secret],
    cpu=2,
    memory=4096,
    scaledown_window=300,
    timeout=900,
    max_containers=1,  # single writer for the embedded Qdrant/SQLite on the Volume
)
@modal.concurrent(max_inputs=20)  # work is OpenAI-I/O-bound; one container fans out
@modal.asgi_app()
def web():
    return _build_gateway()


@app.function(image=image)
def probe() -> str:
    """`modal run` sanity check: image runs under 3.14 and the apps import."""
    import sys

    import naive.api  # noqa: F401
    import stream.api  # noqa: F401
    from stream.snapshot import SnapshotAnalyzer

    return (
        f"python={sys.version.split()[0]} snapshot_backend={SnapshotAnalyzer.backend} "
        f"naive={naive.api.app.title!r} stream={stream.api.app.title!r}"
    )


@app.local_entrypoint()
def main() -> None:
    print(probe.remote())
