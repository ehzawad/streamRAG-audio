# Frontend

`frontend/` is the React/Vite GUI. It owns browser presentation and request
lifecycle only; it does not run benchmarks, read gold answers, or implement a
RAG path.

## Run locally

Start the API or APIs you want to use, then:

```bash
make setup-frontend
make dev-frontend
```

Open <http://127.0.0.1:5173/> and choose:

- `/naive` for the baseline;
- `/stream` for StreamRAG;
- `/compare` for the same commit sent to both services.

Vite proxies `/api/naive/*` and `/api/stream/*` to ports 8001 and 8002. The
Docker image exposes the same routes through nginx, so users need one origin and
do not need to know backend ports.

Each route preserves chat context across follow-ups. **New chat** starts a new
session. Compare keeps one independent conversation per backend.

```bash
make check-frontend
docker build --tag streamrag-frontend frontend
```

The image supports deep links and unbuffered SSE. Public deployment still needs
TLS, authentication, spend controls, and persistent service volumes.
