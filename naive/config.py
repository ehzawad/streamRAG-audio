from __future__ import annotations

import os
from pathlib import Path

from shared.config import ROOT, Settings

settings = Settings(
    qdrant_path=Path(os.getenv("QDRANT_PATH", ROOT / "var" / "naive" / "qdrant")),
    runtime_db=Path(os.getenv("RUNTIME_DB", ROOT / "var" / "naive" / "runtime.sqlite3")),
    metrics_log=Path(os.getenv("METRICS_LOG", ROOT / "var" / "naive" / "requests.jsonl")),
)
settings.validate()
