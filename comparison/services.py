#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from comparison.identity import comparison_issues, service_role_issues

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "crag_eval"
DEFAULT_STATE_ROOT = ROOT / "comparison" / "benchmark" / "results" / "services"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Instance:
    name: str
    port: int
    state_dir: Path

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def qdrant_path(self) -> Path:
        return self.state_dir / "qdrant"

    @property
    def runtime_db(self) -> Path:
        return self.state_dir / "runtime.sqlite3"

    @property
    def metrics_log(self) -> Path:
        return self.state_dir / "requests.jsonl"

    @property
    def service_log(self) -> Path:
        return self.state_dir / "service.log"


@dataclass
class ServiceProcess:
    instance: Instance
    process: subprocess.Popen[bytes]
    log_handle: Any


def dataset_status(dataset_dir: Path) -> str:
    summary_path = dataset_dir / "dataset_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"dataset summary is missing: {summary_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"dataset summary is invalid JSON: {summary_path}") from exc
    return str(summary.get("selection", {}).get("approval_status", "unknown"))


def require_dataset(dataset_dir: Path) -> dict[str, Any]:
    # Each service re-verifies the full checksum manifest at load; this only
    # needs enough to launch and bind both services to the same corpus.
    missing = [
        name
        for name in ("checksums.sha256", "dataset_summary.json")
        if not (dataset_dir / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"dataset is missing files: {missing}")
    documents = next(
        (
            dataset_dir / name
            for name in ("documents.jsonl.bz2", "documents.jsonl")
            if (dataset_dir / name).is_file()
        ),
        None,
    )
    if documents is None:
        raise RuntimeError("dataset is missing a documents file")
    return {
        "approval_status": dataset_status(dataset_dir),
        "serving_dataset_checksum": sha256_file(dataset_dir / "checksums.sha256"),
        "documents_sha256": sha256_file(documents),
    }


def instances(args: argparse.Namespace) -> tuple[Instance, Instance]:
    state_root = args.state_root.resolve()
    naive = Instance("naive", args.naive_port, state_root / "naive")
    stream = Instance("stream", args.stream_port, state_root / "stream")
    if naive.port == stream.port:
        raise RuntimeError("Naive and Stream ports must differ")
    if naive.state_dir == stream.state_dir or naive.qdrant_path == stream.qdrant_path:
        raise RuntimeError("Naive and Stream state/Qdrant paths must differ")
    if state_root in {Path("/"), Path.home(), ROOT, args.dataset_dir.resolve()}:
        raise RuntimeError(f"unsafe benchmark state root: {state_root}")
    return naive, stream


def child_environment(dataset_dir: Path, instance: Instance) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            # The committed evaluation corpus is review-gated (candidate).
            "ALLOW_UNREVIEWED_DATASET": "1",
            "DATASET_DIR": str(dataset_dir),
            # Empty values override any .env managed-Qdrant settings and force separate
            # persisted local stores for the two benchmark processes.
            "QDRANT_URL": "",
            "QDRANT_API_KEY": "",
            "QDRANT_PATH": str(instance.qdrant_path),
            "RUNTIME_DB": str(instance.runtime_db),
            "METRICS_LOG": str(instance.metrics_log),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def start_service(dataset_dir: Path, instance: Instance) -> ServiceProcess:
    instance.state_dir.mkdir(parents=True, exist_ok=True)
    log_handle = instance.service_log.open("ab", buffering=0)
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        f"{instance.name}.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(instance.port),
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=child_environment(dataset_dir, instance),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        log_handle.close()
        raise
    return ServiceProcess(instance, process, log_handle)


def stop_service(service: ServiceProcess) -> None:
    process = service.process
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    service.log_handle.close()


def log_tail(path: Path, lines: int = 20) -> str:
    if not path.is_file():
        return "(no service log)"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def wait_for_status(service: ServiceProcess, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error = "service did not respond"
    while time.monotonic() < deadline:
        return_code = service.process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"{service.instance.name} service exited with {return_code}\n"
                f"{log_tail(service.instance.service_log)}"
            )
        try:
            response = httpx.get(f"{service.instance.base_url}/v1/data/status", timeout=2)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(0.25)
    raise RuntimeError(
        f"timed out waiting for {service.instance.name}: {last_error}\n"
        f"{log_tail(service.instance.service_log)}"
    )


def validate_status(
    status: dict[str, Any],
    name: str,
    *,
    require_index: bool,
    contract: dict[str, Any],
) -> None:
    if status.get("implementation") != name:
        raise RuntimeError(f"{name} API advertises {status.get('implementation')!r}")
    expected_snapshots = name == "stream"
    if status.get("supports_snapshots") is not expected_snapshots:
        raise RuntimeError(f"{name} supports_snapshots must be {expected_snapshots}")
    if status.get("serving_dataset_checksum") != contract["serving_dataset_checksum"]:
        raise RuntimeError(f"{name} serves a different dataset than requested")
    if require_index:
        issues = service_role_issues(status, name)
        if issues:
            raise RuntimeError(f"{name} service is not ready: {issues}")
        indexed = int(status.get("indexed_chunks") or 0)
        desired = int(status.get("indexed_desired_chunks") or -1)
        if indexed != desired:
            raise RuntimeError(f"{name} indexed chunk count does not match desired chunks")


def compare_statuses(statuses: dict[str, dict[str, Any]]) -> None:
    issues = comparison_issues(statuses["naive"], statuses["stream"])
    if issues:
        raise RuntimeError(f"services are not a fair, isolated pair: {issues}")


def public_configuration(
    dataset_dir: Path,
    state_root: Path,
    pair: tuple[Instance, Instance],
    contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_dir": str(dataset_dir),
        "approval_status": contract["approval_status"],
        "allow_unreviewed_dataset": True,
        "serving_dataset_checksum": contract["serving_dataset_checksum"],
        "state_root": str(state_root),
        "instances": {
            instance.name: {
                "base_url": instance.base_url,
                "qdrant_path": str(instance.qdrant_path),
                "runtime_db": str(instance.runtime_db),
                "metrics_log": str(instance.metrics_log),
                "service_log": str(instance.service_log),
            }
            for instance in pair
        },
    }


def preflight(
    args: argparse.Namespace,
) -> tuple[tuple[Instance, Instance], dict[str, Any]]:
    dataset_dir = args.dataset_dir.resolve()
    contract = require_dataset(dataset_dir)
    pair = instances(args)
    print(
        json.dumps(
            public_configuration(dataset_dir, args.state_root.resolve(), pair, contract),
            indent=2,
            sort_keys=True,
        )
    )
    return pair, contract


def sync_instance(
    dataset_dir: Path,
    contract: dict[str, Any],
    instance: Instance,
    timeout_s: float,
) -> dict[str, Any]:
    service = start_service(dataset_dir, instance)
    try:
        initial = wait_for_status(service, timeout_s)
        validate_status(initial, instance.name, require_index=False, contract=contract)
        response = httpx.post(
            f"{instance.base_url}/v1/data/sync",
            timeout=httpx.Timeout(30, read=None),
        )
        response.raise_for_status()
        try:
            sync_report = response.json()
        except ValueError as exc:
            raise RuntimeError(f"{instance.name} sync returned invalid JSON") from exc
        if not isinstance(sync_report, dict):
            raise RuntimeError(f"{instance.name} sync returned a non-object report")
        status_response = httpx.get(f"{instance.base_url}/v1/data/status", timeout=10)
        status_response.raise_for_status()
        status = status_response.json()
        validate_status(status, instance.name, require_index=True, contract=contract)
        status["sync_report"] = sync_report
        return status
    finally:
        stop_service(service)


def clone_quiescent_index(seed: Instance, target: Instance) -> None:
    if seed.state_dir == target.state_dir or seed.state_dir.parent != target.state_dir.parent:
        raise RuntimeError("seed and target must be distinct siblings under one state root")
    if not seed.qdrant_path.is_dir() or not seed.runtime_db.is_file():
        raise RuntimeError("quiescent seed is missing Qdrant or index metadata")
    if any(Path(f"{seed.runtime_db}{suffix}").exists() for suffix in ("-wal", "-shm")):
        raise RuntimeError("seed SQLite state is not quiescent")

    target.state_dir.mkdir(parents=True, exist_ok=True)
    if target.qdrant_path.exists():
        shutil.rmtree(target.qdrant_path)
    for path in (
        target.runtime_db,
        Path(f"{target.runtime_db}-wal"),
        Path(f"{target.runtime_db}-shm"),
        target.metrics_log,
        target.service_log,
    ):
        if path.exists():
            path.unlink()
    shutil.copytree(seed.qdrant_path, target.qdrant_path)
    shutil.copy2(seed.runtime_db, target.runtime_db)

    with sqlite3.connect(f"file:{target.runtime_db}?mode=ro", uri=True) as connection:
        result = connection.execute("PRAGMA integrity_check").fetchone()
    if result != ("ok",):
        raise RuntimeError(f"cloned SQLite index metadata failed integrity check: {result}")


def inspect_instance(
    dataset_dir: Path,
    contract: dict[str, Any],
    instance: Instance,
    timeout_s: float,
) -> dict[str, Any]:
    service = start_service(dataset_dir, instance)
    try:
        status = wait_for_status(service, timeout_s)
        validate_status(status, instance.name, require_index=True, contract=contract)
        return status
    finally:
        stop_service(service)


def command_sync(args: argparse.Namespace) -> None:
    pair, contract = preflight(args)
    dataset_dir = args.dataset_dir.resolve()
    state_root = args.state_root.resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".index-seed-", dir=state_root) as seed_root:
        seed = Instance("naive", args.naive_port, Path(seed_root))
        seed_status = sync_instance(dataset_dir, contract, seed, args.startup_timeout_s)
        for instance in pair:
            clone_quiescent_index(seed, instance)

    statuses = {
        instance.name: inspect_instance(dataset_dir, contract, instance, args.startup_timeout_s)
        for instance in pair
    }
    for status in statuses.values():
        status["sync_report"] = seed_status["sync_report"]
        status["index_provisioning"] = "one_quiescent_seed_cloned_to_isolated_stores"
    compare_statuses(statuses)
    output = state_root / "sync-status.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(statuses, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Built one index and cloned it into two isolated stores. Status: {output}")


def command_serve(args: argparse.Namespace) -> None:
    pair, contract = preflight(args)
    dataset_dir = args.dataset_dir.resolve()
    services: list[ServiceProcess] = []
    try:
        for instance in pair:
            services.append(start_service(dataset_dir, instance))
        statuses = {
            service.instance.name: wait_for_status(service, args.startup_timeout_s)
            for service in services
        }
        for name, status in statuses.items():
            validate_status(status, name, require_index=True, contract=contract)
        compare_statuses(statuses)
        print(
            "Two isolated services are ready. In another terminal run:\n"
            "  make benchmark\n"
            "Press Ctrl-C here after the run finishes."
        )
        while True:
            for service in services:
                return_code = service.process.poll()
                if return_code is not None:
                    raise RuntimeError(
                        f"{service.instance.name} service exited with {return_code}\n"
                        f"{log_tail(service.instance.service_log)}"
                    )
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for service in reversed(services):
            stop_service(service)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync and serve two isolated evaluation APIs over the committed dataset",
        epilog="The service launcher itself never executes benchmark questions.",
    )
    parser.add_argument(
        "command",
        choices=("check", "sync", "serve"),
        help="check configuration, sync both indexes, or serve both APIs",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--naive-port", type=int, default=8001)
    parser.add_argument("--stream-port", type=int, default=8002)
    parser.add_argument("--startup-timeout-s", type=float, default=60.0)
    return parser


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = build_parser()
    args = parser.parse_args()
    if not 1 <= args.naive_port <= 65535 or not 1 <= args.stream_port <= 65535:
        parser.error("ports must be between 1 and 65535")
    if args.startup_timeout_s <= 0:
        parser.error("--startup-timeout-s must be positive")
    try:
        if args.command == "check":
            preflight(args)
        elif args.command == "sync":
            command_sync(args)
        else:
            command_serve(args)
    except KeyboardInterrupt as exc:
        raise SystemExit(130) from exc
    except (RuntimeError, httpx.HTTPError, OSError) as exc:
        raise SystemExit(f"benchmark service error: {exc}") from exc


def raise_keyboard_interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    # Ensure Ctrl-C reaches this parent, which then terminates both children.
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    main()
