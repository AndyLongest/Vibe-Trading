"""Quiet local report-server bootstrap for interactive backtest runs."""

from __future__ import annotations

import csv
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_REPORT_PORT = 8000
_SPAWNED_PROCESSES: list[subprocess.Popen[Any]] = []


def load_backtest_metrics(run_dir: str | Path | None) -> dict[str, float]:
    """Read the scalar metrics row that proves a turn produced a backtest."""
    if not run_dir:
        return {}
    path = Path(run_dir) / "artifacts" / "metrics.csv"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle), None) or {}
    except (OSError, csv.Error):
        return {}

    metrics: dict[str, float] = {}
    for key, value in row.items():
        try:
            metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def _candidate_ports() -> list[int]:
    raw = os.environ.get("VIBE_TRADING_REPORT_PORT", "").strip()
    try:
        preferred = int(raw) if raw else DEFAULT_REPORT_PORT
    except ValueError:
        preferred = DEFAULT_REPORT_PORT
    preferred = preferred if 1 <= preferred <= 65535 else DEFAULT_REPORT_PORT

    candidates = [preferred, DEFAULT_REPORT_PORT, 8899]
    candidates.extend(port for port in range(preferred + 1, min(preferred + 7, 65536)))
    return list(dict.fromkeys(candidates))


def _port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def _run_url(port: int, run_id: str) -> str:
    return f"http://127.0.0.1:{port}/runs/{quote(run_id, safe='')}?view=dashboard"


def _vibe_report_ready(port: int, run_id: str) -> bool:
    headers: dict[str, str] = {"Accept": "application/json"}
    api_key = os.environ.get("API_AUTH_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(_run_url(port, run_id), headers=headers)
    try:
        with urlopen(request, timeout=0.6) as response:  # noqa: S310 - fixed loopback URL
            return response.status == 200
    except (HTTPError, URLError, OSError, TimeoutError):
        return False


def _spawn_report_server(port: int) -> subprocess.Popen[Any] | None:
    agent_dir = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "cli._legacy",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    kwargs: dict[str, Any] = {
        "cwd": str(agent_dir),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        )
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(command, **kwargs)
    except OSError:
        return None
    _SPAWNED_PROCESSES.append(process)
    return process


def ensure_report_server(run_id: str, *, timeout_seconds: float = 30.0) -> str | None:
    """Return a working report URL, quietly starting a loopback server if needed."""
    if not run_id:
        return None

    candidates = _candidate_ports()
    for port in candidates:
        if _vibe_report_ready(port, run_id):
            return _run_url(port, run_id)

    port = next((candidate for candidate in candidates if not _port_is_open(candidate)), None)
    if port is None:
        return None
    process = _spawn_report_server(port)
    if process is None:
        return None

    deadline = time.monotonic() + max(0.1, timeout_seconds)
    while time.monotonic() < deadline:
        if _vibe_report_ready(port, run_id):
            return _run_url(port, run_id)
        if process.poll() is not None:
            return None
        time.sleep(0.15)
    return None
