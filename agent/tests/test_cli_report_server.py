from __future__ import annotations

from pathlib import Path

from rich.console import Console
from unittest.mock import Mock

from cli import report_server


def test_load_backtest_metrics_reads_scalar_row(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "metrics.csv").write_text(
        "total_return,annual_return,sharpe,max_drawdown,win_rate,trade_count\n"
        "0.824,0.131,1.08,-0.187,0.564,42\n",
        encoding="utf-8",
    )

    metrics = report_server.load_backtest_metrics(tmp_path)

    assert metrics["total_return"] == 0.824
    assert metrics["trade_count"] == 42


def test_ensure_report_server_reuses_running_local_server(monkeypatch) -> None:
    monkeypatch.setattr(report_server, "_candidate_ports", lambda: [8123])
    monkeypatch.setattr(report_server, "_vibe_report_ready", lambda port, run_id: True)

    url = report_server.ensure_report_server("run 1")

    assert url == "http://127.0.0.1:8123/runs/run%201?view=dashboard"


def test_interactive_backtest_result_prints_metrics_and_dashboard_link(tmp_path: Path, monkeypatch) -> None:
    from cli.main import _print_interactive_result

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "metrics.csv").write_text(
        "total_return,annual_return,sharpe,max_drawdown,win_rate,trade_count\n"
        "0.824,0.131,1.08,-0.187,0.564,42\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        report_server,
        "ensure_report_server",
        lambda run_id: f"http://127.0.0.1:8000/runs/{run_id}?view=dashboard",
    )
    console = Console(record=True, force_terminal=False, color_system=None, width=100)

    _print_interactive_result(
        console,
        {"status": "success", "run_id": "run_123", "run_dir": str(tmp_path), "content": "Done."},
        3.5,
    )
    output = console.export_text()

    assert "82.4%" in output
    assert "Sharpe" in output and "1.08" in output
    assert "run_123" in output
    assert "http://127.0.0.1:8000/runs/run_123?view=dashboard" in output


def test_single_prompt_run_prints_dashboard_link(tmp_path: Path, monkeypatch) -> None:
    from cli import _legacy
    from src import preflight

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "metrics.csv").write_text("total_return\n0.1\n", encoding="utf-8")
    result = {
        "status": "success",
        "run_id": "single-run",
        "run_dir": str(tmp_path),
        "content": "Done.",
    }
    monkeypatch.setattr(preflight, "run_preflight", lambda console: [])
    monkeypatch.setattr(_legacy, "_ensure_session_id", lambda prompt: "session")
    monkeypatch.setattr(_legacy, "_run_agent", lambda *args, **kwargs: result)
    monkeypatch.setattr(_legacy, "_print_result", Mock())
    monkeypatch.setattr(
        report_server,
        "ensure_report_server",
        lambda run_id: f"http://127.0.0.1:8000/runs/{run_id}?view=dashboard",
    )
    test_console = Console(record=True, force_terminal=False, color_system=None, width=100)
    monkeypatch.setattr(_legacy, "console", test_console)

    exit_code = _legacy.cmd_run("Backtest AAPL", 10)

    assert exit_code == 0
    assert "http://127.0.0.1:8000/runs/single-run?view=dashboard" in test_console.export_text()
