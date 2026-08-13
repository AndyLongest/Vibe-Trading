"""Capital integrity regressions for the synthetic options engine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtest.engines.options_portfolio import historical_volatility, run_options_backtest


_DATES = pd.bdate_range("2025-01-01", periods=4)
_BARS = pd.DataFrame(
    {
        "open": [100.0, 101.0, 102.0, 103.0],
        "high": [101.0, 102.0, 103.0, 104.0],
        "low": [99.0, 100.0, 101.0, 102.0],
        "close": [100.0, 101.0, 102.0, 103.0],
        "volume": [1000, 1000, 1000, 1000],
    },
    index=_DATES,
)


class _Loader:
    name = "synthetic"

    def fetch(self, codes, start_date, end_date):  # noqa: ANN001
        return {"SPY": _BARS.copy()}


class _Engine:
    def __init__(self, legs):  # noqa: ANN001
        self.legs = legs

    def generate(self, data_map):  # noqa: ANN001
        return [{
            "date": "2025-01-01",
            "action": "open",
            "underlying": "SPY",
            "legs": self.legs,
        }]


def _run(tmp_path: Path, legs, *, initial_cash: float = 1_000.0, options_config=None):  # noqa: ANN001
    return run_options_backtest(
        {
            "codes": ["SPY"],
            "start_date": "2025-01-01",
            "end_date": "2025-01-06",
            "engine": "options",
            "initial_cash": initial_cash,
            "commission": 0.0,
            "options_config": {"contract_multiplier": 100, **(options_config or {})},
        },
        _Loader(),
        _Engine(legs),
        tmp_path,
    )


def test_historical_volatility_warmup_does_not_backfill_from_future() -> None:
    close = pd.Series([100.0, 101.0, 99.0, 102.0], index=_DATES)
    original = historical_volatility(close, window=3)
    changed_future = historical_volatility(
        pd.Series([100.0, 101.0, 250.0, 400.0], index=_DATES), window=3
    )

    assert original.iloc[0] == 0.3
    assert original.iloc[1] == 0.3
    assert changed_future.iloc[:2].equals(original.iloc[:2])


def test_unaffordable_long_option_is_rejected(tmp_path: Path) -> None:
    _run(tmp_path, [{"type": "call", "strike": 100, "expiry": "2025-12-31", "qty": 1_000_000}])

    trades = pd.read_csv(tmp_path / "artifacts" / "trades.csv")
    equity = pd.read_csv(tmp_path / "artifacts" / "equity.csv")
    rejected = pd.read_csv(tmp_path / "artifacts" / "rejections.csv")
    assert trades.empty
    assert equity["cash"].min() == 1_000.0
    assert rejected.iloc[0]["reason"] == "insufficient_buying_power"


def test_naked_short_requires_margin(tmp_path: Path) -> None:
    _run(tmp_path, [{"type": "put", "strike": 100, "expiry": "2025-12-31", "qty": -10}])

    assert pd.read_csv(tmp_path / "artifacts" / "trades.csv").empty
    assert len(pd.read_csv(tmp_path / "artifacts" / "rejections.csv")) == 1


def test_multileg_open_is_atomic_when_buying_power_is_insufficient(tmp_path: Path) -> None:
    _run(tmp_path, [
        {"type": "call", "strike": 100, "expiry": "2025-12-31", "qty": -1},
        {"type": "call", "strike": 200, "expiry": "2025-12-31", "qty": 1_000_000},
    ])

    assert pd.read_csv(tmp_path / "artifacts" / "trades.csv").empty
    assert len(pd.read_csv(tmp_path / "artifacts" / "rejections.csv")) == 1


def test_unknown_iv_source_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="iv_source"):
        _run(
            tmp_path,
            [{"type": "call", "strike": 100, "expiry": "2025-12-31", "qty": 1}],
            options_config={"iv_source": "market"},
        )


def test_weekend_expiry_uses_last_pre_expiry_close(tmp_path: Path) -> None:
    bars = pd.DataFrame(
        {
            "open": [90.0, 110.0], "high": [91.0, 111.0],
            "low": [89.0, 109.0], "close": [90.0, 110.0],
            "volume": [1000, 1000],
        },
        index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
    )

    class WeekendLoader:
        name = "synthetic"
        def fetch(self, codes, start_date, end_date):  # noqa: ANN001
            return {"SPY": bars}

    class WeekendEngine:
        def generate(self, data_map):  # noqa: ANN001
            return [{
                "date": "2025-01-03", "action": "open", "underlying": "SPY",
                "legs": [{"type": "call", "strike": 100, "expiry": "2025-01-04", "qty": 1}],
            }]

    run_options_backtest(
        {
            "codes": ["SPY"], "start_date": "2025-01-03", "end_date": "2025-01-06",
            "engine": "options", "initial_cash": 10_000, "commission": 0,
            "options_config": {"contract_multiplier": 1},
        }, WeekendLoader(), WeekendEngine(), tmp_path,
    )
    trades = pd.read_csv(tmp_path / "artifacts" / "trades.csv")
    settlement = trades[trades["side"].isin(["exercise", "expire"])].iloc[0]
    assert settlement["side"] == "expire"
    assert settlement["price"] == 0.0
