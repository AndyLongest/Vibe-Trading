"""Options portfolio backtest engine (v2).

Supports European and American options via Black-Scholes model with
IV smile approximation.  Synthesises theoretical option prices from
underlying prices; supports multi-leg strategies.

v2 enhancements over v1:
  - American option support (early exercise heuristic for calls on dividends,
    always-exercise check for deep ITM puts)
  - IV smile model: skew adjustment based on moneyness (log(K/S))
  - Portfolio-level Greeks aggregation

Signal interface: OptionsSignalEngine.generate(data_map) returns a list of trade instructions.
Artifacts: equity.csv, metrics.csv, trades.csv, greeks.csv.

Black-Scholes price and Greeks come from ``src.quantlib.options``. What stays
here is the engine's own volatility surface -- historical vol, the smile, and
the per-leg vol every pricing site must agree on.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.quantlib.options import bs_greeks, bs_price, normalise_option_type


# --- Historical volatility ---


def historical_volatility(close: pd.Series, window: int = 30) -> pd.Series:
    """Calculate annualised historical volatility from a close price series.

    Args:
        close: Close price Series.
        window: Rolling window in days.

    Returns:
        Annualised historical volatility Series.
    """
    log_ret = np.log(close / close.shift(1))
    # ``min_periods=2`` uses only observations available as of each bar.  Do
    # not backfill the warm-up window from the first future 30-day estimate:
    # that leaks future volatility into the beginning of the backtest.
    hv = log_ret.rolling(window=window, min_periods=2).std() * np.sqrt(252)
    return hv.fillna(0.3)


def _validate_options_config(options_cfg: Dict[str, Any]) -> None:
    """Reject unsupported or economically invalid option-engine settings."""
    iv_source = str(options_cfg.get("iv_source", "historical")).lower()
    if iv_source != "historical":
        raise ValueError(
            f"Unsupported options_config.iv_source={iv_source!r}; "
            "only 'historical' is currently implemented"
        )
    exercise_style = str(options_cfg.get("exercise_style", "european")).lower()
    if exercise_style not in {"european", "american"}:
        raise ValueError("options_config.exercise_style must be 'european' or 'american'")
    for key in ("contract_multiplier", "short_margin_rate", "short_margin_floor"):
        value = float(options_cfg.get(key, 1.0 if key == "contract_multiplier" else 0.2 if key == "short_margin_rate" else 0.1))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"options_config.{key} must be positive and finite")


def _short_margin_requirement(
    pos: "OptionPosition",
    spot: float,
    option_price: float,
    multiplier: float,
    margin_rate: float,
    margin_floor: float,
) -> float:
    """Return a conservative naked-short initial margin requirement.

    This is an OCC-style approximation, not a broker-specific schedule.  It
    deliberately applies no spread offsets, so bounded multi-leg structures
    remain conservative until an exchange-aware margin model is supplied.
    """
    if pos.qty >= 0:
        return 0.0
    otm = max(pos.strike - spot, 0.0) if pos.option_type == "call" else max(spot - pos.strike, 0.0)
    per_unit = option_price + max(margin_rate * spot - otm, margin_floor * spot)
    return per_unit * abs(pos.qty) * multiplier


def _portfolio_short_margin(
    positions: List["OptionPosition"],
    spot_prices: Dict[str, float],
    ivs: Dict[str, float],
    ts: pd.Timestamp,
    *,
    risk_free_rate: float,
    multiplier: float,
    iv_skew: float,
    iv_curvature: float,
    margin_rate: float,
    margin_floor: float,
) -> float:
    """Aggregate conservative margin for all open short option legs."""
    total = 0.0
    for pos in positions:
        if pos.qty >= 0:
            continue
        spot = spot_prices.get(pos.underlying_code, 0.0)
        base_iv = ivs.get(pos.underlying_code, 0.3)
        iv = leg_iv(spot, pos.strike, base_iv, iv_skew, iv_curvature)
        price = bs_price(
            spot, pos.strike, pos.time_to_expiry(ts), risk_free_rate, iv, pos.option_type
        )
        total += _short_margin_requirement(
            pos, spot, price, multiplier, margin_rate, margin_floor
        )
    return total


# --- IV Smile model (v2) ---


def iv_smile_adjustment(S: float, K: float, base_iv: float,
                        skew: float = -0.15, curvature: float = 0.05) -> float:
    """Adjust IV for moneyness using a quadratic smile model.

    IV(K) = base_iv + skew * log(K/S) + curvature * log(K/S)^2

    Args:
        S: Spot price.
        K: Strike price.
        base_iv: At-the-money implied volatility.
        skew: Slope of the smile (negative = put skew). Default -0.15.
        curvature: Curvature of the smile (always positive). Default 0.05.

    Returns:
        Adjusted implied volatility, floored at 0.01.
    """
    if S <= 0 or K <= 0:
        return max(base_iv, 0.01)
    log_moneyness = np.log(K / S)
    adj = base_iv + skew * log_moneyness + curvature * log_moneyness ** 2
    return max(adj, 0.01)


def leg_iv(S: float, K: float, base_iv: float, skew: float, curvature: float) -> float:
    """Return the implied vol a single leg is priced at.

    Every site that prices a leg must go through this — opening, marking to
    market, Greeks, and the American continuation value. Opening a leg on the
    smile and marking it at flat at-the-money vol books a fictitious profit the
    instant the position exists: on a 30-day 10%-OTM call at ``skew=-0.15`` the
    gap is +16.7% of premium, and +93.0% at 20% OTM, which then contaminates
    Sharpe, Calmar and drawdown.

    Args:
        S: Spot price.
        K: Strike price.
        base_iv: At-the-money implied volatility.
        skew: Slope of the smile; ``0`` with ``curvature`` disables the smile.
        curvature: Curvature of the smile.

    Returns:
        The leg's implied volatility.
    """
    if skew == 0 and curvature == 0:
        return base_iv
    return iv_smile_adjustment(S, K, base_iv, skew, curvature)


# --- Option positions ---


class OptionPosition:
    """A single option leg position.

    Attributes:
        option_type: "call" or "put", folded to lower case on construction so
            that settlement here and pricing in ``src.quantlib.options`` cannot
            disagree about a leg typed ``"Call"``.
        strike: Strike price.
        expiry: Expiry date.
        qty: Quantity (positive = long, negative = short).
        entry_price: Theoretical option price at entry.
        entry_date: Entry date string.
        underlying_code: Underlying instrument code.

    Raises:
        ValueError: If ``option_type`` is neither call nor put.
    """

    def __init__(self, option_type: str, strike: float, expiry: str,
                 qty: int, entry_price: float, entry_date: str,
                 underlying_code: str):
        self.option_type = normalise_option_type(option_type)
        self.strike = strike
        self.expiry = pd.Timestamp(expiry)
        self.qty = qty
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.underlying_code = underlying_code

    def time_to_expiry(self, current_date: pd.Timestamp) -> float:
        """Calculate time remaining to expiry in years.

        Args:
            current_date: Current date.

        Returns:
            Time to expiry in years.
        """
        days = (self.expiry - current_date).days
        return max(days / 365.0, 0.0)

    def is_expired(self, current_date: pd.Timestamp) -> bool:
        """Check whether the option has expired.

        Args:
            current_date: Current date.

        Returns:
            True if expired.
        """
        return current_date >= self.expiry

    def intrinsic_value(self, spot: float) -> float:
        """Calculate intrinsic value.

        Args:
            spot: Underlying spot price.

        Returns:
            Intrinsic value.
        """
        if self.option_type == "call":
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)


# --- Backtest driver ---


def run_options_backtest(
    config: Dict[str, Any],
    loader: Any,
    engine: Any,
    run_dir: Path,
    bars_per_year: int = 252,
) -> Dict[str, Any]:
    """Options backtest entry point.

    Day-by-day simulation:
    1. Read underlying price for the current day
    2. Mark all open option positions to market (BS)
    3. Execute trade instructions from the signal (open/close)
    4. Automatically exercise ITM options or expire OTM options at maturity
    5. Record P&L and Greeks

    Args:
        config: Backtest config; must include codes, start_date, end_date, initial_cash,
                and options_config (risk_free_rate, iv_source).
        loader: DataLoader instance (must have a fetch method).
        engine: OptionsSignalEngine instance (generate method returns a list of trade instructions).
        run_dir: Run directory path.
        bars_per_year: Bars per year.

    Returns:
        Metrics dictionary.

    Raises:
        SystemExit: When no data is fetched.
    """
    codes = config.get("codes", [])
    start_date = config.get("start_date", "")
    end_date = config.get("end_date", "")
    initial_cash = config.get("initial_cash", 1_000_000)
    commission = config.get("commission", 0.001)
    options_cfg = config.get("options_config", {})
    _validate_options_config(options_cfg)
    risk_free_rate = options_cfg.get("risk_free_rate", 0.05)
    contract_multiplier = options_cfg.get("contract_multiplier", 1.0)
    exercise_style = options_cfg.get("exercise_style", "european")  # v2: "european" or "american"
    iv_skew = options_cfg.get("iv_skew", 0.0)         # v2: smile skew param (0 = flat)
    iv_curvature = options_cfg.get("iv_curvature", 0.0)  # v2: smile curvature
    enforce_capital = options_cfg.get("enforce_capital_constraints", True) is not False
    short_margin_rate = float(options_cfg.get("short_margin_rate", 0.20))
    short_margin_floor = float(options_cfg.get("short_margin_floor", 0.10))

    # Load underlying data
    data_map = loader.fetch(codes, start_date, end_date)
    if not data_map:
        print(json.dumps({"error": "No data fetched"}))
        sys.exit(1)

    # Compute implied volatility (approximated by historical volatility)
    iv_map: Dict[str, pd.Series] = {}
    for code, df in data_map.items():
        iv_map[code] = historical_volatility(df["close"])

    # Generate trade signals
    signals = engine.generate(data_map)

    # Build trading date sequence
    all_dates = set()
    for df in data_map.values():
        all_dates.update(df.index)
    dates = sorted(all_dates)

    # Index signals by date
    signal_by_date: Dict[str, List[Dict[str, Any]]] = {}
    for sig in signals:
        d = sig.get("date", "")
        signal_by_date.setdefault(d, []).append(sig)

    # Day-by-day simulation
    cash = float(initial_cash)
    positions: List[OptionPosition] = []
    trade_records: List[Dict[str, Any]] = []
    greeks_records: List[Dict[str, Any]] = []
    equity_records: List[Dict[str, Any]] = []
    rejection_records: List[Dict[str, Any]] = []

    for current_date in dates:
        ts = pd.Timestamp(current_date)
        date_str = str(ts.date()) if hasattr(ts, "date") else str(ts)

        # 1. Get underlying price and IV for the current day
        spot_prices: Dict[str, float] = {}
        ivs: Dict[str, float] = {}
        for code, df in data_map.items():
            if ts in df.index:
                spot_prices[code] = float(df.at[ts, "close"])
                ivs[code] = float(iv_map[code].at[ts]) if ts in iv_map[code].index else 0.3
            else:
                # Use the last available price
                before = df.index[df.index <= ts]
                if len(before) > 0:
                    last = before[-1]
                    spot_prices[code] = float(df.at[last, "close"])
                    ivs[code] = float(iv_map[code].at[last]) if last in iv_map[code].index else 0.3

        # 2a. American early exercise (v2): exercise if intrinsic > continuation
        if exercise_style == "american":
            for pos in list(positions):
                if pos.is_expired(ts):
                    continue  # handled below
                spot = spot_prices.get(pos.underlying_code, 0.0)
                iv_val_ex = ivs.get(pos.underlying_code, 0.3)
                T_ex = pos.time_to_expiry(ts)
                if T_ex <= 0:
                    continue
                intrinsic = pos.intrinsic_value(spot)
                # The continuation value must use the same vol the leg is
                # marked at, or early exercise triggers off a mispriced hold.
                iv_ex = leg_iv(spot, pos.strike, iv_val_ex, iv_skew, iv_curvature)
                continuation = bs_price(spot, pos.strike, T_ex, risk_free_rate, iv_ex, pos.option_type)
                if intrinsic > 0 and intrinsic > continuation * 1.02:
                    # Early exercise is optimal
                    settlement = intrinsic * pos.qty * contract_multiplier
                    cash += settlement
                    pnl = (intrinsic - pos.entry_price) * pos.qty * contract_multiplier
                    trade_records.append({
                        "timestamp": date_str,
                        "code": pos.underlying_code,
                        "option_type": pos.option_type,
                        "strike": pos.strike,
                        "expiry": str(pos.expiry.date()),
                        "side": "early_exercise",
                        "price": round(intrinsic, 4),
                        "qty": pos.qty,
                        "pnl": round(pnl, 4),
                        "entry_date": pos.entry_date,
                    })
                    positions.remove(pos)

        # 2b. Handle expiry
        expired = [p for p in positions if p.is_expired(ts)]
        for pos in expired:
            # When expiry falls on a non-trading day, settle from the last
            # underlying bar on or before expiry.  Using the next Monday's
            # close would introduce post-expiry information into settlement.
            underlying_df = data_map.get(pos.underlying_code)
            spot = spot_prices.get(pos.underlying_code, 0.0)
            if underlying_df is not None:
                settlement_dates = underlying_df.index[underlying_df.index <= pos.expiry]
                if len(settlement_dates) > 0:
                    spot = float(underlying_df.at[settlement_dates[-1], "close"])
            intrinsic = pos.intrinsic_value(spot)

            # Expiry: recover intrinsic value (entry_price already deducted at open)
            settlement = intrinsic * pos.qty * contract_multiplier
            cash += settlement
            pnl = (intrinsic - pos.entry_price) * pos.qty * contract_multiplier

            side = "exercise" if intrinsic > 0 else "expire"
            trade_records.append({
                "timestamp": date_str,
                "code": pos.underlying_code,
                "option_type": pos.option_type,
                "strike": pos.strike,
                "expiry": str(pos.expiry.date()),
                "side": side,
                "price": round(intrinsic, 4),
                "qty": pos.qty,
                "pnl": round(pnl, 4),
                "entry_date": pos.entry_date,
            })
            positions.remove(pos)

        # 3. Execute today's signals
        day_signals = signal_by_date.get(date_str, [])
        for sig in day_signals:
            action = sig.get("action", "")
            legs = sig.get("legs", [])
            underlying = sig.get("underlying", codes[0] if codes else "")

            spot = spot_prices.get(underlying, 0.0)
            iv_val = ivs.get(underlying, 0.3)

            prepared_legs: List[Dict[str, Any]] = []
            for leg in legs:
                # Fold before it is priced, matched and recorded: config comes
                # from the user, and a raw "Call" would price as a call and
                # settle as a put.
                leg_type = normalise_option_type(leg.get("type", "call"))
                strike = leg.get("strike", spot)
                expiry = leg.get("expiry", "")
                qty = leg.get("qty", 1)

                expiry_ts = pd.Timestamp(expiry)
                T = max((expiry_ts - ts).days / 365.0, 0.001)

                adj_iv = leg_iv(spot, strike, iv_val, iv_skew, iv_curvature)
                opt_price = bs_price(spot, strike, T, risk_free_rate, adj_iv, leg_type)

                prepared_legs.append({
                    "raw": leg,
                    "type": leg_type,
                    "strike": strike,
                    "expiry": expiry,
                    "qty": qty,
                    "price": opt_price,
                })

            # Opening a structure is atomic.  Price every leg first, then test
            # the resulting cash and short-margin requirement before recording
            # any fill.  This prevents half-built spreads and unlimited credit.
            if action == "open" and enforce_capital:
                prospective_cash = cash
                prospective_positions = list(positions)
                for prepared in prepared_legs:
                    qty = prepared["qty"]
                    premium = prepared["price"] * abs(qty) * contract_multiplier
                    prospective_cash += premium * (1 - commission) if qty < 0 else -premium * (1 + commission)
                    prospective_positions.append(OptionPosition(
                        option_type=prepared["type"],
                        strike=prepared["strike"],
                        expiry=prepared["expiry"],
                        qty=qty,
                        entry_price=prepared["price"],
                        entry_date=date_str,
                        underlying_code=underlying,
                    ))
                required_margin = _portfolio_short_margin(
                    prospective_positions,
                    spot_prices,
                    ivs,
                    ts,
                    risk_free_rate=risk_free_rate,
                    multiplier=contract_multiplier,
                    iv_skew=iv_skew,
                    iv_curvature=iv_curvature,
                    margin_rate=short_margin_rate,
                    margin_floor=short_margin_floor,
                )
                if prospective_cash < required_margin - 1e-9:
                    rejection_records.append({
                        "timestamp": date_str,
                        "underlying": underlying,
                        "action": action,
                        "reason": "insufficient_buying_power",
                        "cash_after_premium": round(prospective_cash, 4),
                        "required_margin": round(required_margin, 4),
                        "legs": len(prepared_legs),
                    })
                    continue

            for prepared in prepared_legs:
                leg = prepared["raw"]
                leg_type = prepared["type"]
                strike = prepared["strike"]
                expiry = prepared["expiry"]
                qty = prepared["qty"]
                opt_price = prepared["price"]

                if action == "open":
                    # Open: long pays premium, short receives premium
                    abs_cost = opt_price * abs(qty) * contract_multiplier
                    if qty > 0:
                        cash -= abs_cost * (1 + commission)
                    else:
                        cash += abs_cost * (1 - commission)

                    positions.append(OptionPosition(
                        option_type=leg_type,
                        strike=strike,
                        expiry=expiry,
                        qty=qty,
                        entry_price=opt_price,
                        entry_date=date_str,
                        underlying_code=underlying,
                    ))

                    trade_records.append({
                        "timestamp": date_str,
                        "code": underlying,
                        "option_type": leg_type,
                        "strike": strike,
                        "expiry": expiry,
                        "side": "buy" if qty > 0 else "sell",
                        "price": round(opt_price, 4),
                        "qty": qty,
                        "pnl": 0.0,
                        "entry_date": date_str,
                    })

                elif action == "close":
                    # Close: find matching position, honoring a partial-close qty.
                    matched = _find_matching_position(
                        positions, underlying, leg_type, strike, expiry)
                    if matched:
                        # An explicit leg ``qty`` closes only that many contracts
                        # (clamped to the open size); a close leg with no ``qty``
                        # closes the whole lot (legacy behavior). Cash/PnL and the
                        # remaining position all scale to the amount actually closed
                        # so a partial close no longer flattens the lot (#577).
                        requested = leg.get("qty")
                        full_mag = abs(matched.qty)
                        close_mag = full_mag if requested is None else min(abs(requested), full_mag)
                        if close_mag <= 0:
                            continue
                        sign = 1 if matched.qty > 0 else -1
                        closed_qty = sign * close_mag
                        remaining_qty = matched.qty - closed_qty

                        pnl = (opt_price - matched.entry_price) * closed_qty * contract_multiplier
                        abs_close = opt_price * close_mag * contract_multiplier
                        if matched.qty > 0:
                            # Long close: sell to recover
                            cash += abs_close * (1 - commission)
                        else:
                            # Short close: buy back
                            cash -= abs_close * (1 + commission)

                        trade_records.append({
                            "timestamp": date_str,
                            "code": underlying,
                            "option_type": leg_type,
                            "strike": strike,
                            "expiry": expiry,
                            "side": "close",
                            "price": round(opt_price, 4),
                            "qty": closed_qty,
                            "pnl": round(pnl, 4),
                            "entry_date": matched.entry_date,
                        })
                        if abs(remaining_qty) < 1e-9:
                            positions.remove(matched)
                        else:
                            # Reduce the open lot to its remainder instead of
                            # removing it (new object; positions stay immutable).
                            positions[positions.index(matched)] = OptionPosition(
                                option_type=matched.option_type,
                                strike=matched.strike,
                                expiry=matched.expiry,
                                qty=remaining_qty,
                                entry_price=matched.entry_price,
                                entry_date=matched.entry_date,
                                underlying_code=matched.underlying_code,
                            )

        # 4. Compute portfolio mark-to-market value and Greeks
        portfolio_value = cash
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0

        for pos in positions:
            spot = spot_prices.get(pos.underlying_code, 0.0)
            iv_val = ivs.get(pos.underlying_code, 0.3)
            T = pos.time_to_expiry(ts)

            mark_iv = leg_iv(spot, pos.strike, iv_val, iv_skew, iv_curvature)

            mark_price = bs_price(spot, pos.strike, T, risk_free_rate, mark_iv, pos.option_type)
            portfolio_value += mark_price * pos.qty * contract_multiplier

            greeks = bs_greeks(spot, pos.strike, T, risk_free_rate, mark_iv, pos.option_type)
            total_delta += greeks["delta"] * pos.qty * contract_multiplier
            total_gamma += greeks["gamma"] * pos.qty * contract_multiplier
            total_theta += greeks["theta"] * pos.qty * contract_multiplier
            total_vega += greeks["vega"] * pos.qty * contract_multiplier
            total_rho += greeks["rho"] * pos.qty * contract_multiplier

        equity_records.append({
            "timestamp": date_str,
            "equity": round(portfolio_value, 4),
            "cash": round(cash, 4),
            "positions_value": round(portfolio_value - cash, 4),
        })

        greeks_records.append({
            "timestamp": date_str,
            "delta": round(total_delta, 6),
            "gamma": round(total_gamma, 6),
            "theta": round(total_theta, 6),
            "vega": round(total_vega, 6),
            "rho": round(total_rho, 6),
            "num_positions": len(positions),
        })

    # Compute metrics
    equity_df = pd.DataFrame(equity_records)
    if equity_df.empty:
        print(json.dumps({"error": "No equity data generated"}))
        sys.exit(1)

    equity_series = equity_df.set_index("timestamp")["equity"]
    metrics = _calc_options_metrics(equity_series, initial_cash, trade_records, bars_per_year)

    # Write artifacts
    out = run_dir / "artifacts"
    out.mkdir(parents=True, exist_ok=True)

    for code, df in data_map.items():
        df.to_csv(out / f"ohlcv_{code}.csv")

    equity_df.to_csv(out / "equity.csv", index=False)

    trade_cols = ["timestamp", "code", "option_type", "strike", "expiry",
                  "side", "price", "qty", "pnl", "entry_date"]
    pd.DataFrame(trade_records or [], columns=trade_cols).to_csv(
        out / "trades.csv", index=False)

    pd.DataFrame(greeks_records).to_csv(out / "greeks.csv", index=False)
    pd.DataFrame(
        rejection_records,
        columns=["timestamp", "underlying", "action", "reason", "cash_after_premium", "required_margin", "legs"],
    ).to_csv(out / "rejections.csv", index=False)
    if rejection_records:
        metrics.setdefault("warnings", []).append(
            f"Rejected {len(rejection_records)} option structure(s) for insufficient buying power."
        )
    pd.DataFrame([metrics]).to_csv(out / "metrics.csv", index=False)

    from backtest.run_card import write_run_card
    write_run_card(
        run_dir,
        config,
        metrics,
        data_sources=[str(getattr(loader, "name", config.get("source", "")))],
        strategy_path=run_dir / "code" / "signal_engine.py",
        warnings=config.get("content_filter_warnings") or None,
    )

    print(json.dumps(metrics, indent=2, allow_nan=False))
    return metrics


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _find_matching_position(
    positions: List[OptionPosition],
    underlying: str,
    option_type: str,
    strike: float,
    expiry: str,
) -> Optional[OptionPosition]:
    """Find a matching open position.

    Args:
        positions: Current open positions.
        underlying: Underlying instrument code.
        option_type: Option type, already folded by ``normalise_option_type``;
            ``OptionPosition`` folds its own, so both sides compare in lower
            case.
        strike: Strike price.
        expiry: Expiry date string.

    Returns:
        Matching position, or None if not found.
    """
    expiry_ts = pd.Timestamp(expiry)
    for pos in positions:
        if (pos.underlying_code == underlying
                and pos.option_type == option_type
                and abs(pos.strike - strike) < 1e-6
                and pos.expiry == expiry_ts):
            return pos
    return None


def _calc_options_metrics(
    equity: pd.Series,
    initial_cash: float,
    trades: List[Dict[str, Any]],
    bars_per_year: int = 252,
) -> Dict[str, Any]:
    """Calculate options backtest metrics.

    Args:
        equity: Equity series.
        initial_cash: Initial capital.
        trades: List of trade records.
        bars_per_year: Bars per year.

    Returns:
        Metrics dictionary.
    """
    warnings: List[str] = []
    n = len(equity)
    equity_vals = pd.to_numeric(equity, errors="coerce").astype(float)
    path_is_finite = bool(n and np.isfinite(equity_vals.to_numpy()).all())

    final_raw: float | None = None
    final_value: float | None = None
    if n:
        terminal = float(equity_vals.iloc[-1])
        if np.isfinite(terminal):
            final_raw = terminal
            final_value = round(terminal, 2)
        else:
            warnings.append(
                "Final equity is non-finite; final and return metrics are undefined."
            )
    else:
        warnings.append(
            "No equity observations were produced; equity metrics are undefined."
        )

    valid_initial_cash = np.isfinite(initial_cash) and initial_cash > 0
    total_ret: float | None = None
    if final_raw is not None and valid_initial_cash:
        total_ret = final_raw / float(initial_cash) - 1
    elif final_raw is not None:
        warnings.append(
            "Total return is undefined because initial cash is not positive and finite."
        )

    ann_ret: float | None = None
    if n < 2:
        warnings.append("Annual return requires at least two equity observations.")
    elif total_ret is None:
        warnings.append(
            "Annual return is undefined because total return is unavailable."
        )
    elif final_raw is not None and final_raw < 0:
        warnings.append("Annual return is undefined when final equity is negative.")
    elif bars_per_year <= 0:
        warnings.append(
            "Annual return is undefined because bars_per_year is not positive."
        )
    else:
        growth = final_raw / float(initial_cash)
        # Explosive paths (e.g. 1m bars) can OverflowError before isfinite.
        try:
            candidate = float(growth ** (bars_per_year / (n - 1)) - 1)
        except OverflowError:
            candidate = float("inf")
        if np.isfinite(candidate):
            ann_ret = candidate
        else:
            warnings.append("Annual return is non-finite for this equity path.")

    returns: pd.Series | None = None
    if n >= 2 and path_is_finite:
        candidate_returns = equity_vals.pct_change(fill_method=None).iloc[1:]
        if np.isfinite(candidate_returns.to_numpy()).all():
            returns = candidate_returns
        else:
            warnings.append(
                "Risk ratios are undefined because equity returns are non-finite."
            )
    elif n >= 2:
        warnings.append(
            "Path-dependent metrics are undefined because equity contains non-finite values."
        )

    max_dd: float | None = None
    if path_is_finite:
        peak = equity_vals.cummax()
        if bool((peak > 0).all()):
            dd = (equity_vals - peak) / peak
            max_dd = float(dd.min())
        else:
            warnings.append(
                "Maximum drawdown is undefined because peak equity is not positive."
            )

    sharpe: float | None = None
    if returns is not None and len(returns) > 1 and bars_per_year > 0:
        vol = float(returns.std())
        if np.isfinite(vol) and vol > 1e-12:
            sharpe = float(returns.mean() / vol * np.sqrt(bars_per_year))
        else:
            warnings.append(
                "Sharpe ratio is undefined because return volatility is zero."
            )
    elif bars_per_year <= 0:
        warnings.append("Sharpe ratio requires a positive bars_per_year value.")
    else:
        warnings.append("Sharpe ratio requires at least two finite returns.")

    calmar: float | None = None
    if ann_ret is not None and max_dd is not None and abs(max_dd) > 1e-12:
        calmar = ann_ret / abs(max_dd)
    else:
        warnings.append(
            "Calmar ratio requires a defined annual return and a nonzero drawdown."
        )

    sortino: float | None = None
    if returns is not None and bars_per_year > 0:
        downside = returns[returns < 0]
        if len(downside) > 1:
            downside_std = float(downside.std())
            if np.isfinite(downside_std) and downside_std > 1e-12:
                sortino = float(returns.mean() / downside_std * np.sqrt(bars_per_year))
        if sortino is None:
            warnings.append(
                "Sortino ratio requires at least two varying downside returns."
            )
    elif bars_per_year <= 0:
        warnings.append("Sortino ratio requires a positive bars_per_year value.")
    else:
        warnings.append("Sortino ratio requires finite returns.")

    # Trade statistics
    closed_pnl: List[float] = []
    ignored_pnl_records = 0
    for t in trades:
        raw_pnl = t.get("pnl")
        if raw_pnl is None:
            ignored_pnl_records += 1
            continue
        try:
            val = float(raw_pnl)
        except (TypeError, ValueError):
            ignored_pnl_records += 1
            continue
        if not np.isfinite(val):
            ignored_pnl_records += 1
            continue
        if val != 0:
            closed_pnl.append(val)
    if ignored_pnl_records:
        warnings.append(
            f"Ignored PnL for {ignored_pnl_records} trade records "
            "(missing or non-numeric pnl); win rate and profit/loss ratio "
            "are computed from the remaining trades only."
        )
    wins = [p for p in closed_pnl if p > 0]
    losses = [p for p in closed_pnl if p < 0]
    win_rate = len(wins) / len(closed_pnl) if closed_pnl else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 1e-10
    pl_ratio = avg_win / avg_loss if avg_loss > 1e-10 else 0.0

    return {
        "final_value": final_value,
        "total_return": round(total_ret, 6) if total_ret is not None else None,
        "annual_return": round(ann_ret, 6) if ann_ret is not None else None,
        "max_drawdown": round(max_dd, 6) if max_dd is not None else None,
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "calmar": round(calmar, 4) if calmar is not None else None,
        "sortino": round(sortino, 4) if sortino is not None else None,
        "trade_count": len(trades),
        "win_rate": round(win_rate, 4),
        "profit_loss_ratio": round(pl_ratio, 4),
        "warnings": warnings,
    }
