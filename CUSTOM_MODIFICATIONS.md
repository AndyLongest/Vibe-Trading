# Custom modifications

This file records local/product changes maintained on the `custom-dashboard`
branch. Add a dated entry whenever the custom distribution changes. Keep the
upstream project history in `CHANGELOG.md`; do not mix upstream release notes
into this ledger.

## 2026-08-13 — Options backtest integrity and reporting

### Execution and accounting

- Removed look-ahead leakage from historical-volatility warm-up. Early bars now
  use only information available at that date and a neutral fallback, never a
  future 30-day estimate.
- Added strict validation for `iv_source`, exercise style, contract multiplier,
  and margin parameters. Unsupported volatility sources now fail explicitly.
- Added capital checks for long-option premium and conservative initial margin
  for naked short options.
- Made each multi-leg opening instruction atomic: all legs fill or the complete
  structure is rejected.
- Added `artifacts/rejections.csv` and warnings for structures rejected because
  buying power was insufficient.
- Corrected non-trading-day expiry settlement to use the last underlying close
  on or before expiry rather than a post-expiry trading-day close.
- Kept the engine explicitly identified as synthetic Black–Scholes pricing; it
  does not claim to replay historical option quotes.

### Dashboard and API

- Added API delivery for `greeks.csv` and `rejections.csv`.
- Added an options-specific Dashboard section for Delta, Gamma, Theta and Vega.
- Added a rejected-structure table and a visible synthetic-pricing disclosure.
- Derived return and drawdown series from equity when option artifacts do not
  carry the stock engine's `ret` and `drawdown` columns.

### Validation

- Added regressions for volatility look-ahead, unaffordable long options,
  naked-short margin, multi-leg atomicity and unsupported IV sources.

## 2026-08-12 — Upstream synchronization and complete custom dashboard

### Upstream integration

- Fast-forwarded local `main` to official commit `84155b9` and merged it into
  `custom-dashboard` while retaining local functionality.
- Reinstalled the package in editable mode in the `vibetrading` Conda
  environment and rebuilt the frontend.

### Backtest correctness

- Changed `positions.csv` to contain actual post-fill, mark-to-market weights.
- Added `target_positions.csv` for optimizer-requested weights.
- Routed invested-weight metrics and risk x-ray calculations through actual
  positions instead of optimizer intent.
- Adopted upstream atomic target rebalancing and made generated target-weight
  strategies select `position_adjustment: "rebalance"`.
- Submitted the focused upstream contribution as
  [HKUDS/Vibe-Trading#1082](https://github.com/HKUDS/Vibe-Trading/pull/1082).

### Grounding correctness

- Prevented algebraic identity constants such as `1 - cost_rate` and
  `1 - 成本率` from being misclassified as unverified market prices.
- Kept genuine statements such as `closing price = 1 CNY` subject to grounding.
- Submitted the focused upstream contribution as
  [HKUDS/Vibe-Trading#1083](https://github.com/HKUDS/Vibe-Trading/pull/1083).

### Strategy research Dashboard

- Added a report-style backtest Dashboard with headline KPIs, strategy versus
  benchmark equity, drawdown and rolling risk, realized trade P&L, trade ledger
  and complete metrics.
- Added automatic local report-server startup and direct Dashboard links after
  CLI/web backtest completion.
- Stabilized report identity when the agent executes a backtest in a detached
  allowed run directory by archiving deterministic artifacts into the active run.
- Replaced raw prompt/run-ID headings with report titles derived from strategy,
  symbols and run-card metadata; retained the run ID as secondary traceability.
- Submitted the focused upstream contribution as
  [HKUDS/Vibe-Trading#1084](https://github.com/HKUDS/Vibe-Trading/pull/1084).

### Factor research Dashboard

- Added factor-level rows to alpha benchmark results and optional factor-ID
  selection.
- Added IC–IR map, theme status distribution, top/bottom evidence tables,
  complete diagnostics and method-boundary explanations.

### Distribution

- Published the full integrated branch at
  `AndyLongest/Vibe-Trading:custom-dashboard`.
- Saved the original reference-dashboard design inventory in
  `dashboard_style.txt`.
