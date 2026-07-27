"""Produce reproducible research weights from a local minute-bar file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .optimize import mean_cvar_weights
from .returns import daily_returns_from_minute_bars


def run_research(
    minute_bars: pd.DataFrame,
    *,
    min_minutes_per_session: int = 300,
    max_weight: float = 0.05,
    risk_aversion: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return daily log returns, simple returns, and a research target-weight table.

    This is an allocation research result only. It does not know account value,
    tax constraints, borrow availability, or suitability and cannot submit an
    order.
    """
    daily_log, daily_simple = daily_returns_from_minute_bars(
        minute_bars, min_minutes_per_session=min_minutes_per_session
    )
    complete = daily_simple.dropna(axis=1, thresh=max(20, len(daily_simple) // 2)).dropna(axis=0, how="any")
    if len(complete) < 20 or complete.shape[1] < 2:
        raise ValueError("need at least 20 complete daily observations for two or more assets")
    if max_weight * complete.shape[1] < 1:
        raise ValueError("max_weight is too small for the number of usable assets")
    result = mean_cvar_weights(
        complete, max_weight=max_weight, risk_aversion=risk_aversion
    )
    weights = result.weights.rename("target_weight").sort_values(ascending=False).reset_index(names="symbol")
    weights["solver_status"] = result.status
    weights["historical_expected_daily_return"] = result.expected_return
    weights["historical_cvar"] = result.cvar
    return daily_log, daily_simple, weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Create research target weights from one-minute bars.")
    parser.add_argument("--input", required=True, help="minute-bar CSV: timestamp,symbol,close")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--risk-aversion", type=float, default=5.0)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs, simple, weights = run_research(
        pd.read_csv(args.input), max_weight=args.max_weight, risk_aversion=args.risk_aversion
    )
    logs.to_csv(output_dir / "daily_asset_log_returns.csv")
    simple.to_csv(output_dir / "daily_asset_simple_returns.csv")
    weights.to_csv(output_dir / "research_target_weights.csv", index=False)
    print(f"Wrote {len(weights)} research weights to {output_dir / 'research_target_weights.csv'}")


if __name__ == "__main__":
    main()
