"""Research-only fifteen-minute specialization of the causal intraday audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .five_minute_intraday_backtest import (
    FiveMinuteForecastBacktestResult,
    _write_markdown_report,
    reconcile_intraday_forecast_outcomes,
    run_intraday_forecast_backtest,
)

FIFTEEN_MINUTE_LOOKBACK_OBSERVATIONS = 520
FIFTEEN_MINUTE_MIN_COVARIANCE_SCENARIOS = 200


def run_fifteen_minute_forecast_backtest(
    minute_bars: pd.DataFrame,
    **kwargs: object,
) -> FiveMinuteForecastBacktestResult:
    """Run the same causal audit at a 15-minute interval.

    A regular session has 26 non-overlapping 15-minute outcomes, so the default
    520-observation lookback spans roughly twenty sessions.  The 200-scenario
    covariance minimum is intentionally independent from the five-minute
    audit's 500-scenario setting.
    """
    options: dict[str, object] = {
        "lookback_observations": FIFTEEN_MINUTE_LOOKBACK_OBSERVATIONS,
        "min_covariance_scenarios": FIFTEEN_MINUTE_MIN_COVARIANCE_SCENARIOS,
    }
    options.update(kwargs)
    return run_intraday_forecast_backtest(minute_bars, interval_minutes=15, **options)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a causal research-only fifteen-minute intraday forecast audit.")
    parser.add_argument("--input", required=True, help="CSV or CSV.GZ with timestamp,symbol,close one-minute bars")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--session-date", help="YYYY-MM-DD; defaults to the latest complete session in the input")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-weight", type=float, default=0.10)
    parser.add_argument("--risk-aversion", type=float, default=10.0)
    parser.add_argument("--lookback-observations", type=int, default=FIFTEEN_MINUTE_LOOKBACK_OBSERVATIONS)
    parser.add_argument("--min-training-sessions", type=int, default=10)
    parser.add_argument("--min-covariance-scenarios", type=int, default=FIFTEEN_MINUTE_MIN_COVARIANCE_SCENARIOS)
    parser.add_argument("--seasonality-prior-observations", type=int, default=20)
    parser.add_argument(
        "--fallback-input",
        help="Optional one-minute fallback bars used only to repair otherwise-missing realized endpoints",
    )
    args = parser.parse_args()
    result = run_fifteen_minute_forecast_backtest(
        pd.read_csv(args.input),
        session_date=args.session_date,
        top_n=args.top_n,
        max_weight=args.max_weight,
        risk_aversion=args.risk_aversion,
        lookback_observations=args.lookback_observations,
        min_training_sessions=args.min_training_sessions,
        min_covariance_scenarios=args.min_covariance_scenarios,
        seasonality_prior_observations=args.seasonality_prior_observations,
    )
    if args.fallback_input:
        result = reconcile_intraday_forecast_outcomes(
            result,
            pd.read_csv(args.fallback_input),
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.ledger.to_csv(output / "fifteen_minute_forecast_ledger.csv", index=False)
    result.holdings.to_csv(output / "fifteen_minute_forecast_holdings.csv", index=False)
    (output / "fifteen_minute_forecast_summary.json").write_text(json.dumps(result.summary, indent=2) + "\n")
    _write_markdown_report(output / "fifteen_minute_forecast_report.md", result)
    print(
        f"Wrote {len(result.ledger)} fifteen-minute decision rows and {len(result.holdings)} holdings rows "
        f"for {result.summary['evaluated_session_date']}"
    )


if __name__ == "__main__":
    main()
