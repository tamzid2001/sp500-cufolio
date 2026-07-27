"""Research-only forward intraday return ranking with purged time-series validation.

This module deliberately produces candidate portfolio *research* weights, never
orders. Its requested horizon is measured in trading minutes.  With 15-minute
bars, a 500-minute request is represented by 34 bars (510 trading minutes) and
is disclosed as such rather than silently rounded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .optimize import forecast_mean_variance_weights

NEW_YORK = ZoneInfo("America/New_York")
INTERVAL_MINUTES = {"1m": 1, "15m": 15}
FEATURE_COLUMNS = ("return_1", "return_4", "return_13", "return_26", "volatility_26")


@dataclass(frozen=True)
class ForwardModelResult:
    validation: pd.DataFrame
    portfolio: pd.DataFrame
    status: dict[str, object]


def _regular_session_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "symbol", "close"}
    if missing := required.difference(bars.columns):
        raise ValueError(f"intraday bars are missing required columns: {sorted(missing)}")
    clean = bars.loc[:, ["timestamp", "symbol", "close"]].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")
    clean["symbol"] = clean["symbol"].astype(str).str.upper().str.strip()
    clean["close"] = pd.to_numeric(clean["close"], errors="coerce")
    clean = clean.dropna(subset=["timestamp", "close"])
    clean = clean[clean["close"] > 0].drop_duplicates(["symbol", "timestamp"], keep="last")
    local = clean["timestamp"].dt.tz_convert(NEW_YORK)
    clock = local.dt.time
    regular = (clock >= pd.Timestamp("09:30").time()) & (clock <= pd.Timestamp("16:00").time())
    clean = clean.loc[regular].copy()
    clean["session_date"] = local.loc[regular].dt.tz_localize(None).dt.normalize()
    if clean.empty:
        raise ValueError("no positive bars fall in the US regular session")
    return clean.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def build_forward_dataset(
    bars: pd.DataFrame, *, interval: str, horizon_minutes: int = 500
) -> tuple[pd.DataFrame, int, int]:
    """Create non-overlapping-feature rows and their future trading-bar targets.

    A target's timestamp is retained so validation can purge every training
    observation whose outcome would overlap the future validation block.
    """
    if interval not in INTERVAL_MINUTES:
        raise ValueError(f"unsupported interval {interval!r}")
    if horizon_minutes < 1:
        raise ValueError("horizon_minutes must be positive")
    bar_minutes = INTERVAL_MINUTES[interval]
    horizon_bars = int(np.ceil(horizon_minutes / bar_minutes))
    effective_minutes = horizon_bars * bar_minutes
    frame = _regular_session_bars(bars)
    grouped = frame.groupby("symbol", sort=False)
    frame["log_close"] = np.log(frame["close"])
    frame["return_1"] = grouped["log_close"].diff(1)
    frame["return_4"] = grouped["log_close"].diff(4)
    frame["return_13"] = grouped["log_close"].diff(13)
    frame["return_26"] = grouped["log_close"].diff(26)
    frame["volatility_26"] = frame.groupby("symbol", sort=False)["return_1"].transform(
        lambda value: value.rolling(26).std()
    )
    frame["future_close"] = grouped["close"].shift(-horizon_bars)
    frame["target_timestamp"] = grouped["timestamp"].shift(-horizon_bars)
    frame["forward_log_return"] = np.log(frame["future_close"] / frame["close"])
    dataset = frame.dropna(subset=[*FEATURE_COLUMNS, "forward_log_return", "target_timestamp"]).copy()
    return dataset, horizon_bars, effective_minutes


def _purged_walk_forward(
    dataset: pd.DataFrame, *, validation_blocks: int = 3
) -> pd.DataFrame:
    """Evaluate strictly later time blocks after purging overlapping labels."""
    timestamps = pd.DatetimeIndex(sorted(dataset["timestamp"].unique()))
    if len(timestamps) < validation_blocks * 10:
        return pd.DataFrame(columns=["fold", "train_rows", "test_rows", "mse", "spearman_ic"])
    starts = np.linspace(int(len(timestamps) * 0.55), len(timestamps) - 1, validation_blocks + 1, dtype=int)
    rows: list[dict[str, object]] = []
    for fold, (start_index, end_index) in enumerate(zip(starts[:-1], starts[1:]), start=1):
        test_start, test_end = timestamps[start_index], timestamps[end_index]
        # Labels observed on/after test_start are excluded from fitting.
        train = dataset[dataset["target_timestamp"] < test_start]
        test = dataset[(dataset["timestamp"] >= test_start) & (dataset["timestamp"] < test_end)]
        if len(train) < 100 or len(test) < 20:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(train.loc[:, FEATURE_COLUMNS], train["forward_log_return"])
        prediction = model.predict(test.loc[:, FEATURE_COLUMNS])
        correlation = spearmanr(prediction, test["forward_log_return"]).statistic
        rows.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "test_rows": len(test),
                "test_start": test_start,
                "test_end": test_end,
                "mse": float(np.mean((prediction - test["forward_log_return"].to_numpy()) ** 2)),
                "spearman_ic": float(correlation) if np.isfinite(correlation) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _select_candidate_symbols(predictions: pd.Series, *, top_n: int, max_weight: float) -> pd.Index:
    if top_n < 2:
        raise ValueError("top_n must be at least two")
    selected = predictions.dropna().sort_values(ascending=False).head(top_n)
    if len(selected) < 2:
        raise ValueError("fewer than two assets have current features")
    if max_weight * len(selected) < 1 - 1e-12:
        raise ValueError("max_weight is too small for the selected candidate count")
    return selected.index


def run_forward_research(
    bars: pd.DataFrame,
    *,
    interval: str,
    horizon_minutes: int = 500,
    top_n: int = 20,
    max_weight: float = 0.10,
    min_sessions: int = 20,
) -> ForwardModelResult:
    dataset, horizon_bars, effective_minutes = build_forward_dataset(
        bars, interval=interval, horizon_minutes=horizon_minutes
    )
    session_count = int(dataset["session_date"].nunique())
    validation = _purged_walk_forward(dataset)
    status: dict[str, object] = {
        "model_run": False,
        "requested_horizon_minutes": horizon_minutes,
        "effective_horizon_minutes": effective_minutes,
        "interval": interval,
        "horizon_bars": horizon_bars,
        "sessions_with_features": session_count,
        "minimum_sessions": min_sessions,
        "validation_folds": int(len(validation)),
        "mean_spearman_ic": float(validation["spearman_ic"].mean()) if not validation.empty else None,
    }
    if session_count < min_sessions or len(validation) < 2:
        status["reason"] = "insufficient complete regular-session history for purged validation"
        return ForwardModelResult(validation, pd.DataFrame(columns=["symbol", "target_weight"]), status)
    latest_timestamp = dataset["timestamp"].max()
    history = dataset[dataset["target_timestamp"] < latest_timestamp]
    # Build current features from the same causal feature construction, then
    # select rows at the latest feature timestamp (which have no known target).
    feature_frame = _regular_session_bars(bars)
    group = feature_frame.groupby("symbol", sort=False)
    feature_frame["log_close"] = np.log(feature_frame["close"])
    feature_frame["return_1"] = group["log_close"].diff(1)
    feature_frame["return_4"] = group["log_close"].diff(4)
    feature_frame["return_13"] = group["log_close"].diff(13)
    feature_frame["return_26"] = group["log_close"].diff(26)
    feature_frame["volatility_26"] = feature_frame.groupby("symbol", sort=False)["return_1"].transform(
        lambda value: value.rolling(26).std()
    )
    current_features = feature_frame[feature_frame["timestamp"] == latest_timestamp].dropna(subset=FEATURE_COLUMNS)
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(history.loc[:, FEATURE_COLUMNS], history["forward_log_return"])
    predicted = pd.Series(
        model.predict(current_features.loc[:, FEATURE_COLUMNS]), index=current_features["symbol"], name="predicted_forward_log_return"
    )
    selected_symbols = _select_candidate_symbols(predicted, top_n=top_n, max_weight=max_weight)
    scenarios = (
        history.pivot(index="timestamp", columns="symbol", values="forward_log_return")
        .reindex(columns=selected_symbols)
        .dropna(axis=0, how="any")
    )
    allocation = forecast_mean_variance_weights(
        scenarios,
        predicted.reindex(selected_symbols),
        risk_aversion=10.0,
        max_weight=max_weight,
    )
    weights = allocation.weights
    portfolio = (
        pd.DataFrame({"symbol": weights.index, "target_weight": weights.values})
        .assign(predicted_forward_log_return=lambda value: value["symbol"].map(predicted))
        .sort_values("target_weight", ascending=False)
        .reset_index(drop=True)
    )
    status["model_run"] = True
    status["reason"] = "research_only"
    status["objective"] = "maximize forecast return minus 10.0 times forward-return variance"
    status["optimizer_status"] = allocation.status
    status["portfolio_predicted_forward_log_return"] = allocation.expected_return
    status["forward_return_scenarios"] = int(len(scenarios))
    status["latest_feature_timestamp"] = latest_timestamp.isoformat()
    return ForwardModelResult(validation, portfolio, status)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create research-only forward intraday candidate weights.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--interval", choices=sorted(INTERVAL_MINUTES), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon-minutes", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-weight", type=float, default=0.10)
    args = parser.parse_args()
    result = run_forward_research(
        pd.read_csv(args.input),
        interval=args.interval,
        horizon_minutes=args.horizon_minutes,
        top_n=args.top_n,
        max_weight=args.max_weight,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.validation.to_csv(output_dir / "forward_500m_validation.csv", index=False)
    result.portfolio.to_csv(output_dir / "forward_500m_candidate_portfolio.csv", index=False)
    (output_dir / "forward_500m_status.json").write_text(json.dumps(result.status, indent=2) + "\n")
    print(f"Wrote {len(result.portfolio)} research-only candidate weights")


if __name__ == "__main__":
    main()
