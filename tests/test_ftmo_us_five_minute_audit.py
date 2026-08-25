from __future__ import annotations

import lzma
import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest

from cufolio_cpu.ftmo_us_five_minute_audit import (
    M1_CANDLE_STRUCT,
    _parse_native_m1_close,
    load_ftmo_us_mapping,
    native_m1_to_five_minute_quotes,
    native_m1_to_timeframe_quotes,
    run_ftmo_us_five_minute_audit,
    run_ftmo_us_no_rebalance_audit,
    run_ftmo_us_timeframe_audit,
    select_verified_assets,
)


def _native_endpoint_rows(periods: int = 900) -> pd.DataFrame:
    """Synthetic native M1 endpoints; one final closed minute per M5 bucket."""
    rng = np.random.default_rng(418)
    symbols = ["EURUSD.SIM", "GBPUSD.SIM", "USDJPY.SIM", "US500.SIM", "XAUUSD.SIM"]
    endpoints = pd.date_range("2026-01-01 00:04", periods=periods, freq="5min", tz="UTC")
    rows: list[dict[str, object]] = []
    for number, symbol in enumerate(symbols):
        returns = rng.normal(0.00001 * (number + 1), 0.00016, periods)
        mid = (100.0 + number * 20.0) * np.exp(np.cumsum(returns))
        spread = 0.00006 * (number + 1)
        rows.extend(
            {
                "timestamp": timestamp,
                "ftmo_symbol": symbol,
                "bid_close": float(value * (1 - spread / 2)),
                "ask_close": float(value * (1 + spread / 2)),
            }
            for timestamp, value in zip(endpoints, mid, strict=True)
        )
    return pd.DataFrame(rows)


def test_native_endpoint_conversion_requires_the_exact_final_m1_close() -> None:
    minute = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T00:03:00Z", "ftmo_symbol": "EURUSD.SIM", "bid_close": 1.1, "ask_close": 1.1001},
            {"timestamp": "2026-01-01T00:04:00Z", "ftmo_symbol": "EURUSD.SIM", "bid_close": 1.2, "ask_close": 1.2001},
            {"timestamp": "2026-01-01T00:08:00Z", "ftmo_symbol": "EURUSD.SIM", "bid_close": 1.3, "ask_close": 1.3001},
        ]
    )

    converted = native_m1_to_five_minute_quotes(minute)

    assert len(converted) == 1
    assert converted.loc[0, "timestamp"] == pd.Timestamp("2026-01-01T00:05:00Z")
    assert converted.loc[0, "bid_close"] == 1.2
    assert converted.loc[0, "ask_close"] == 1.2001


def test_hourly_endpoint_conversion_requires_the_final_minute_of_each_hour() -> None:
    minute = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T00:58:00Z", "ftmo_symbol": "EURUSD.SIM", "bid_close": 1.1, "ask_close": 1.1001},
            {"timestamp": "2026-01-01T00:59:00Z", "ftmo_symbol": "EURUSD.SIM", "bid_close": 1.2, "ask_close": 1.2001},
            {"timestamp": "2026-01-01T01:58:00Z", "ftmo_symbol": "EURUSD.SIM", "bid_close": 1.3, "ask_close": 1.3001},
        ]
    )

    converted = native_m1_to_timeframe_quotes(minute, timeframe="H1")

    assert len(converted) == 1
    assert converted.loc[0, "timestamp"] == pd.Timestamp("2026-01-01T01:00:00Z")
    assert converted.loc[0, "bid_close"] == 1.2


@pytest.mark.parametrize(
    ("timeframe", "frequency", "periods", "lookback", "minimum", "evaluation_start", "evaluation_end"),
    [
        ("H1", "1h", 900, 720, 250, "2026-02-01", "2026-02-05"),
        ("H4", "4h", 480, 360, 100, "2026-03-05", "2026-03-10"),
        ("D1", "1D", 100, 60, 20, "2026-03-05", "2026-03-20"),
    ],
)
def test_interval_audits_and_no_rebalance_holdings_are_causal(
    timeframe: str,
    frequency: str,
    periods: int,
    lookback: int,
    minimum: int,
    evaluation_start: str,
    evaluation_end: str,
) -> None:
    rng = np.random.default_rng(74)
    symbols = ["EURUSD.SIM", "GBPUSD.SIM", "USDJPY.SIM", "US500.SIM", "XAUUSD.SIM"]
    timestamps = pd.date_range("2026-01-01", periods=periods, freq=frequency, tz="UTC")
    rows: list[dict[str, object]] = []
    for number, symbol in enumerate(symbols):
        mids = (100.0 + 10.0 * number) * np.exp(np.cumsum(rng.normal(0.00002, 0.0002, periods)))
        spread = 0.0001 * (number + 1)
        rows.extend(
            {
                "timestamp": timestamp,
                "ftmo_symbol": symbol,
                "bid_close": float(mid * (1 - spread / 2)),
                "ask_close": float(mid * (1 + spread / 2)),
                "mid_close": float(mid),
            }
            for timestamp, mid in zip(timestamps, mids, strict=True)
        )
    quotes = pd.DataFrame(rows)
    rebalanced = run_ftmo_us_timeframe_audit(
        quotes,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        timeframe=timeframe,
        top_n=5,
        max_weight=0.20,
        lookback_windows=lookback,
        min_training_windows=minimum,
    )
    realized = rebalanced.ledger.loc[rebalanced.ledger["realized_status"].eq("ok")]
    assert not realized.empty
    assert rebalanced.summary["timeframe"] == timeframe
    no_rebalance = run_ftmo_us_no_rebalance_audit(
        quotes,
        rebalanced,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        timeframe=timeframe,
    )
    assert no_rebalance.summary["portfolio_mode"] == "no_rebalance_buy_and_hold"
    assert len(no_rebalance.holdings) == 5


def test_native_m1_offset_uses_explicit_seconds_without_a_deprecation_warning() -> None:
    raw = M1_CANDLE_STRUCT.pack(60, 110_000, 110_025, 109_990, 110_030, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        rows = _parse_native_m1_close(lzma.compress(raw), source_day=date(2026, 7, 1), price_divisor=100_000)

    assert rows == [(pd.Timestamp("2026-07-01T00:01:00Z"), 1.10025)]


def test_ftmo_us_mapping_matches_the_current_us_universe_shape() -> None:
    mapping = load_ftmo_us_mapping()
    manifest = pd.DataFrame(
        {
            "code": mapping["ftmo_symbol"],
            "assetClass": mapping["asset_class"],
            "name": mapping["ftmo_symbol"],
        }
    )
    selected = select_verified_assets(mapping, manifest)

    assert len(mapping) == 49
    assert mapping["asset_class"].value_counts().to_dict() == {"fx": 42, "commodity": 3, "index": 3, "crypto": 1}
    assert set(selected["ftmo_symbol"]) == set(mapping["ftmo_symbol"])


def test_five_minute_ftmo_proxy_audit_is_causal_and_applies_bid_ask_execution() -> None:
    quotes = native_m1_to_five_minute_quotes(_native_endpoint_rows())
    options = {
        "evaluation_start": "2026-01-03",
        "evaluation_end": "2026-01-03",
        "top_n": 5,
        "max_weight": 0.20,
        "lookback_windows": 300,
        "min_training_windows": 100,
    }
    baseline = run_ftmo_us_five_minute_audit(quotes, **options)

    ledger = baseline.ledger
    realized = ledger.loc[ledger["realized_status"].eq("ok")].copy()
    assert not realized.empty
    assert (pd.to_datetime(ledger.loc[ledger["forecast_status"].eq("ok"), "training_end"], utc=True) < pd.to_datetime(ledger.loc[ledger["forecast_status"].eq("ok"), "decision_timestamp"], utc=True)).all()
    assert baseline.summary["costs_included"] == "Dukascopy proxy BID/ASK spread only"
    assert baseline.summary["ftmo_execution_claim"] is False

    first = realized.iloc[0]
    first_holdings = baseline.holdings.loc[
        baseline.holdings["decision_timestamp"].eq(first["decision_timestamp"])
    ].copy()
    expected_executable = float(
        (first_holdings["target_weight"] * first_holdings["realized_executable_proxy_asset_log_return"]).sum()
    )
    assert np.isclose(first["actual_executable_proxy_log_return"], expected_executable)
    assert first["actual_executable_proxy_log_return"] < first["actual_mid_portfolio_log_return"]

    no_rebalance = run_ftmo_us_no_rebalance_audit(
        quotes,
        baseline,
        evaluation_start=options["evaluation_start"],
        evaluation_end=options["evaluation_end"],
        timeframe="M5",
    )
    assert no_rebalance.summary["portfolio_mode"] == "no_rebalance_buy_and_hold"
    assert len(no_rebalance.ledger) == 1
    assert len(no_rebalance.holdings) == 5
    assert no_rebalance.summary["hold_entry_timestamp"] == first["decision_timestamp"]

    # Altering prices after the first forecast horizon must not alter its
    # forecast weights or predicted returns.  It may change realized outcomes.
    changed = quotes.copy()
    future = pd.to_datetime(changed["timestamp"], utc=True) > pd.Timestamp(first["target_end"])
    changed.loc[future, ["bid_close", "ask_close", "mid_close"]] *= 1.5
    rerun = run_ftmo_us_five_minute_audit(changed, **options)
    rerun_holdings = rerun.holdings.loc[
        rerun.holdings["decision_timestamp"].eq(first["decision_timestamp"]),
        ["symbol", "target_weight", "predicted_asset_log_return"],
    ].sort_values("symbol").reset_index(drop=True)
    baseline_holdings = first_holdings.loc[:, ["symbol", "target_weight", "predicted_asset_log_return"]].sort_values(
        "symbol"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline_holdings, rerun_holdings)
