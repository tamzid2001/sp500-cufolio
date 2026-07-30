from __future__ import annotations

import numpy as np
import pandas as pd

from cufolio_cpu.ftmo_us_five_minute_audit import (
    load_ftmo_us_mapping,
    native_m1_to_five_minute_quotes,
    run_ftmo_us_five_minute_audit,
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
