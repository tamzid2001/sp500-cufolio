import numpy as np
import pandas as pd
import pytest

from cufolio_cpu.intraday_forward import build_forward_dataset, run_forward_research


def _fifteen_minute_bars() -> pd.DataFrame:
    rows = []
    for day_number, session in enumerate(pd.bdate_range("2026-01-02", periods=30)):
        for bar_number, timestamp in enumerate(pd.date_range(session + pd.Timedelta(hours=14, minutes=30), periods=27, freq="15min", tz="UTC")):
            for symbol, drift in [("AAA", 0.0004), ("BBB", 0.0002)]:
                rows.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "close": 100 * np.exp(drift * (day_number * 27 + bar_number)),
                    }
                )
    return pd.DataFrame(rows)


def test_forward_target_is_later_than_feature_timestamp() -> None:
    dataset, horizon_bars, effective_minutes = build_forward_dataset(
        _fifteen_minute_bars(), interval="15m", horizon_minutes=500
    )
    assert horizon_bars == 34
    assert effective_minutes == 510
    assert (dataset["target_timestamp"] > dataset["timestamp"]).all()


def test_forward_research_uses_purged_validation_and_capped_long_only_weights() -> None:
    result = run_forward_research(
        _fifteen_minute_bars(), interval="15m", horizon_minutes=500, top_n=2, max_weight=0.60
    )
    assert result.status["model_run"] is True
    assert len(result.validation) >= 2
    assert (result.validation["test_start"] >= result.validation["test_end"].shift(1)).iloc[1:].all()
    assert result.portfolio["target_weight"].sum() == pytest.approx(1.0)
    assert (result.portfolio["target_weight"] >= 0).all()
