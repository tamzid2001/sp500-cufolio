import numpy as np
import pandas as pd
import pytest

from cufolio_cpu.returns import daily_returns_from_minute_bars, portfolio_daily_returns


def test_intraday_logs_sum_without_overnight_return() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-02T14:30:00Z",  # 09:30 ET
                "2026-01-02T14:31:00Z",
                "2026-01-02T14:32:00Z",
                "2026-01-05T14:30:00Z",  # next session; no overnight return
                "2026-01-05T14:31:00Z",
            ],
            "symbol": ["AAPL"] * 5,
            "close": [100.0, 101.0, 102.0, 200.0, 202.0],
        }
    )
    logs, simple = daily_returns_from_minute_bars(bars, min_minutes_per_session=1)
    assert logs.iloc[0, 0] == pytest.approx(np.log(102 / 100))
    assert logs.iloc[1, 0] == pytest.approx(np.log(202 / 200))
    assert simple.iloc[1, 0] == pytest.approx(0.01)


def test_insufficient_minute_coverage_is_not_zero() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": ["2026-01-02T14:30:00Z", "2026-01-02T14:31:00Z"],
            "symbol": ["AAPL", "AAPL"],
            "close": [100, 101],
        }
    )
    logs, _ = daily_returns_from_minute_bars(bars, min_minutes_per_session=2)
    assert pd.isna(logs.iloc[0, 0])


def test_portfolio_weights_are_lagged_by_default() -> None:
    index = pd.to_datetime(["2026-01-02", "2026-01-05"])
    returns = pd.DataFrame({"A": [0.10, 0.00], "B": [0.00, 0.10]}, index=index)
    weights = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=index)
    result = portfolio_daily_returns(returns, weights)
    assert result.loc[index[0], "portfolio_simple_return"] == pytest.approx(0.0)
    assert result.loc[index[1], "portfolio_simple_return"] == pytest.approx(0.0)


def test_held_asset_missing_return_marks_portfolio_return_missing() -> None:
    index = pd.to_datetime(["2026-01-02", "2026-01-05"])
    returns = pd.DataFrame({"A": [0.01, np.nan], "B": [0.0, 0.01]}, index=index)
    result = portfolio_daily_returns(returns, pd.Series({"A": 1.0, "B": 0.0}), lag_weights=False)
    assert pd.isna(result.loc[index[1], "portfolio_simple_return"])
