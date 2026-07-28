from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from cufolio_cpu.five_minute_paper_daemon import rolling_history_start
from cufolio_cpu.five_minute_paper_session import (
    _completed_bar_at,
    _regular_minute_cache_projection,
    five_minute_events,
    run_five_minute_paper_session,
)


def test_five_minute_schedule_has_one_order_boundary_per_horizon_and_no_rebalance_events() -> None:
    events = five_minute_events(date(2026, 7, 28))
    targets = [at for kind, at in events if kind == "forecast_and_order"]
    assert len(targets) == 77
    assert [at.tz_convert("America/New_York").strftime("%H:%M") for at in targets[:2]] == ["09:35", "09:40"]
    assert targets[-1].tz_convert("America/New_York").strftime("%H:%M") == "15:55"
    assert [kind for kind, _ in events].count("forecast_and_order") == 77
    assert "rebalance" not in {kind for kind, _ in events}
    assert events[-1][0] == "flatten"
    assert events[-1][1].tz_convert("America/New_York").strftime("%H:%M") == "16:00"


def test_five_minute_cache_keeps_full_minutes_for_recent_sessions() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": [
                "2026-06-01T13:30:00Z", "2026-06-01T13:31:00Z",
                "2026-07-28T13:30:00Z", "2026-07-28T13:31:00Z", "2026-07-28T20:00:00Z",
            ],
            "symbol": ["AAA"] * 5,
            "close": [100, 101, 102, 103, 999],
        }
    )
    projected = _regular_minute_cache_projection(bars)
    assert list(projected["timestamp"]) == [
        pd.Timestamp("2026-06-01T13:30:00Z"), pd.Timestamp("2026-06-01T13:31:00Z"),
        pd.Timestamp("2026-07-28T13:30:00Z"), pd.Timestamp("2026-07-28T13:31:00Z"),
    ]


def test_completed_minute_never_uses_in_progress_bar() -> None:
    assert _completed_bar_at(pd.Timestamp("2026-07-28T13:35:00Z"), date(2026, 7, 28)) == pd.Timestamp("2026-07-28T13:34:00Z")
    assert _completed_bar_at(pd.Timestamp("2026-07-28T13:30:30Z"), date(2026, 7, 28)) is None


def test_five_minute_session_rejects_live_mode_without_explicit_acknowledgement(tmp_path) -> None:
    with pytest.raises(ValueError, match="allow_live_trading"):
        run_five_minute_paper_session(
            session_day=date(2026, 7, 28), history_start="2026-06-01T13:30:00Z", output_dir=tmp_path,
            mode="live",
        )


def test_five_minute_rolling_history_start_has_model_warmup() -> None:
    reference = pd.Timestamp("2026-07-28T13:35:00Z")
    assert rolling_history_start(reference, calendar_days=28) == "2026-06-30T13:30:00+00:00"
    with pytest.raises(ValueError, match="at least 28"):
        rolling_history_start(reference, calendar_days=27)
