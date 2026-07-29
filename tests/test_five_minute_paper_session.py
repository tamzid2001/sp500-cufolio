from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from cufolio_cpu.five_minute_paper_daemon import rolling_history_start
from cufolio_cpu.five_minute_paper_session import (
    _forecast_horizon_for_decision,
    _handoff_must_wait_for_flatten,
    _load_checkpoint,
    _merge_current_minutes,
    _persist_checkpoint,
    _completed_bar_at,
    _regular_minute_cache_projection,
    five_minute_events,
    run_five_minute_paper_session,
)
from cufolio_cpu.hourly_paper_session import MinuteCacheHealth


def test_five_minute_schedule_has_one_order_boundary_per_horizon_and_no_rebalance_events() -> None:
    events = five_minute_events(date(2026, 7, 28))
    targets = [at for kind, at in events if kind == "forecast_and_order"]
    assert len(targets) == 78
    assert [at.tz_convert("America/New_York").strftime("%H:%M") for at in targets[:2]] == ["09:30", "09:35"]
    assert targets[-1].tz_convert("America/New_York").strftime("%H:%M") == "15:55"
    assert [kind for kind, _ in events].count("forecast_and_order") == 78
    assert "rebalance" not in {kind for kind, _ in events}
    assert events[-1][0] == "flatten"
    assert events[-1][1].tz_convert("America/New_York").strftime("%H:%M") == "15:59"
    assert _forecast_horizon_for_decision(targets[0], date(2026, 7, 28)) == 5
    assert _forecast_horizon_for_decision(targets[-1], date(2026, 7, 28)) == 4
    assert targets[-1] + timedelta(minutes=4) == events[-1][1]
    assert _handoff_must_wait_for_flatten("flatten", events[-1][1], events[-1][1] - timedelta(minutes=1))
    assert not _handoff_must_wait_for_flatten("forecast_and_order", targets[-1], targets[-1] - timedelta(minutes=1))


def test_five_minute_cache_keeps_full_minutes_for_recent_sessions() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": [
                "2026-06-01T13:29:00Z", "2026-06-01T13:30:00Z", "2026-06-01T13:31:00Z",
                "2026-07-28T13:29:00Z", "2026-07-28T13:30:00Z", "2026-07-28T13:31:00Z", "2026-07-28T20:00:00Z",
            ],
            "symbol": ["AAA"] * 7,
            "close": [99, 100, 101, 102, 103, 104, 999],
        }
    )
    projected = _regular_minute_cache_projection(bars)
    assert list(projected["timestamp"]) == [
        pd.Timestamp("2026-06-01T13:29:00Z"), pd.Timestamp("2026-06-01T13:30:00Z"), pd.Timestamp("2026-06-01T13:31:00Z"),
        pd.Timestamp("2026-07-28T13:29:00Z"), pd.Timestamp("2026-07-28T13:30:00Z"), pd.Timestamp("2026-07-28T13:31:00Z"),
    ]


def test_full_universe_cache_retains_only_exact_five_minute_endpoints() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": [
                "2026-07-28T13:29:00Z",  # 09:29 opening anchor
                "2026-07-28T13:30:00Z",
                "2026-07-28T13:34:00Z",
                "2026-07-28T13:35:00Z",
                "2026-07-28T19:59:00Z",
            ],
            "symbol": ["AAA"] * 5,
            "close": [99, 100, 101, 102, 103],
        }
    )

    projected = _regular_minute_cache_projection(bars, endpoint_only=True)

    assert projected["timestamp"].tolist() == [
        pd.Timestamp("2026-07-28T13:29:00Z"),
        pd.Timestamp("2026-07-28T13:34:00Z"),
        pd.Timestamp("2026-07-28T19:59:00Z"),
    ]


def test_completed_minute_never_uses_in_progress_bar() -> None:
    assert _completed_bar_at(pd.Timestamp("2026-07-28T13:35:00Z"), date(2026, 7, 28)) == pd.Timestamp("2026-07-28T13:34:00Z")
    assert _completed_bar_at(pd.Timestamp("2026-07-28T13:30:00Z"), date(2026, 7, 28)) == pd.Timestamp("2026-07-28T13:29:00Z")
    assert _completed_bar_at(pd.Timestamp("2026-07-28T13:29:30Z"), date(2026, 7, 28)) is None


def test_five_minute_session_rejects_live_mode_without_explicit_acknowledgement(tmp_path) -> None:
    with pytest.raises(ValueError, match="allow_live_trading"):
        run_five_minute_paper_session(
            session_day=date(2026, 7, 28), history_start="2026-06-01T13:30:00Z", output_dir=tmp_path,
            mode="live",
        )


def test_new_session_discards_old_completed_events_but_not_the_separate_minute_cache(tmp_path) -> None:
    checkpoint = tmp_path / "state.json"
    checkpoint.write_text(
        '{"format_version": 1, "session_date": "2026-07-28", '
        '"completed_events": ["forecast_and_order:old"], "ledger": [{"event": "old"}]}',
        encoding="utf-8",
    )

    fresh = _load_checkpoint(checkpoint, date(2026, 7, 29))

    assert fresh == {
        "format_version": 1,
        "session_date": "2026-07-29",
        "completed_events": [],
        "ledger": [],
    }


def test_five_minute_checkpoint_write_is_atomic_and_immediately_restorable(tmp_path) -> None:
    checkpoint_path = tmp_path / "state.json"
    checkpoint = {"format_version": 1, "session_date": "2026-07-29", "completed_events": [], "ledger": []}

    _persist_checkpoint(
        checkpoint_path,
        checkpoint,
        {"forecast_and_order:2026-07-29T13:30:00+00:00"},
        [{"event": "five_minute_forecast_target_order"}],
    )

    assert _load_checkpoint(checkpoint_path, date(2026, 7, 29))["completed_events"] == [
        "forecast_and_order:2026-07-29T13:30:00+00:00"
    ]
    assert not list(tmp_path.glob(".state.json.tmp"))


def test_five_minute_rolling_history_start_has_model_warmup() -> None:
    reference = pd.Timestamp("2026-07-28T13:35:00Z")
    assert rolling_history_start(reference, calendar_days=28) == "2026-06-30T13:29:00+00:00"
    with pytest.raises(ValueError, match="at least 28"):
        rolling_history_start(reference, calendar_days=27)


def test_websocket_precompute_does_not_issue_redundant_rest_repair(monkeypatch) -> None:
    class Stream:
        available = True

        def completed_bars_through(self, _completed):
            return pd.DataFrame(
                {"timestamp": ["2026-07-28T14:33:00Z"], "symbol": ["AAA"], "close": [101.0]}
            )

    def unexpected_rest(*_args, **_kwargs):
        raise AssertionError("websocket-backed precompute should not perform an IEX REST repair")

    monkeypatch.setattr("cufolio_cpu.five_minute_paper_session.download_minute_bars", unexpected_rest)
    update = _merge_current_minutes(
        pd.DataFrame(
            {"timestamp": ["2026-07-28T14:32:00Z"], "symbol": ["AAA"], "close": [100.0]}
        ),
        ["AAA"],
        session_day=date(2026, 7, 28),
        observed_at=pd.Timestamp("2026-07-28T14:34:00Z"),
        cache_path=None,
        minute_stream=Stream(),
        health=MinuteCacheHealth(),
        repair_from_rest=False,
        persist=False,
    )

    assert list(update.new_rows["timestamp"]) == [pd.Timestamp("2026-07-28T14:33:00Z")]
    assert list(update.bars["close"]) == [100.0, 101.0]


def test_full_universe_threaded_latest_poll_keeps_only_exact_completed_endpoint(monkeypatch) -> None:
    def latest(_symbols):
        return pd.DataFrame(
            {
                "timestamp": ["2026-07-28T14:32:00Z", "2026-07-28T14:34:00Z"],
                "symbol": ["STALE", "FRESH"],
                "close": [99.0, 101.0],
            }
        )

    def unexpected_rest(*_args, **_kwargs):
        raise AssertionError("full-universe latest-bar polling must not fan out into historical REST repairs")

    monkeypatch.setattr("cufolio_cpu.five_minute_paper_session.download_latest_iex_minute_bars", latest)
    monkeypatch.setattr("cufolio_cpu.five_minute_paper_session.download_minute_bars", unexpected_rest)
    update = _merge_current_minutes(
        pd.DataFrame({"timestamp": ["2026-07-28T14:29:00Z"], "symbol": ["AAA"], "close": [100.0]}),
        ["AAA", "FRESH", "STALE"],
        session_day=date(2026, 7, 28),
        observed_at=pd.Timestamp("2026-07-28T14:35:00Z"),
        cache_path=None,
        minute_stream=None,
        health=MinuteCacheHealth(),
        repair_from_rest=False,
        persist=False,
        endpoint_only=True,
        allow_rest_repair=False,
        allow_yfinance_fallback=False,
        latest_bar_polling=True,
    )

    assert update.new_rows.to_dict("records") == [
        {"timestamp": pd.Timestamp("2026-07-28T14:34:00Z"), "symbol": "FRESH", "close": 101.0}
    ]
    assert update.bars["symbol"].tolist() == ["AAA", "FRESH"]
