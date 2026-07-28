from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

import cufolio_cpu.hourly_paper_session as hourly_paper_session
from cufolio_cpu.hourly_paper_session import (
    _default_checkpoint,
    _load_checkpoint,
    _persist_checkpoint,
    _restore_targets,
    run_hourly_paper_session,
    session_events,
)
from cufolio_cpu.hourly_paper_daemon import rolling_history_start, run_hourly_paper_daemon


def test_hourly_paper_session_refreshes_final_target_at_1420_and_rebalances_every_fifteen_minutes() -> None:
    events = session_events(date(2026, 7, 28))
    selections = [event for event in events if event.kind == "select"]
    rebalances = [event for event in events if event.kind == "rebalance"]

    assert [event.selection_at.tz_convert("America/New_York").strftime("%H:%M") for event in selections] == [
        "09:20", "10:20", "11:20", "12:20", "14:20"
    ]
    assert [event.target_start.tz_convert("America/New_York").strftime("%H:%M") for event in selections] == [
        "10:30", "11:30", "12:30", "13:30", "14:30"
    ]
    assert len(rebalances) == 20
    assert [event.due_at.tz_convert("America/New_York").strftime("%H:%M") for event in rebalances[:4]] == [
        "10:30", "10:45", "11:00", "11:15"
    ]
    final_target_rebalances = [
        event for event in rebalances
        if event.target_start.tz_convert("America/New_York").strftime("%H:%M") == "14:30"
    ]
    assert [event.selection_at.tz_convert("America/New_York").strftime("%H:%M") for event in final_target_rebalances] == [
        "14:20", "14:20", "14:20", "14:20"
    ]
    assert [event.due_at.tz_convert("America/New_York").strftime("%H:%M") for event in final_target_rebalances] == [
        "14:30", "14:45", "15:00", "15:15"
    ]
    assert events[-1].kind == "flatten"
    assert events[-1].due_at.tz_convert("America/New_York").strftime("%H:%M") == "15:30"
    assert all(event.target_start.tz_convert("America/New_York").strftime("%H:%M") != "15:30" for event in selections)


def test_hourly_session_rejects_live_mode_without_explicit_acknowledgement(tmp_path) -> None:
    with pytest.raises(ValueError, match="allow_live_trading"):
        run_hourly_paper_session(
            session_day=date(2026, 7, 28),
            history_start="2026-06-01T13:30:00Z",
            output_dir=tmp_path,
            mode="live",
        )


def test_hourly_handoff_checkpoint_restores_selected_weights_exactly(tmp_path) -> None:
    session_day = date(2026, 7, 28)
    state_path = tmp_path / "hourly-state.json"
    checkpoint = _default_checkpoint(session_day)
    target_start = session_events(session_day)[0].target_start
    assert target_start is not None
    _persist_checkpoint(
        state_path,
        checkpoint,
        completed_events={"select:2026-07-28T13:20:00+00:00"},
        targets_by_start={target_start: {"AAA": Decimal("0.6"), "BBB": Decimal("0.4")}},
        forecast_details={target_start: {"target_start": target_start.isoformat()}},
        ledger=[{"event": "forecast_selected", "target_start": target_start.isoformat()}],
    )

    restored = _load_checkpoint(state_path, session_day)
    restored_targets = _restore_targets(restored)
    assert restored["completed_events"] == ["select:2026-07-28T13:20:00+00:00"]
    assert restored_targets[target_start] == {"AAA": Decimal("0.6"), "BBB": Decimal("0.4")}
    assert restored["ledger"][-1]["event"] == "forecast_selected"


def test_hourly_daemon_rejects_non_positive_handoff_duration(tmp_path) -> None:
    with pytest.raises(ValueError, match="run_seconds"):
        run_hourly_paper_daemon(
            run_seconds=0,
            state_path=tmp_path / "state.json",
            history_start="2026-06-01T13:30:00Z",
            output_dir=tmp_path / "output",
            top_n=20,
            max_weight="0.10",
        )


def test_minute_endpoint_cache_appends_fresh_rows_and_skips_overnight_gap(tmp_path, monkeypatch) -> None:
    """A handoff keeps only exact model endpoints and asks Alpaca for new data."""
    cache_path = tmp_path / "minute-endpoints.csv.gz"
    first_decision = pd.Timestamp("2026-07-28T14:20:00Z")  # 10:20 New York
    second_decision = pd.Timestamp("2026-07-29T13:20:00Z")  # next-day 09:20
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fake_download(symbols, start, end):
        calls.append((pd.Timestamp(start), pd.Timestamp(end)))
        if len(calls) == 1:
            return pd.DataFrame(
                {
                    "timestamp": [
                        "2026-07-28T13:20:00Z",  # overlap: authoritative replacement
                        "2026-07-28T13:21:00Z",  # not a model endpoint
                        "2026-07-28T13:30:00Z",
                        "2026-07-28T14:20:00Z",
                    ],
                    "symbol": ["AAA", "AAA", "AAA", "AAA"],
                    "close": [101, 999, 102, 103],
                }
            )
        return pd.DataFrame(
            {
                "timestamp": ["2026-07-29T13:20:00Z"],
                "symbol": ["AAA"],
                "close": [104],
            }
        )

    monkeypatch.setattr(hourly_paper_session, "download_minute_bars", fake_download)
    initial = pd.DataFrame(
        {
            "timestamp": ["2026-07-28T13:20:00Z"],
            "symbol": ["AAA"],
            "close": [100],
        }
    )
    first = hourly_paper_session._history_through(
        initial,
        ["AAA"],
        start="2026-06-01T13:20:00Z",
        decision_at=first_decision,
        cache_path=cache_path,
    )
    second = hourly_paper_session._history_through(
        first,
        ["AAA"],
        start="2026-06-01T13:20:00Z",
        decision_at=second_decision,
        cache_path=cache_path,
    )

    assert calls[0] == (pd.Timestamp("2026-07-28T13:20:00Z"), first_decision + pd.Timedelta(minutes=1))
    # A new session fetches its exact 09:20 minute rather than every overnight
    # minute since the prior 10:20 endpoint.
    assert calls[1] == (second_decision, second_decision + pd.Timedelta(minutes=1))
    assert first.loc[first["timestamp"] == pd.Timestamp("2026-07-28T13:20:00Z"), "close"].item() == 101
    # The newest session keeps each completed minute; older sessions compact
    # back to model endpoints at the next day's handoff.
    assert pd.Timestamp("2026-07-28T13:21:00Z") in set(first["timestamp"])
    assert list(second["timestamp"]) == [
        pd.Timestamp("2026-07-28T13:20:00Z"),
        pd.Timestamp("2026-07-28T13:30:00Z"),
        pd.Timestamp("2026-07-28T14:20:00Z"),
        pd.Timestamp("2026-07-29T13:20:00Z"),
    ]
    pd.testing.assert_frame_equal(hourly_paper_session._read_minute_cache(cache_path), second)


def test_current_session_minute_refresh_merges_only_completed_minutes(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / "minute-endpoints.csv.gz"
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fake_download(symbols, start, end):
        calls.append((pd.Timestamp(start), pd.Timestamp(end)))
        return pd.DataFrame(
            {
                "timestamp": ["2026-07-28T14:20:00Z", "2026-07-28T14:21:00Z", "2026-07-28T14:22:00Z"],
                "symbol": ["AAA", "AAA", "AAA"],
                "close": [101, 102, 999],
            }
        )

    monkeypatch.setattr(hourly_paper_session, "download_minute_bars", fake_download)
    existing = pd.DataFrame(
        {
            "timestamp": ["2026-07-28T14:19:00Z"],
            "symbol": ["AAA"],
            "close": [100],
        }
    )
    refreshed = hourly_paper_session._refresh_current_session_minutes(
        existing,
        ["AAA"],
        session_day=date(2026, 7, 28),
        observed_at=pd.Timestamp("2026-07-28T14:22:15Z"),
        cache_path=cache_path,
    )

    # 10:21 New York is the last fully closed minute at 10:22:15. Even if
    # the provider returns a 10:22 boundary row, it is discarded as partial.
    assert calls == [(pd.Timestamp("2026-07-28T14:20:00Z"), pd.Timestamp("2026-07-28T14:22:00Z"))]
    assert list(refreshed["timestamp"]) == [
        pd.Timestamp("2026-07-28T14:19:00Z"),
        pd.Timestamp("2026-07-28T14:20:00Z"),
        pd.Timestamp("2026-07-28T14:21:00Z"),
    ]
    pd.testing.assert_frame_equal(hourly_paper_session._read_minute_cache(cache_path), refreshed)


def test_current_session_uses_yahoo_only_when_iex_websocket_is_disconnected(tmp_path, monkeypatch) -> None:
    class DisconnectedStream:
        error = None
        connected = False

        def completed_bars_through(self, _completed):
            return pd.DataFrame(columns=["timestamp", "symbol", "close"])

    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fake_yahoo(symbols, start, end):
        calls.append((pd.Timestamp(start), pd.Timestamp(end)))
        return pd.DataFrame(
            {"timestamp": ["2026-07-28T14:20:00Z"], "symbol": ["AAA"], "close": [101]}
        )

    def fail_iex(*_args, **_kwargs):
        raise AssertionError("a live refresh must not poll the Alpaca historical endpoint while streaming")

    monkeypatch.setattr(hourly_paper_session, "download_minute_bars", fail_iex)
    monkeypatch.setattr(hourly_paper_session, "download_yfinance_minute_bars", fake_yahoo)
    refreshed = hourly_paper_session._refresh_current_session_minutes(
        pd.DataFrame(columns=["timestamp", "symbol", "close"]),
        ["AAA"],
        session_day=date(2026, 7, 28),
        observed_at=pd.Timestamp("2026-07-28T14:21:15Z"),
        cache_path=tmp_path / "minute-endpoints.csv.gz",
        minute_stream=DisconnectedStream(),
    )

    assert calls == [(pd.Timestamp("2026-07-28T13:20:00Z"), pd.Timestamp("2026-07-28T14:21:00Z"))]
    assert refreshed.loc[0, "symbol"] == "AAA"
    assert refreshed.loc[0, "close"] == 101


def test_last_completed_session_minute_never_uses_partial_or_pre_session_bar() -> None:
    assert hourly_paper_session._last_completed_session_minute(
        pd.Timestamp("2026-07-28T13:20:59Z"), date(2026, 7, 28)
    ) is None
    assert hourly_paper_session._last_completed_session_minute(
        pd.Timestamp("2026-07-28T13:21:00Z"), date(2026, 7, 28)
    ) == pd.Timestamp("2026-07-28T13:20:00Z")


def test_complete_history_session_requires_every_hourly_label_endpoint() -> None:
    day = pd.Timestamp("2026-07-27", tz="America/New_York")
    endpoints = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=7, freq="h")
    bars = pd.DataFrame(
        {
            "timestamp": endpoints.tz_convert("UTC"),
            "symbol": ["AAA"] * len(endpoints),
            "close": range(100, 107),
        }
    )

    assert hourly_paper_session._complete_history_sessions(bars, date(2026, 7, 28)) == 1
    assert hourly_paper_session._complete_history_sessions(bars.iloc[:-1], date(2026, 7, 28)) == 0


def test_rolling_history_start_uses_compact_window_and_rejects_too_little_history() -> None:
    reference = pd.Timestamp("2026-07-28T13:20:00Z")
    assert rolling_history_start(reference, calendar_days=45) == "2026-06-13T13:20:00+00:00"
    with pytest.raises(ValueError, match="at least 35"):
        rolling_history_start(reference, calendar_days=34)
