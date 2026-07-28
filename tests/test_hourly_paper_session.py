from __future__ import annotations

from datetime import date

import pytest

from cufolio_cpu.hourly_paper_session import run_hourly_paper_session, session_events


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
