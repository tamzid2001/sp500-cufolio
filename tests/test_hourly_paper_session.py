from __future__ import annotations

from datetime import date

from cufolio_cpu.hourly_paper_session import session_events


def test_hourly_paper_session_has_hourly_selections_and_fifteen_minute_rebalances() -> None:
    events = session_events(date(2026, 7, 28))
    selections = [event for event in events if event.kind == "select"]
    rebalances = [event for event in events if event.kind == "rebalance"]

    assert [event.decision_at.tz_convert("America/New_York").strftime("%H:%M") for event in selections] == [
        "10:30", "11:30", "12:30", "13:30", "14:30"
    ]
    assert len(rebalances) == 15
    assert events[-1].kind == "flatten"
    assert events[-1].due_at.tz_convert("America/New_York").strftime("%H:%M") == "15:30"
    assert all(event.decision_at.tz_convert("America/New_York").strftime("%H:%M") != "15:30" for event in selections)
