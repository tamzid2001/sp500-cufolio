from __future__ import annotations

from itertools import chain

import pandas as pd

from cufolio_cpu import five_minute_paper_daemon


def test_daemon_handles_after_hours_idle_heartbeat_without_starting_a_session(monkeypatch, capsys, tmp_path) -> None:
    monotonic_values = chain([0.0, 0.0, 0.0, 0.0, 1.0])
    monkeypatch.setattr(five_minute_paper_daemon.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(five_minute_paper_daemon.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        five_minute_paper_daemon,
        "utc_now",
        lambda: pd.Timestamp("2026-07-30T00:19:00Z"),
    )
    monkeypatch.setattr(
        five_minute_paper_daemon,
        "run_five_minute_paper_session",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("after-hours daemon must stay idle")),
    )

    five_minute_paper_daemon.run_five_minute_paper_daemon(
        run_seconds=1,
        state_path=tmp_path / "state.json",
        output_dir=tmp_path / "artifacts",
        universe_cache_path=tmp_path / "universe.csv",
        top_n=20,
        max_weight="0.10",
    )

    output = capsys.readouterr().out
    assert "FIVE MINUTE PAPER DAEMON IDLE" in output
    assert "FIVE MINUTE PAPER DAEMON HANDOFF READY" in output
