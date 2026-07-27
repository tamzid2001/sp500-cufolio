from __future__ import annotations

import pandas as pd

from cufolio_cpu.intraday_selection import _last_completed_15_minute_boundary


def test_history_endpoint_excludes_in_progress_fifteen_minute_bar() -> None:
    boundary = _last_completed_15_minute_boundary("2026-07-27T14:47:31-04:00")

    assert pd.Timestamp(boundary) == pd.Timestamp("2026-07-27T18:45:00Z")
