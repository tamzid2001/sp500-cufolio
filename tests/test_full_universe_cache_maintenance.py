from __future__ import annotations

import importlib.util
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latest_completed_market_session_uses_official_close_and_calendar(monkeypatch) -> None:
    cache = _load_tool("full_universe_cache_tool", "tools/cache_full_alpaca_five_minute_input.py")
    requested = []

    class FakeTradingClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def get_calendar(self, filters):
            requested.append(filters)
            return [
                SimpleNamespace(
                    date=date(2026, 7, 28),
                    close=datetime(2026, 7, 28, 16, 0, tzinfo=ZoneInfo("America/New_York")),
                ),
                SimpleNamespace(
                    date=date(2026, 7, 29),
                    close=datetime(2026, 7, 29, 16, 0, tzinfo=ZoneInfo("America/New_York")),
                ),
            ]

    monkeypatch.setattr(cache, "TradingClient", FakeTradingClient)

    before_close = cache.latest_completed_market_session(
        "key", "secret", now=cache.pd.Timestamp("2026-07-29T19:59:00Z")
    )
    after_close = cache.latest_completed_market_session(
        "key", "secret", now=cache.pd.Timestamp("2026-07-29T20:01:00Z")
    )

    assert before_close == date(2026, 7, 28)
    assert after_close == date(2026, 7, 29)
    assert requested[0].start == date(2026, 7, 15)
    assert requested[0].end == date(2026, 7, 29)


def test_stale_cache_reset_is_limited_to_known_cache_files(tmp_path) -> None:
    cache = _load_tool("full_universe_cache_reset_tool", "tools/cache_full_alpaca_five_minute_input.py")
    known = [cache.UNIVERSE_FILENAME, cache.BARS_FILENAME, cache.PARTS_FILENAME, cache.METADATA_FILENAME]
    for filename in known:
        (tmp_path / filename).write_text("cache", encoding="utf-8")
    sentinel = tmp_path / "must-not-delete.txt"
    sentinel.write_text("keep", encoding="utf-8")

    cache._discard_stale_cache(tmp_path)

    assert all(not (tmp_path / filename).exists() for filename in known)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_verify_cache_rejects_an_out_of_date_or_tampered_input(tmp_path) -> None:
    cache = _load_tool("full_universe_cache_verify_tool", "tools/cache_full_alpaca_five_minute_input.py")
    universe = tmp_path / cache.UNIVERSE_FILENAME
    endpoints = tmp_path / cache.BARS_FILENAME
    universe.write_text("symbol\nAAA\n", encoding="utf-8")
    endpoints.write_text("timestamp,symbol,close\n", encoding="utf-8")
    metadata = {
        "complete": True,
        "evaluation_end": "2026-07-29",
        "universe_symbols": 1,
        "retained_rows": 0,
        "universe_sha256": cache._sha256(universe),
        "endpoint_data_sha256": cache._sha256(endpoints),
    }
    (tmp_path / cache.METADATA_FILENAME).write_text(cache.json.dumps(metadata), encoding="utf-8")

    verified = cache.verify_cache(tmp_path, date(2026, 7, 29))
    assert verified["universe_symbols"] == 1

    endpoints.write_text("tampered", encoding="utf-8")
    try:
        cache.verify_cache(tmp_path, date(2026, 7, 29))
    except RuntimeError as error:
        assert "checksum mismatch" in str(error)
    else:
        raise AssertionError("tampered endpoint data was accepted")


def test_rolling_endpoint_cache_advances_verified_history_without_redownload(tmp_path) -> None:
    cache = _load_tool("full_universe_cache_roll_forward_tool", "tools/cache_full_alpaca_five_minute_input.py")
    universe = tmp_path / cache.UNIVERSE_FILENAME
    endpoints = tmp_path / cache.BARS_FILENAME
    rolling = tmp_path / "live-endpoints.csv.gz"
    pd.DataFrame({"symbol": ["AAA"], "tradable": [True], "fractionable": [True]}).to_csv(universe, index=False)
    pd.DataFrame(
        {
            "timestamp": ["2026-04-01T13:29:00Z", "2026-07-28T19:54:00Z"],
            "symbol": ["AAA", "AAA"],
            "close": [100.0, 101.0],
        }
    ).to_csv(endpoints, index=False, compression="gzip")
    pd.DataFrame(
        {
            "timestamp": ["2026-07-29T19:54:00Z"],
            "symbol": ["AAA"],
            "close": [102.0],
        }
    ).to_csv(rolling, index=False, compression="gzip")
    metadata = {
        "complete": True,
        "evaluation_end": "2026-07-28",
        "universe_symbols": 1,
        "retained_rows": 2,
        "universe_sha256": cache._sha256(universe),
        "endpoint_data_sha256": cache._sha256(endpoints),
    }
    (tmp_path / cache.METADATA_FILENAME).write_text(cache.json.dumps(metadata), encoding="utf-8")

    rolled = cache.append_rolling_endpoint_cache(tmp_path, rolling, evaluation_end=date(2026, 7, 29))

    assert rolled["evaluation_end"] == "2026-07-29"
    assert rolled["rolled_forward_from_evaluation_end"] == "2026-07-28"
    assert rolled["rolling_endpoint_rows_merged"] == 1
    persisted = pd.read_csv(endpoints, compression="gzip")
    assert persisted["timestamp"].tolist() == [
        "2026-04-01 13:29:00+00:00",
        "2026-07-28 19:54:00+00:00",
        "2026-07-29 19:54:00+00:00",
    ]


def test_checkpoint_merge_is_monotonic_and_deduplicates_events() -> None:
    publisher = _load_tool("five_minute_checkpoint_publisher", "tools/publish_five_minute_checkpoint.py")
    remote = {
        "format_version": 1,
        "session_date": "2026-07-29",
        "completed_events": ["forecast:09:30"],
        "ledger": [{"event": "one"}],
    }
    local = {
        "format_version": 2,
        "session_date": "2026-07-29",
        "completed_events": ["forecast:09:30", "forecast:09:35"],
        "ledger": [{"event": "one"}, {"event": "two"}],
    }

    merged = publisher.merge_checkpoints(remote, local)

    assert merged["format_version"] == 2
    assert merged["completed_events"] == ["forecast:09:30", "forecast:09:35"]
    assert merged["ledger"] == [{"event": "one"}, {"event": "two"}]
