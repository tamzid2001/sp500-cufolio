from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pandas as pd

from cufolio_cpu.alpaca import (
    AlpacaMinuteBarStream,
    download_latest_iex_minute_bars,
    download_minute_bars,
    download_minute_endpoint_bars,
    load_symbols,
)
from cufolio_cpu.universe import cached_alpaca_tradable_fractionable_universe


def test_alpaca_symbol_loader_prefers_source_symbol_for_share_classes(tmp_path) -> None:
    universe = tmp_path / "universe.csv"
    pd.DataFrame(
        {
            "symbol": ["BRK-B", "AAPL"],
            "source_symbol": ["BRK.B", "AAPL"],
        }
    ).to_csv(universe, index=False)

    assert load_symbols(universe) == ["BRK.B", "AAPL"]


def test_historical_minute_downloader_explicitly_uses_iex(monkeypatch) -> None:
    """A paper account must never silently make a recent SIP request."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient

    requests = []

    class FakeClient:
        def __init__(self, *_args) -> None:
            pass

        def get_stock_bars(self, request):
            requests.append(request)
            return SimpleNamespace(
                df=pd.DataFrame(
                    {"close": [101.25]},
                    index=pd.MultiIndex.from_tuples(
                        [("AAA", pd.Timestamp("2026-07-28T14:20:00Z"))], names=["symbol", "timestamp"]
                    ),
                )
            )

    monkeypatch.setenv("ALPACA_API_KEY", "paper-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "paper-secret")
    monkeypatch.setattr(StockHistoricalDataClient, "__new__", staticmethod(lambda *_args, **_kwargs: FakeClient()))

    bars = download_minute_bars(["AAA"], "2026-07-28T14:20:00Z", "2026-07-28T14:21:00Z")

    assert requests[0].feed == DataFeed.IEX
    assert bars.loc[0, "close"] == 101.25


def test_threaded_latest_minute_downloader_batches_and_uses_iex(monkeypatch) -> None:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient

    requests = []

    class FakeClient:
        def __init__(self, *_args) -> None:
            pass

        def get_stock_latest_bar(self, request):
            requests.append(request)
            return {
                symbol: SimpleNamespace(timestamp=pd.Timestamp("2026-07-28T14:20:00Z"), close=100.0 + index)
                for index, symbol in enumerate(request.symbol_or_symbols)
            }

    monkeypatch.setenv("ALPACA_API_KEY", "paper-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "paper-secret")
    monkeypatch.setattr(StockHistoricalDataClient, "__new__", staticmethod(lambda *_args, **_kwargs: FakeClient()))

    bars = download_latest_iex_minute_bars(["AAA", "BBB", "CCC"], batch_size=2, max_workers=2)

    assert len(requests) == 2
    assert all(request.feed == DataFeed.IEX for request in requests)
    assert bars["symbol"].tolist() == ["AAA", "BBB", "CCC"]


def test_iex_websocket_buffer_returns_only_completed_bar_minutes() -> None:
    stream = AlpacaMinuteBarStream(["AAA"])
    asyncio.run(
        stream._on_bar(
            {"symbol": "AAA", "timestamp": "2026-07-28T14:20:00Z", "close": 101.25}
        )
    )
    asyncio.run(
        stream._on_bar(
            {"symbol": "AAA", "timestamp": "2026-07-28T14:21:00Z", "close": 101.50}
        )
    )

    completed = stream.completed_bars_through(pd.Timestamp("2026-07-28T14:20:00Z"))

    assert completed.to_dict("records") == [
        {"timestamp": pd.Timestamp("2026-07-28T14:20:00Z"), "symbol": "AAA", "close": 101.25}
    ]


def test_iex_websocket_drain_releases_each_completed_bar_once() -> None:
    stream = AlpacaMinuteBarStream(["AAA"])
    asyncio.run(stream._on_bar({"symbol": "AAA", "timestamp": "2026-07-28T14:20:00Z", "close": 101.25}))
    asyncio.run(stream._on_bar({"symbol": "AAA", "timestamp": "2026-07-28T14:20:00Z", "close": 101.50}))
    asyncio.run(stream._on_bar({"symbol": "AAA", "timestamp": "2026-07-28T14:21:00Z", "close": 101.75}))

    drained = stream.drain_completed_bars_through(pd.Timestamp("2026-07-28T14:20:00Z"))

    assert drained.to_dict("records") == [
        {"timestamp": pd.Timestamp("2026-07-28T14:20:00Z"), "symbol": "AAA", "close": 101.50}
    ]
    assert stream.drain_completed_bars_through(pd.Timestamp("2026-07-28T14:20:00Z")).empty
    assert stream.completed_bars_through(pd.Timestamp("2026-07-28T14:21:00Z")).to_dict("records") == [
        {"timestamp": pd.Timestamp("2026-07-28T14:21:00Z"), "symbol": "AAA", "close": 101.75}
    ]


def test_cached_full_alpaca_universe_requires_tradable_fractionable_symbols(tmp_path) -> None:
    universe = tmp_path / "alpaca_tradable_fractionable_universe.csv"
    pd.DataFrame(
        {
            "symbol": ["aapl", "NOTRADABLE", "NOTFRACTIONAL", None, "aapl"],
            "tradable": [True, False, True, True, True],
            "fractionable": [True, True, False, True, True],
        }
    ).to_csv(universe, index=False)

    loaded = cached_alpaca_tradable_fractionable_universe(universe)

    assert loaded["symbol"].tolist() == ["AAPL"]
    assert loaded["universe_source"].tolist() == ["cached_alpaca_tradable_fractionable_snapshot"]


def test_endpoint_downloader_keeps_only_requested_new_york_minutes_and_uses_iex(monkeypatch) -> None:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient

    requests = []

    class FakeClient:
        def __init__(self, *_args) -> None:
            pass

        def get_stock_bars(self, request):
            requests.append(request)
            return SimpleNamespace(
                df=pd.DataFrame(
                    {"close": [100.0, 101.0, 102.0]},
                    index=pd.MultiIndex.from_tuples(
                        [
                            ("AAA", pd.Timestamp("2026-07-28T13:20:00Z")),  # 09:20 ET
                            ("AAA", pd.Timestamp("2026-07-28T13:21:00Z")),
                            ("AAA", pd.Timestamp("2026-07-28T14:30:00Z")),  # 10:30 ET
                        ],
                        names=["symbol", "timestamp"],
                    ),
                )
            )

    monkeypatch.setenv("ALPACA_API_KEY", "paper-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "paper-secret")
    monkeypatch.setattr(StockHistoricalDataClient, "__new__", staticmethod(lambda *_args, **_kwargs: FakeClient()))

    bars = download_minute_endpoint_bars(
        ["AAA"],
        "2026-07-28T13:00:00Z",
        "2026-07-28T15:00:00Z",
        endpoint_times={pd.Timestamp("09:20").time(), pd.Timestamp("10:30").time()},
    )

    assert requests[0].feed == DataFeed.IEX
    assert bars["timestamp"].tolist() == [
        pd.Timestamp("2026-07-28T13:20:00Z"),
        pd.Timestamp("2026-07-28T14:30:00Z"),
    ]
