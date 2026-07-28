from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pandas as pd

from cufolio_cpu.alpaca import AlpacaMinuteBarStream, download_minute_bars, load_symbols


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
