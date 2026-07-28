from __future__ import annotations

import pandas as pd

from cufolio_cpu.alpaca import load_symbols


def test_alpaca_symbol_loader_prefers_source_symbol_for_share_classes(tmp_path) -> None:
    universe = tmp_path / "universe.csv"
    pd.DataFrame(
        {
            "symbol": ["BRK-B", "AAPL"],
            "source_symbol": ["BRK.B", "AAPL"],
        }
    ).to_csv(universe, index=False)

    assert load_symbols(universe) == ["BRK.B", "AAPL"]
