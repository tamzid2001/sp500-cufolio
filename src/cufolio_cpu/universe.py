"""Universe acquisition utilities for reproducible research runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def current_sp500_universe() -> pd.DataFrame:
    """Fetch the current Wikipedia constituent table for short-horizon research.

    This is a convenience source, not a point-in-time constituent history. Use a
    licensed historical-membership source for survivorship-bias-free backtests.
    """
    tables = pd.read_html(WIKIPEDIA_SP500_URL)
    table = next((candidate for candidate in tables if "Symbol" in candidate.columns), None)
    if table is None:
        raise RuntimeError("could not find the current S&P 500 constituent table")
    result = table.rename(columns={"Symbol": "source_symbol", "Security": "security"}).copy()
    result["symbol"] = result["source_symbol"].astype(str).str.replace(".", "-", regex=False)
    columns = [column for column in ["symbol", "source_symbol", "security", "GICS Sector"] if column in result]
    return result.loc[:, columns].drop_duplicates("symbol").sort_values("symbol")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the current Wikipedia S&P 500 constituent table.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    universe = current_sp500_universe()
    universe.to_csv(output, index=False)
    print(f"Wrote {len(universe)} current S&P 500 symbols to {output}")


if __name__ == "__main__":
    main()
