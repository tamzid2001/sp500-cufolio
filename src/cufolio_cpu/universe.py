"""Universe acquisition utilities for reproducible research runs."""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DATASETS_SP500_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)
USER_AGENT = "sp500-cufolio-research/0.1 (GitHub Actions; non-commercial research)"


def current_sp500_universe() -> pd.DataFrame:
    """Fetch the current Wikipedia constituent table for short-horizon research.

    This is a convenience source, not a point-in-time constituent history. Use a
    licensed historical-membership source for survivorship-bias-free backtests.
    """
    # pandas' direct URL handling uses a generic urllib request, which is
    # rejected by Wikipedia from some hosted CI runners. Fetching with a clear
    # research user agent keeps the preferred source usable when available.
    try:
        import requests

        response = requests.get(WIKIPEDIA_SP500_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        table = next((candidate for candidate in tables if "Symbol" in candidate.columns), None)
        if table is None:
            raise RuntimeError("could not find the current S&P 500 constituent table")
        source = "wikipedia"
    except Exception as wikipedia_error:
        # The maintained public CSV is sourced from the same constituent table.
        # It is a fallback for availability, not point-in-time index history.
        try:
            table = pd.read_csv(DATASETS_SP500_URL)
            if "Symbol" not in table.columns:
                raise RuntimeError("fallback S&P 500 CSV has no Symbol column")
            source = "datasets/s-and-p-500-companies"
        except Exception as fallback_error:
            raise RuntimeError(
                "could not retrieve the current S&P 500 universe from either "
                f"Wikipedia ({wikipedia_error}) or the documented CSV fallback ({fallback_error})"
            ) from fallback_error
    result = table.rename(columns={"Symbol": "source_symbol", "Security": "security"}).copy()
    result["symbol"] = result["source_symbol"].astype(str).str.replace(".", "-", regex=False)
    result["universe_source"] = source
    columns = [
        column
        for column in ["symbol", "source_symbol", "security", "GICS Sector", "universe_source"]
        if column in result
    ]
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
