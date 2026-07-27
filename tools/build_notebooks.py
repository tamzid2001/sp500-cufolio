"""Generate deterministic CPU-only counterparts of the copied cuFOLIO notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
BOOTSTRAP = dedent(
    """
    from pathlib import Path
    import sys

    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "src").exists():
            sys.path.insert(0, str(candidate / "src"))
            break

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from cufolio_cpu.returns import daily_returns_from_minute_bars
    from cufolio_cpu.synthetic import synthetic_minute_bars
    """
).strip()
PREPARE = dedent(
    """
    bars = synthetic_minute_bars(sessions=45, seed=42)
    daily_log, daily_simple = daily_returns_from_minute_bars(bars)
    print(f"{len(bars):,} minute bars -> {daily_simple.shape[0]} daily sessions x {daily_simple.shape[1]} assets")
    daily_simple.tail()
    """
).strip()


def cell(cell_type: str, source: str) -> dict[str, object]:
    base: dict[str, object] = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def make_notebook(title: str, introduction: str, code_cells: list[str]) -> dict[str, object]:
    return {
        "cells": [
            cell(
                "markdown",
                f"# {title}\n\n"
                "**CPU-only counterpart of an NVIDIA cuFOLIO notebook.** The matching "
                "unmodified GPU notebook is in `../upstream_notebooks/`. This version "
                "uses deterministic synthetic one-minute bars so it runs on GitHub Actions "
                "without NVIDIA infrastructure.\n\n"
                + introduction,
            ),
            cell("code", BOOTSTRAP),
            *[cell("code", source) for source in code_cells],
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build() -> dict[str, dict[str, object]]:
    cvar_result = dedent(
        """
        from cufolio_cpu.optimize import mean_cvar_weights

        result = mean_cvar_weights(
            daily_simple, risk_aversion=5.0, confidence=0.95, max_weight=0.30
        )
        print("Solver status:", result.status)
        print("Expected daily return:", f"{result.expected_return:.5%}")
        print("Historical 95% CVaR loss:", f"{result.cvar:.5%}")
        result.weights.sort_values(ascending=False)
        """
    ).strip()
    cvar_chart = dedent(
        """
        portfolio = daily_simple @ result.weights
        plt.figure(figsize=(8, 3))
        plt.plot((1 + portfolio).cumprod(), label="Mean–CVaR")
        plt.title("Illustrative CPU Mean–CVaR cumulative return")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.show()
        """
    ).strip()
    mean_variance_result = dedent(
        """
        from cufolio_cpu.optimize import mean_variance_weights

        result = mean_variance_weights(daily_simple, risk_aversion=15.0, max_weight=0.30)
        print("Solver status:", result.status)
        print("Expected daily return:", f"{result.expected_return:.5%}")
        result.weights.sort_values(ascending=False)
        """
    ).strip()
    frontier_result = dedent(
        """
        from cufolio_cpu.optimize import efficient_frontier

        frontier = efficient_frontier(daily_simple, [0.5, 1.0, 2.0, 5.0, 10.0], max_weight=0.30)
        frontier[["risk_aversion", "expected_return", "cvar", "status"]]
        """
    ).strip()
    frontier_chart = dedent(
        """
        plt.figure(figsize=(6, 4))
        plt.plot(frontier["cvar"], frontier["expected_return"], marker="o")
        for row in frontier.itertuples():
            plt.annotate(str(row.risk_aversion), (row.cvar, row.expected_return))
        plt.xlabel("Historical CVaR loss")
        plt.ylabel("Expected daily return")
        plt.title("CPU Mean–CVaR frontier")
        plt.show()
        """
    ).strip()
    rebalancing_result = dedent(
        """
        from cufolio_cpu.backtest import walk_forward_rebalance

        performance, weights = walk_forward_rebalance(
            daily_simple,
            lookback_days=20,
            rebalance_every_days=5,
            risk_aversion=5.0,
            max_weight=0.30,
            transaction_cost_bps=5.0,
        )
        performance.tail()
        """
    ).strip()
    rebalancing_chart = dedent(
        """
        plt.figure(figsize=(8, 3))
        plt.plot(performance["equity"], label="walk-forward Mean–CVaR")
        plt.title("Causal rebalancing backtest")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.show()
        print("Final equity:", f"{performance['equity'].iloc[-1]:.4f}")
        """
    ).strip()
    launchable_result = dedent(
        """
        from cufolio_cpu.optimize import mean_cvar_weights
        from cufolio_cpu.backtest import walk_forward_rebalance

        allocation = mean_cvar_weights(daily_simple, risk_aversion=5.0, max_weight=0.30)
        performance, weights = walk_forward_rebalance(
            daily_simple, lookback_days=20, rebalance_every_days=5
        )
        print("Allocation status:", allocation.status)
        print("Latest allocation:")
        print(allocation.weights.sort_values(ascending=False).to_frame("weight"))
        print("Final walk-forward equity:", f"{performance['equity'].iloc[-1]:.4f}")
        """
    ).strip()
    return {
        "cvar_basic_cpu.ipynb": make_notebook(
            "Mean–CVaR portfolio optimization from one-minute S&P-style returns",
            "Minute log returns are summed within each regular session. The optimizer receives daily simple returns and solves a long-only, fully invested Mean–CVaR problem on CPU.",
            [PREPARE, cvar_result, cvar_chart],
        ),
        "mean_variance_basic_cpu.ipynb": make_notebook(
            "Mean–variance portfolio optimization",
            "This counterpart provides the Markowitz baseline using the same daily returns derived from intraday data.",
            [PREPARE, mean_variance_result],
        ),
        "efficient_frontier_cpu.ipynb": make_notebook(
            "Mean–CVaR efficient frontier",
            "The frontier solves several CPU Mean–CVaR problems at increasing risk-aversion levels. It is intentionally small for normal GitHub-hosted runners.",
            [PREPARE, frontier_result, frontier_chart],
        ),
        "rebalancing_strategies_cpu.ipynb": make_notebook(
            "Causal rebalancing strategy backtest",
            "Each rebalance uses only the prior 20 complete daily returns; the resulting allocation is applied beginning with the next trade date. Transaction costs are charged on turnover.",
            [PREPARE, rebalancing_result, rebalancing_chart],
        ),
        "launchable_cpu.ipynb": make_notebook(
            "CPU launchable portfolio-optimization workflow",
            "A compact end-to-end execution path: minute bars → daily returns → Mean–CVaR allocation → causal daily backtest. It deliberately replaces NVIDIA's GPU launchable environment with a normal GitHub-hosted CPU runner.",
            [PREPARE, launchable_result],
        ),
    }


def main() -> None:
    NOTEBOOKS.mkdir(exist_ok=True)
    for filename, notebook in build().items():
        (NOTEBOOKS / filename).write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        print(f"generated notebooks/{filename}")


if __name__ == "__main__":
    main()
