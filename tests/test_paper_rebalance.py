from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cufolio_cpu.daily_cycle import run_daily_cycle
from cufolio_cpu.daily_selection import daily_simple_returns, select_daily_long_only_portfolio
from cufolio_cpu.paper_rebalance import (
    load_target_weights,
    run_end_of_day_flatten,
    run_end_of_day_transition,
    run_rebalance,
)


class FakePaperClient:
    def __init__(
        self,
        *,
        account: dict[str, object] | None = None,
        market_open: bool = True,
        positions: list[dict[str, object]] | None = None,
        open_orders: list[dict[str, object]] | None = None,
    ) -> None:
        self.account = account or {"equity": "1000", "cash": "1000"}
        self.market_open = market_open
        self.positions = positions or []
        self.open_orders = open_orders or []
        self.submitted: list[dict[str, str]] = []
        self.closed_positions: list[str] = []
        self.asset_requests: list[str] = []
        self.mode = "paper"
        self.base_url = "https://paper-api.alpaca.markets/v2"

    def get_account(self) -> dict[str, object]:
        return self.account

    def get_clock(self) -> dict[str, object]:
        return {"timestamp": "2026-07-27T14:30:00Z", "is_open": self.market_open}

    def get_positions(self) -> list[dict[str, object]]:
        return self.positions

    def get_open_orders(self, symbols: list[str] | None = None) -> list[dict[str, object]]:
        return self.open_orders

    def get_asset(self, symbol: str) -> dict[str, object]:
        self.asset_requests.append(symbol)
        return {"tradable": True, "fractionable": True}

    def submit_order(self, payload: dict[str, str]) -> dict[str, object]:
        self.submitted.append(payload)
        return {"id": f"order-{len(self.submitted)}", "status": "accepted"}

    def close_position(self, symbol: str) -> dict[str, object]:
        self.closed_positions.append(symbol)
        return {"id": f"close-{len(self.closed_positions)}"}


def test_configured_research_targets_are_fully_invested_and_capped() -> None:
    target_file = Path(__file__).parents[1] / "assets" / "paper_target_weights.csv"
    targets = load_target_weights(target_file)
    assert len(targets) == 15
    assert sum(targets.values(), Decimal()) == Decimal("1.000000000")
    assert all(0 < weight <= Decimal("0.10") for weight in targets.values())


def test_market_closed_does_not_query_assets_or_plan_orders() -> None:
    client = FakePaperClient(market_open=False)
    result = run_rebalance(client, {"AAA": Decimal("0.5"), "BBB": Decimal("0.5")})
    assert result["status"] == "market_closed"
    assert result["orders"] == []
    assert client.asset_requests == []


def test_initial_cash_is_split_into_fractional_notional_buy_orders() -> None:
    client = FakePaperClient()
    result = run_rebalance(client, {"AAA": Decimal("0.5"), "BBB": Decimal("0.5")})
    assert result["status"] == "buy_orders_planned"
    assert result["orders"] == [
        {
            "symbol": "AAA",
            "side": "buy",
            "notional": "500.00",
            "target_weight": "0.500000000",
            "current_notional": "0.00",
            "target_notional": "500.00",
        },
        {
            "symbol": "BBB",
            "side": "buy",
            "notional": "500.00",
            "target_weight": "0.500000000",
            "current_notional": "0.00",
            "target_notional": "500.00",
        },
    ]
    assert client.submitted == []


def test_available_cash_scales_buy_orders_without_using_buying_power() -> None:
    client = FakePaperClient(account={"equity": "1000", "cash": "100"})
    result = run_rebalance(client, {"AAA": Decimal("0.5"), "BBB": Decimal("0.5")})
    assert [order["notional"] for order in result["orders"]] == ["50.00", "50.00"]


def test_overweight_target_sells_first_and_defers_buys_to_later_cycle() -> None:
    client = FakePaperClient(
        account={"equity": "1000", "cash": "200"},
        positions=[{"symbol": "AAA", "side": "long", "market_value": "800", "qty": "8"}],
    )
    result = run_rebalance(client, {"AAA": Decimal("0.5"), "BBB": Decimal("0.5")}, execute=True)
    assert result["status"] == "sell_orders_submitted_waiting_for_fill"
    assert result["orders"] == [
        {
            "symbol": "AAA",
            "side": "sell",
            "qty": "3.000000000",
            "target_weight": "0.500000000",
            "current_notional": "800.00",
            "target_notional": "500.00",
        }
    ]
    assert len(client.submitted) == 1
    assert client.submitted[0]["side"] == "sell"
    assert client.submitted[0]["qty"] == "3.000000000"
    assert "notional" not in client.submitted[0]


def test_small_weight_drift_does_not_churn_after_market_order_fills() -> None:
    client = FakePaperClient(
        account={"equity": "1000", "cash": "0"},
        positions=[
            {"symbol": "AAA", "side": "long", "market_value": "502", "qty": "5.02"},
            {"symbol": "BBB", "side": "long", "market_value": "498", "qty": "4.98"},
        ],
    )
    result = run_rebalance(client, {"AAA": Decimal("0.5"), "BBB": Decimal("0.5")})
    assert result["status"] == "within_rebalance_tolerance_or_no_cash"
    assert result["orders"] == []


def test_any_open_target_order_blocks_a_second_cycle() -> None:
    client = FakePaperClient(open_orders=[{"id": "manual-order"}])
    result = run_rebalance(client, {"AAA": Decimal("0.5"), "BBB": Decimal("0.5")}, execute=True)
    assert result["status"] == "waiting_for_open_target_orders"
    assert client.submitted == []


def test_daily_target_change_exits_non_target_positions_before_any_buys() -> None:
    client = FakePaperClient(
        account={"equity": "1000", "cash": "200"},
        positions=[{"symbol": "OLD", "side": "long", "market_value": "200", "qty": "2"}],
    )
    result = run_rebalance(
        client,
        {"AAA": Decimal("0.5"), "BBB": Decimal("0.5")},
        liquidate_non_target_positions=True,
        execute=True,
    )
    assert result["status"] == "sell_orders_submitted_waiting_for_fill"
    assert [order["symbol"] for order in result["orders"]] == ["OLD"]
    assert [order["side"] for order in client.submitted] == ["sell"]


def test_end_of_day_flatten_closes_every_long_position() -> None:
    client = FakePaperClient(
        positions=[
            {"symbol": "AAA", "side": "long", "market_value": "500", "qty": "5"},
            {"symbol": "BBB", "side": "long", "market_value": "500", "qty": "10"},
        ]
    )
    result = run_end_of_day_flatten(client, execute=True)
    assert result["status"] == "end_of_day_flatten_submitted"
    assert client.closed_positions == ["AAA", "BBB"]


def test_end_of_day_transition_retains_overlap_and_exits_only_removed_symbols() -> None:
    client = FakePaperClient(
        positions=[
            {"symbol": "AAA", "side": "long", "market_value": "500", "qty": "5"},
            {"symbol": "OLD", "side": "long", "market_value": "500", "qty": "10"},
        ]
    )
    result = run_end_of_day_transition(
        client, {"AAA": Decimal("0.5"), "NEW": Decimal("0.5")}, execute=True
    )
    assert result["status"] == "end_of_day_transition_submitted"
    assert result["retained_symbols"] == ["AAA"]
    assert client.closed_positions == ["OLD"]


def _daily_bars() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sessions = pd.bdate_range("2026-01-02", periods=100)
    for rank in range(30):
        symbol = f"S{rank:02d}"
        drift = 0.0001 + rank * 0.00001
        for index, session in enumerate(sessions):
            rows.append(
                {
                    "timestamp": session.tz_localize("UTC") + pd.Timedelta(hours=20),
                    "symbol": symbol,
                    "close": 100 * np.exp(drift * index + 0.002 * np.sin(index + rank)),
                }
            )
    return pd.DataFrame(rows)


def test_daily_selector_scores_full_input_and_emits_capped_long_only_targets() -> None:
    bars = _daily_bars()
    as_of = "2026-06-01T14:30:00Z"
    returns = daily_simple_returns(bars, before_session=as_of)
    assert returns.index.max() < pd.Timestamp("2026-06-01")
    selection = select_daily_long_only_portfolio(
        bars,
        as_of_session=as_of,
        lookback_sessions=60,
        candidate_count=20,
        top_n=10,
        scenario_count=300,
    )
    assert selection.status["eligible_symbols"] == 30
    assert selection.targets["target_weight"].sum() == pytest.approx(1.0)
    assert (selection.targets["target_weight"] >= 0).all()
    assert (selection.targets["target_weight"] <= 0.10 + 1e-8).all()


def test_close_buffer_uses_prepared_next_target_before_transitioning(tmp_path: Path) -> None:
    client = FakePaperClient(positions=[{"symbol": "AAA", "side": "long", "market_value": "500", "qty": "5"}])
    client.get_clock = lambda: {  # type: ignore[method-assign]
        "timestamp": "2026-07-27T19:35:00Z",
        "next_close": "2026-07-27T20:00:00Z",
        "next_open": "2026-07-28T13:30:00Z",
        "is_open": True,
    }
    targets = tmp_path / "target.csv"
    status = tmp_path / "target.json"
    targets.write_text("symbol,target_weight\nBBB,0.5\nCCC,0.5\n")
    status.write_text('{"target_session": "2026-07-28", "prepared_at": "2026-07-27T14:48:00-04:00"}')
    result = run_daily_cycle(
        client,
        targets_path=targets,
        target_status_path=status,
        close_buffer_minutes=30,
        max_weight=Decimal("0.5"),
        execute=True,
    )
    assert result["cycle_phase"] == "end_of_day_transition"
    assert client.closed_positions == ["AAA"]


def test_target_file_rejects_incorrect_total(tmp_path: Path) -> None:
    path = tmp_path / "targets.csv"
    path.write_text(
        "symbol,target_weight\n"
        "AAA,0.1\nBBB,0.1\nCCC,0.1\nDDD,0.1\nEEE,0.1\n"
        "FFF,0.1\nGGG,0.1\nHHH,0.1\nIII,0.1\n"
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_target_weights(path)
