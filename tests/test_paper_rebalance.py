from decimal import Decimal
from pathlib import Path

import pytest

from cufolio_cpu.paper_rebalance import load_target_weights, run_rebalance


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
        self.asset_requests: list[str] = []

    def get_account(self) -> dict[str, object]:
        return self.account

    def get_clock(self) -> dict[str, object]:
        return {"timestamp": "2026-07-27T14:30:00Z", "is_open": self.market_open}

    def get_positions(self) -> list[dict[str, object]]:
        return self.positions

    def get_open_orders(self, symbols: list[str]) -> list[dict[str, object]]:
        return self.open_orders

    def get_asset(self, symbol: str) -> dict[str, object]:
        self.asset_requests.append(symbol)
        return {"tradable": True, "fractionable": True}

    def submit_order(self, payload: dict[str, str]) -> dict[str, object]:
        self.submitted.append(payload)
        return {"id": f"order-{len(self.submitted)}", "status": "accepted"}


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


def test_target_file_rejects_incorrect_total(tmp_path: Path) -> None:
    path = tmp_path / "targets.csv"
    path.write_text(
        "symbol,target_weight\n"
        "AAA,0.1\nBBB,0.1\nCCC,0.1\nDDD,0.1\nEEE,0.1\n"
        "FFF,0.1\nGGG,0.1\nHHH,0.1\nIII,0.1\n"
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_target_weights(path)
