"""Paper-only Alpaca execution for a vetted, long-only target-weight file.

This module deliberately uses Alpaca's paper endpoint directly and refuses any
other base URL.  It is separate from the research modules: an optimizer may
suggest weights, while this module only makes bounded paper-market orders to
move an account toward an already-reviewed target file.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import certifi

PAPER_API_BASE_URL = "https://paper-api.alpaca.markets/v2"
ORDER_ID_PREFIX = "cufolio-paper-"
CENT = Decimal("0.01")
SHARE_INCREMENT = Decimal("0.000000001")
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


class PaperTradingError(RuntimeError):
    """An invalid or rejected paper-trading request."""


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise PaperTradingError(f"Alpaca returned an invalid {field!r} value") from error
    if not result.is_finite():
        raise PaperTradingError(f"Alpaca returned a non-finite {field!r} value")
    return result


def _as_number(value: Decimal, increment: Decimal) -> str:
    """Render a non-scientific, downward-rounded API number."""
    rounded = value.quantize(increment, rounding=ROUND_DOWN)
    return format(rounded, "f")


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str
    field: str
    amount: Decimal
    target_weight: Decimal
    current_notional: Decimal
    target_notional: Decimal

    def payload(self, *, client_order_id: str) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            self.field: _as_number(self.amount, SHARE_INCREMENT if self.field == "qty" else CENT),
            "side": self.side,
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }

    def to_dict(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            self.field: _as_number(self.amount, SHARE_INCREMENT if self.field == "qty" else CENT),
            "target_weight": _as_number(self.target_weight, Decimal("0.000000001")),
            "current_notional": _as_number(self.current_notional, CENT),
            "target_notional": _as_number(self.target_notional, CENT),
        }


class PaperAlpacaClient:
    """Small authenticated client pinned to Alpaca's paper Trading API."""

    def __init__(self, api_key: str, secret_key: str, *, timeout_seconds: int = 20) -> None:
        if not api_key or not secret_key:
            raise PaperTradingError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        self._api_key = api_key
        self._secret_key = secret_key
        self._timeout_seconds = timeout_seconds
        self.base_url = PAPER_API_BASE_URL

    @classmethod
    def from_environment(cls) -> "PaperAlpacaClient":
        return cls(os.getenv("ALPACA_API_KEY", ""), os.getenv("ALPACA_SECRET_KEY", ""))

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        if not path.startswith("/"):
            raise ValueError("API paths must start with '/'")
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={
                "APCA-API-KEY-ID": self._api_key,
                "APCA-API-SECRET-KEY": self._secret_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds, context=TLS_CONTEXT) as response:
                raw = response.read().decode()
        except HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise PaperTradingError(f"Alpaca paper API {method} {path} returned HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise PaperTradingError(f"Alpaca paper API {method} {path} could not be reached") from error
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise PaperTradingError(f"Alpaca paper API {method} {path} returned invalid JSON") from error

    def get_account(self) -> dict[str, Any]:
        return self._request("GET", "/account")

    def get_clock(self) -> dict[str, Any]:
        return self._request("GET", "/clock")

    def get_positions(self) -> list[dict[str, Any]]:
        return self._request("GET", "/positions")

    def get_open_orders(self, symbols: list[str]) -> list[dict[str, Any]]:
        query = urlencode({"status": "open", "symbols": ",".join(symbols), "limit": 500})
        return self._request("GET", f"/orders?{query}")

    def get_asset(self, symbol: str) -> dict[str, Any]:
        return self._request("GET", f"/assets/{symbol}")

    def submit_order(self, payload: dict[str, str]) -> dict[str, Any]:
        return self._request("POST", "/orders", payload)


def load_target_weights(path: str | Path, *, max_weight: Decimal = Decimal("0.10")) -> dict[str, Decimal]:
    """Load a fully invested, long-only portfolio with an explicit concentration cap."""
    targets = pd.read_csv(path)
    required = {"symbol", "target_weight"}
    if missing := required.difference(targets.columns):
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if targets.empty:
        raise ValueError("target-weight file is empty")
    symbols = targets["symbol"].astype(str).str.upper().str.strip()
    if symbols.eq("").any() or symbols.duplicated().any():
        raise ValueError("target symbols must be non-empty and unique")
    values: dict[str, Decimal] = {}
    for symbol, value in zip(symbols, targets["target_weight"], strict=True):
        try:
            weight = Decimal(str(value))
        except InvalidOperation as error:
            raise ValueError(f"{symbol} has an invalid target_weight") from error
        if not weight.is_finite() or weight <= 0:
            raise ValueError(f"{symbol} target_weight must be positive")
        if weight > max_weight:
            raise ValueError(f"{symbol} target_weight exceeds the {max_weight} concentration cap")
        values[symbol] = weight
    total = sum(values.values(), Decimal())
    if abs(total - Decimal("1")) > Decimal("0.000001"):
        raise ValueError(f"target weights must sum to 1.0; received {total}")
    return values


def _position_map(positions: list[dict[str, Any]], targets: dict[str, Decimal]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for position in positions:
        symbol = str(position.get("symbol", "")).upper()
        if symbol not in targets:
            continue
        if str(position.get("side", "long")).lower() != "long":
            raise PaperTradingError(f"{symbol} is a short position; refusing a long-only rebalance")
        result[symbol] = position
    return result


def _ensure_fractionable_assets(client: PaperAlpacaClient, targets: dict[str, Decimal]) -> None:
    ineligible: list[str] = []
    for symbol in targets:
        asset = client.get_asset(symbol)
        if not asset.get("tradable") or not asset.get("fractionable"):
            ineligible.append(symbol)
    if ineligible:
        joined = ", ".join(ineligible)
        raise PaperTradingError(f"refusing partial rebalance: target assets are not tradable and fractionable: {joined}")


def _sell_intents(
    targets: dict[str, Decimal],
    positions: dict[str, dict[str, Any]],
    *,
    equity: Decimal,
    min_order_notional: Decimal,
    min_weight_drift: Decimal,
) -> list[OrderIntent]:
    intents: list[OrderIntent] = []
    for symbol, weight in targets.items():
        target_notional = equity * weight
        position = positions.get(symbol)
        current_notional = _decimal(position.get("market_value", "0"), field=f"{symbol} market_value") if position else Decimal()
        if current_notional <= target_notional:
            continue
        difference = current_notional - target_notional
        if difference < max(min_order_notional, equity * min_weight_drift):
            continue
        if position is None:
            continue
        held_quantity = _decimal(position.get("qty", "0"), field=f"{symbol} quantity")
        quantity = min(held_quantity, held_quantity * difference / current_notional).quantize(
            SHARE_INCREMENT, rounding=ROUND_DOWN
        )
        if quantity <= 0:
            continue
        intents.append(
            OrderIntent(symbol, "sell", "qty", quantity, weight, current_notional, target_notional)
        )
    return intents


def _buy_intents(
    targets: dict[str, Decimal],
    positions: dict[str, dict[str, Any]],
    *,
    equity: Decimal,
    cash: Decimal,
    min_order_notional: Decimal,
    min_weight_drift: Decimal,
) -> list[OrderIntent]:
    candidates: list[OrderIntent] = []
    for symbol, weight in targets.items():
        target_notional = equity * weight
        position = positions.get(symbol)
        current_notional = _decimal(position.get("market_value", "0"), field=f"{symbol} market_value") if position else Decimal()
        difference = target_notional - current_notional
        if difference >= max(min_order_notional, equity * min_weight_drift):
            candidates.append(OrderIntent(symbol, "buy", "notional", difference, weight, current_notional, target_notional))
    total_required = sum((intent.amount for intent in candidates), Decimal())
    if total_required <= 0 or cash <= 0:
        return []
    scale = min(Decimal("1"), cash / total_required)
    scaled: list[OrderIntent] = []
    for intent in candidates:
        amount = (intent.amount * scale).quantize(CENT, rounding=ROUND_DOWN)
        if amount >= min_order_notional:
            scaled.append(
                OrderIntent(
                    intent.symbol,
                    intent.side,
                    intent.field,
                    amount,
                    intent.target_weight,
                    intent.current_notional,
                    intent.target_notional,
                )
            )
    return scaled


def _client_order_id(intent: OrderIntent) -> str:
    bucket = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    return f"{ORDER_ID_PREFIX}{bucket}-{intent.side}-{intent.symbol}".lower()


def run_rebalance(
    client: PaperAlpacaClient,
    targets: dict[str, Decimal],
    *,
    min_order_notional: Decimal = Decimal("1"),
    min_weight_drift: Decimal = Decimal("0.0025"),
    execute: bool = False,
) -> dict[str, Any]:
    """Plan or execute one guarded paper-rebalance cycle.

    Existing CUFOLIO orders cause the cycle to stop.  When targets are
    overweight, only sell orders are submitted; a later 15-minute cycle buys
    after the paper broker reports the sales as filled.  This keeps the system
    cash-only and avoids stacking orders on stale account snapshots.
    """
    if min_order_notional < Decimal("1"):
        raise ValueError("min_order_notional must be at least $1.00")
    if not Decimal("0") <= min_weight_drift < Decimal("1"):
        raise ValueError("min_weight_drift must be between 0 (inclusive) and 1 (exclusive)")
    account = client.get_account()
    if account.get("account_blocked") or account.get("trading_blocked"):
        raise PaperTradingError("paper account is blocked from trading")
    clock = client.get_clock()
    report: dict[str, Any] = {
        "paper_endpoint": PAPER_API_BASE_URL,
        "execute": execute,
        "target_symbols": list(targets),
        "clock_timestamp": clock.get("timestamp"),
        "market_open": bool(clock.get("is_open")),
        "min_weight_drift": _as_number(min_weight_drift, Decimal("0.000000001")),
        "orders": [],
    }
    if not clock.get("is_open"):
        report["status"] = "market_closed"
        return report

    outstanding = client.get_open_orders(list(targets))
    if outstanding:
        report["status"] = "waiting_for_open_target_orders"
        report["open_order_ids"] = [order.get("id") for order in outstanding]
        return report

    _ensure_fractionable_assets(client, targets)
    equity = _decimal(account.get("equity"), field="equity")
    cash = _decimal(account.get("cash"), field="cash")
    if equity <= 0:
        raise PaperTradingError("paper account equity must be positive")
    if cash < 0:
        raise PaperTradingError("paper account cash is negative; refusing margin-financed rebalance")
    report["equity"] = _as_number(equity, CENT)
    report["cash"] = _as_number(cash, CENT)
    positions = _position_map(client.get_positions(), targets)

    sells = _sell_intents(
        targets,
        positions,
        equity=equity,
        min_order_notional=min_order_notional,
        min_weight_drift=min_weight_drift,
    )
    intents = sells or _buy_intents(
        targets,
        positions,
        equity=equity,
        cash=cash,
        min_order_notional=min_order_notional,
        min_weight_drift=min_weight_drift,
    )
    report["orders"] = [intent.to_dict() for intent in intents]
    if not intents:
        report["status"] = "within_rebalance_tolerance_or_no_cash"
        return report
    report["status"] = "sell_orders_planned" if sells else "buy_orders_planned"
    if not execute:
        return report

    submitted: list[dict[str, Any]] = []
    for intent in intents:
        response = client.submit_order(intent.payload(client_order_id=_client_order_id(intent)))
        submitted.append(
            {
                "symbol": intent.symbol,
                "side": intent.side,
                "id": response.get("id"),
                "status": response.get("status"),
            }
        )
    report["submitted_orders"] = submitted
    report["status"] = "sell_orders_submitted_waiting_for_fill" if sells else "buy_orders_submitted"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or execute a paper-only, long-only Alpaca portfolio rebalance.")
    parser.add_argument("--targets", required=True, help="CSV with symbol,target_weight; weights must sum to 1")
    parser.add_argument("--report", required=True, help="JSON report output path")
    parser.add_argument("--max-weight", type=Decimal, default=Decimal("0.10"))
    parser.add_argument("--min-order-notional", type=Decimal, default=Decimal("1"))
    parser.add_argument(
        "--min-weight-drift",
        type=Decimal,
        default=Decimal("0.0025"),
        help="absolute portfolio-weight drift needed to rebalance (default: 0.25%)",
    )
    parser.add_argument("--execute", action="store_true", help="submit orders to Alpaca's paper endpoint; default is plan only")
    args = parser.parse_args()
    result = run_rebalance(
        PaperAlpacaClient.from_environment(),
        load_target_weights(args.targets, max_weight=args.max_weight),
        min_order_notional=args.min_order_notional,
        min_weight_drift=args.min_weight_drift,
        execute=args.execute,
    )
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Paper rebalance: {result['status']} ({len(result['orders'])} planned orders)")


if __name__ == "__main__":
    main()
