"""Daily long-only Mean-CVaR selection from the current full S&P 500 universe.

The whole current index is scored first. The optimizer is then applied to the
highest-ranked candidates using joint, bootstrapped daily-return scenarios. A
cycle uses only sessions before the current market date, so the target is fixed
throughout the trading day and does not leak an in-progress daily close.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .optimize import mean_cvar_weights
from .universe import current_sp500_universe

NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class DailySelection:
    targets: pd.DataFrame
    status: dict[str, object]


def _market_session(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(NEW_YORK)
    else:
        timestamp = timestamp.tz_convert(NEW_YORK)
    return timestamp.normalize().tz_localize(None)


def download_daily_bars(
    symbols: list[str],
    *,
    api_key: str,
    secret_key: str,
    end_session: str | datetime | pd.Timestamp,
    lookback_calendar_days: int = 180,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Download daily closes ending before ``end_session`` in New York time."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if lookback_calendar_days < 60:
        raise ValueError("lookback_calendar_days must be at least 60")
    session = _market_session(end_session)
    end = pd.Timestamp(session, tz=NEW_YORK).tz_convert("UTC").to_pydatetime()
    start = (pd.Timestamp(end) - pd.Timedelta(days=lookback_calendar_days)).to_pydatetime()

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key, secret_key)
    frames: list[pd.DataFrame] = []
    for offset in range(0, len(symbols), batch_size):
        request = StockBarsRequest(
            symbol_or_symbols=symbols[offset : offset + batch_size],
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        frame = client.get_stock_bars(request).df
        if not frame.empty:
            frames.append(frame.reset_index().loc[:, ["timestamp", "symbol", "close"]])
    if not frames:
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])
    result = pd.concat(frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    return result.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"])


def daily_simple_returns(
    daily_bars: pd.DataFrame, *, before_session: str | datetime | pd.Timestamp
) -> pd.DataFrame:
    """Create per-session simple returns without including the current session."""
    required = {"timestamp", "symbol", "close"}
    if missing := required.difference(daily_bars.columns):
        raise ValueError(f"daily bars are missing required columns: {sorted(missing)}")
    frame = daily_bars.loc[:, ["timestamp", "symbol", "close"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "close"])
    frame = frame[frame["close"] > 0]
    frame["session"] = frame["timestamp"].dt.tz_convert(NEW_YORK).dt.tz_localize(None).dt.normalize()
    frame = frame[frame["session"] < _market_session(before_session)]
    closes = frame.groupby(["session", "symbol"], sort=True)["close"].last().unstack("symbol").sort_index()
    return closes.pct_change().iloc[1:]


def _tail_cvar(losses: pd.Series, confidence: float = 0.95) -> float:
    threshold = losses.quantile(confidence)
    tail = losses[losses >= threshold]
    return float(tail.mean()) if not tail.empty else 0.0


def select_daily_long_only_portfolio(
    daily_bars: pd.DataFrame,
    *,
    as_of_session: str | datetime | pd.Timestamp,
    lookback_sessions: int = 90,
    candidate_count: int = 50,
    top_n: int = 20,
    max_weight: float = 0.10,
    scenario_count: int = 2_000,
) -> DailySelection:
    """Score all eligible symbols, then solve capped long-only Mean-CVaR weights.

    The ranking score is trailing average return divided by 95% historical tail
    loss. The selected portfolio is not a prediction or a recommendation; it
    is a reproducible daily research target for a dedicated paper strategy.
    """
    if lookback_sessions < 20:
        raise ValueError("lookback_sessions must be at least 20")
    if not 2 <= top_n <= candidate_count:
        raise ValueError("top_n must be between 2 and candidate_count")
    if candidate_count < top_n:
        raise ValueError("candidate_count must be at least top_n")
    if max_weight <= 0 or max_weight * top_n < 1 - 1e-12:
        raise ValueError("max_weight is too small for top_n")
    if scenario_count < 100:
        raise ValueError("scenario_count must be at least 100")

    returns = daily_simple_returns(daily_bars, before_session=as_of_session)
    if returns.empty:
        raise ValueError("no daily returns are available before the current session")
    trailing = returns.tail(lookback_sessions)
    minimum_history = max(20, int(lookback_sessions * 0.80))
    scores: dict[str, float] = {}
    for symbol in trailing.columns:
        observations = trailing[symbol].dropna()
        if len(observations) < minimum_history:
            continue
        tail_loss = _tail_cvar(-observations)
        scores[str(symbol)] = float(observations.mean() / max(tail_loss, 1e-6))
    ranked = pd.Series(scores, dtype=float).sort_values(ascending=False)
    candidates = ranked.head(candidate_count).index
    scenarios_history = trailing.reindex(columns=candidates).dropna(axis=0, how="any")
    if len(scenarios_history) < 20:
        raise ValueError("insufficient complete history for the highest-ranked S&P 500 candidates")
    selected = ranked.reindex(scenarios_history.columns).sort_values(ascending=False).head(top_n).index
    scenarios_history = scenarios_history.reindex(columns=selected).dropna(axis=0, how="any")
    if len(scenarios_history) < 20:
        raise ValueError("insufficient complete history for the selected daily portfolio")
    seed = int(_market_session(as_of_session).strftime("%Y%m%d"))
    generator = np.random.default_rng(seed)
    scenario_rows = generator.integers(0, len(scenarios_history), size=scenario_count)
    scenarios = scenarios_history.iloc[scenario_rows].reset_index(drop=True)
    optimized = mean_cvar_weights(scenarios, risk_aversion=5.0, confidence=0.95, max_weight=max_weight)
    weights = optimized.weights[optimized.weights > 1e-7]
    weights /= weights.sum()
    targets = (
        pd.DataFrame({"symbol": weights.index, "target_weight": weights.values})
        .assign(rank_score=lambda frame: frame["symbol"].map(ranked))
        .sort_values("target_weight", ascending=False)
        .reset_index(drop=True)
    )
    status: dict[str, object] = {
        "selection_session": _market_session(as_of_session).date().isoformat(),
        "eligible_symbols": int(len(ranked)),
        "candidate_count": int(len(candidates)),
        "optimized_symbols": int(len(targets)),
        "lookback_sessions": int(len(scenarios_history)),
        "bootstrap_scenarios": scenario_count,
        "max_weight": max_weight,
        "optimizer_status": optimized.status,
        "expected_daily_return": optimized.expected_return,
        "historical_cvar": optimized.cvar,
    }
    return DailySelection(targets=targets, status=status)


def current_sp500_daily_targets(
    *,
    api_key: str,
    secret_key: str,
    as_of_session: str | datetime | pd.Timestamp,
    lookback_calendar_days: int = 180,
    lookback_sessions: int = 90,
    candidate_count: int = 50,
    top_n: int = 20,
    max_weight: float = 0.10,
    scenario_count: int = 2_000,
) -> DailySelection:
    """Fetch the current S&P 500, then build today's fixed long-only target."""
    universe = current_sp500_universe()
    # Alpaca uses the exchange-style class-share symbols (for example BRK.B),
    # while the research CSV also carries a Yahoo-compatible dashed alias.
    symbol_column = "source_symbol" if "source_symbol" in universe.columns else "symbol"
    symbols = universe[symbol_column].astype(str).str.upper().drop_duplicates().to_list()
    bars = download_daily_bars(
        symbols,
        api_key=api_key,
        secret_key=secret_key,
        end_session=as_of_session,
        lookback_calendar_days=lookback_calendar_days,
    )
    selection = select_daily_long_only_portfolio(
        bars,
        as_of_session=as_of_session,
        lookback_sessions=lookback_sessions,
        candidate_count=candidate_count,
        top_n=top_n,
        max_weight=max_weight,
        scenario_count=scenario_count,
    )
    status = dict(selection.status)
    status.update(
        {
            "universe_symbols_requested": len(symbols),
            "universe_source": str(universe["universe_source"].iloc[0]),
            "symbols_with_daily_bars": int(bars["symbol"].nunique()),
        }
    )
    return DailySelection(selection.targets, status)
