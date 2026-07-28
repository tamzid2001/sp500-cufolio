"""Causal five-minute portfolio session for a dedicated Alpaca account.

One forecast is generated and submitted at each five-minute boundary.  The
allocation is held for that exact five-minute horizon; this module deliberately
does *not* perform an additional rebalance inside a forecast window.  It uses
the same completed-bar candidate function as the research audit.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, time as clock_time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd

from .alpaca import (
    AlpacaMinuteBarStream,
    download_minute_bars,
    download_minute_endpoint_bars,
    download_yfinance_minute_bars,
)
from .five_minute_intraday_backtest import (
    NEW_YORK,
    OPENING_DECISION_PRICE_TIME,
    SESSION_CLOSE,
    SESSION_OPEN,
    _decision_grid,
    RealtimeIntradayForecastEngine,
)
from .hourly_paper_session import MinuteCacheHealth
from .paper_rebalance import AlpacaTradingClient, load_target_weights, run_end_of_day_flatten, run_rebalance
from .universe import current_sp500_universe

# The opening-inclusive five-minute grid provides 78 labels per session, so
# 780 observations require ten completed sessions. Retaining 16 sessions
# leaves fifteen completed sessions after today's first bar arrives: enough
# for the exact trailing model window while avoiding a 35-session pivot at
# every Action handoff.
MINUTE_CACHE_SESSION_LIMIT = 16
MINIMUM_HISTORY_SESSIONS = 10
MAX_EVENT_LAG = timedelta(seconds=90)
WAIT_HEARTBEAT_INTERVAL = timedelta(minutes=5)
EVENT_PRIORITY_GUARD = timedelta(seconds=45)
END_OF_DAY_HANDOFF_GUARD = timedelta(minutes=2)
FORECAST_CADENCE_MINUTES = 5
CLOSING_FORECAST_HORIZON_MINUTES = 4


@dataclass(frozen=True)
class MinuteCacheUpdate:
    """The durable cache plus only the newly completed bars for the forecast engine."""

    bars: pd.DataFrame
    new_rows: pd.DataFrame
    elapsed_seconds: float = 0.0


def _utc_timestamp(value: datetime | pd.Timestamp | None = None) -> pd.Timestamp:
    timestamp = pd.Timestamp(value or datetime.now(tz=NEW_YORK))
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must include a UTC offset")
    return timestamp.tz_convert("UTC")


def five_minute_events(session_day: date) -> list[tuple[str, pd.Timestamp]]:
    """Return 09:30--15:55 decisions and the 15:59 flat-close event.

    The first 09:30 decision uses the completed 09:29 left-labelled bar and
    forecasts the 09:30--09:35 opening window.
    The final 15:55 decision is handled by a separately trained four-minute
    model so its target ends exactly at the 15:59 flatten boundary.
    """
    return [
        ("forecast_and_order", at)
        for at in _decision_grid(
            session_day,
            interval_minutes=FORECAST_CADENCE_MINUTES,
            opening_decision=True,
        )
    ] + [
        (
            "flatten",
            (
                pd.Timestamp.combine(session_day, SESSION_CLOSE).tz_localize(NEW_YORK)
                - timedelta(minutes=1)
            ).tz_convert("UTC"),
        )
    ]


def _forecast_horizon_for_decision(decision_at: pd.Timestamp, session_day: date) -> int:
    """Use the four-minute closing horizon only for the 15:55 decision."""
    closing_decision = (
        pd.Timestamp.combine(session_day, SESSION_CLOSE).tz_localize(NEW_YORK)
        - timedelta(minutes=FORECAST_CADENCE_MINUTES)
    ).tz_convert("UTC")
    return CLOSING_FORECAST_HORIZON_MINUTES if decision_at == closing_decision else FORECAST_CADENCE_MINUTES


def _handoff_must_wait_for_flatten(
    kind: str, due_at: pd.Timestamp, stop_at: pd.Timestamp | None
) -> bool:
    """Permit a near-deadline 15:59 flatten to finish across Action handoffs."""
    return bool(
        kind == "flatten"
        and stop_at is not None
        and timedelta(0) < due_at - stop_at <= END_OF_DAY_HANDOFF_GUARD
    )


def _regular_minute_cache_projection(bars: pd.DataFrame) -> pd.DataFrame:
    """Normalize a bounded full-minute training cache without endpoint loss."""
    required = {"timestamp", "symbol", "close"}
    if missing := required.difference(bars.columns):
        raise ValueError(f"minute-bar cache is missing columns: {sorted(missing)}")
    clean = bars.loc[:, ["timestamp", "symbol", "close"]].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")
    clean["symbol"] = clean["symbol"].astype(str).str.upper().str.strip()
    clean["close"] = pd.to_numeric(clean["close"], errors="coerce")
    clean = clean.dropna(subset=["timestamp", "close"])
    clean = clean[(clean["symbol"] != "") & (clean["close"] > 0)]
    if clean.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])
    local = clean["timestamp"].dt.tz_convert(NEW_YORK)
    keep = (
        ((local.dt.time >= SESSION_OPEN) & (local.dt.time < SESSION_CLOSE))
        | (local.dt.time == OPENING_DECISION_PRICE_TIME)
    )
    clean = clean.loc[keep].copy()
    if clean.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])
    local = clean["timestamp"].dt.tz_convert(NEW_YORK)
    retained_days = sorted(local.dt.date.unique())[-MINUTE_CACHE_SESSION_LIMIT:]
    clean = clean.loc[local.dt.date.isin(retained_days)]
    return (
        clean.drop_duplicates(["symbol", "timestamp"], keep="last")
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )


def _cache_summary(bars: pd.DataFrame) -> str:
    if bars.empty:
        return "rows=0 latest=none"
    return (
        f"rows={len(bars):,} symbols={bars['symbol'].nunique()} "
        f"latest={pd.to_datetime(bars['timestamp'], utc=True).max().isoformat()}"
    )


def _read_minute_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])
    try:
        bars = _regular_minute_cache_projection(pd.read_csv(path, compression="gzip"))
        print(f"FIVE MINUTE CACHE RESTORED | path={path} {_cache_summary(bars)}", flush=True)
        return bars
    except (EOFError, OSError, UnicodeDecodeError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        print(f"FIVE MINUTE CACHE IGNORED | path={path} reason={error}", flush=True)
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])


def _write_minute_cache(path: Path, bars: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    normalized = _regular_minute_cache_projection(bars)
    normalized.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)


def _completed_bar_at(observed_at: pd.Timestamp, session_day: date) -> pd.Timestamp | None:
    completed = _utc_timestamp(observed_at).floor("min") - timedelta(minutes=1)
    opening_anchor = (
        pd.Timestamp.combine(session_day, OPENING_DECISION_PRICE_TIME)
        .tz_localize(NEW_YORK)
        .tz_convert("UTC")
    )
    close_at = pd.Timestamp.combine(session_day, SESSION_CLOSE).tz_localize(NEW_YORK).tz_convert("UTC") - timedelta(minutes=1)
    if completed < opening_anchor:
        return None
    return min(completed, close_at)


def _record_degraded(health: MinuteCacheHealth, source: str, observed_at: pd.Timestamp, error: Exception) -> None:
    retry_at = health.record_failure(source, observed_at)
    if health.should_emit_notice(observed_at):
        print(
            "FIVE MINUTE CACHE DEGRADED | "
            f"source={source} retry_after={retry_at.isoformat()} reason={error}",
            flush=True,
        )


def _merge_incremental_minute_bars(existing: pd.DataFrame, incremental: pd.DataFrame) -> pd.DataFrame:
    """Replace only overlapping rows; do not re-sort the full rolling cache."""
    if incremental.empty:
        return existing
    incoming = _regular_minute_cache_projection(incremental)
    if incoming.empty:
        return existing
    if existing.empty:
        return incoming
    existing_keys = pd.MultiIndex.from_frame(existing.loc[:, ["symbol", "timestamp"]])
    incoming_keys = pd.MultiIndex.from_frame(incoming.loc[:, ["symbol", "timestamp"]])
    merged = pd.concat([existing.loc[~existing_keys.isin(incoming_keys)], incoming], ignore_index=True)
    local_days = pd.to_datetime(merged["timestamp"], utc=True).dt.tz_convert(NEW_YORK).dt.date
    retained_days = sorted(local_days.unique())[-MINUTE_CACHE_SESSION_LIMIT:]
    return merged.loc[local_days.isin(retained_days)].reset_index(drop=True)


def _merge_current_minutes(
    bars: pd.DataFrame,
    symbols: list[str],
    *,
    session_day: date,
    observed_at: pd.Timestamp,
    cache_path: Path | None,
    minute_stream: AlpacaMinuteBarStream | None,
    health: MinuteCacheHealth,
    repair_from_rest: bool = True,
    persist: bool = True,
) -> MinuteCacheUpdate:
    """Add only completed IEX minutes, then use bounded Yahoo repair if needed.

    The websocket is always preferred.  REST requests explicitly specify IEX
    in ``download_minute_bars``; no code path asks Alpaca for recent SIP data.
    """
    refresh_started = time.monotonic()
    completed = _completed_bar_at(observed_at, session_day)
    if completed is None:
        return MinuteCacheUpdate(
            bars,
            pd.DataFrame(columns=["timestamp", "symbol", "close"]),
            time.monotonic() - refresh_started,
        )
    # ``bars`` is normalized at restore/bootstrap and every durable write.
    # Keeping it in that form avoids a multi-million-row clean/pivot in the
    # one-minute precomputation path.
    normalized = bars
    if not isinstance(normalized.get("timestamp", pd.Series(dtype="object")).dtype, pd.DatetimeTZDtype):
        normalized = _regular_minute_cache_projection(normalized)
    session_anchor = (
        pd.Timestamp.combine(session_day, OPENING_DECISION_PRICE_TIME)
        .tz_localize(NEW_YORK)
        .tz_convert("UTC")
    )
    today = pd.to_datetime(normalized["timestamp"], utc=True).dt.tz_convert(NEW_YORK).dt.date.eq(session_day)
    current = normalized.loc[today]
    # Re-request a small overlap to repair sparse websocket minutes and a
    # partial provider response without downloading the historical panel again.
    fetch_start = session_anchor if current.empty else max(session_anchor, current["timestamp"].max() - timedelta(minutes=4))
    if fetch_start > completed:
        fetch_start = completed
    incremental = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    if minute_stream is not None:
        streamed = minute_stream.completed_bars_through(completed)
        if not streamed.empty:
            incremental = streamed.loc[
                pd.to_datetime(streamed["timestamp"], utc=True, errors="coerce") >= fetch_start
            ].copy()
    source = "alpaca_iex_websocket" if not incremental.empty else "alpaca_iex_rest"
    # A stream can be connected yet legitimately sparse for IEX.  REST repair
    # fills the same five-minute overlap, so candidates are not based on stale
    # prior-day prices.
    if (repair_from_rest or incremental.empty) and health.may_attempt("alpaca_iex_rest", observed_at):
        try:
            rest = download_minute_bars(symbols, fetch_start, completed + timedelta(minutes=1))
            if not rest.empty:
                incremental = pd.concat([incremental, rest], ignore_index=True)
                source = "alpaca_iex_websocket+rest"
                health.record_success("alpaca_iex_rest", observed_at)
        except Exception as error:
            _record_degraded(health, "alpaca_iex_rest", observed_at, error)
    if incremental.empty and health.may_attempt("yfinance_1m_fallback", observed_at):
        try:
            incremental = download_yfinance_minute_bars(symbols, fetch_start, completed + timedelta(minutes=1))
            source = "yfinance_1m_fallback"
            if not incremental.empty:
                health.record_success("yfinance_1m_fallback", observed_at)
        except Exception as error:
            _record_degraded(health, "yfinance_1m_fallback", observed_at, error)
    if not incremental.empty:
        incremental = incremental.loc[
            pd.to_datetime(incremental["timestamp"], utc=True, errors="coerce") <= completed
        ]
        incremental = _regular_minute_cache_projection(incremental)
        normalized = _merge_incremental_minute_bars(normalized, incremental)
        if persist and cache_path is not None:
            _write_minute_cache(cache_path, normalized)
        print(
            "FIVE MINUTE CACHE UPDATED | "
            f"source={source} through={completed.isoformat()} rows={len(incremental):,} "
            f"elapsed={time.monotonic() - refresh_started:.3f}s {_cache_summary(normalized)}",
            flush=True,
        )
    return MinuteCacheUpdate(normalized, incremental, time.monotonic() - refresh_started)


def _prior_sessions(bars: pd.DataFrame, session_day: date) -> int:
    if bars.empty:
        return 0
    days = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(NEW_YORK).dt.date
    return int(days[days < session_day].nunique())


def _prior_opening_anchor_sessions(bars: pd.DataFrame, session_day: date) -> int:
    """Count prior sessions with the causal 09:29 opening decision price."""
    if bars.empty:
        return 0
    timestamps = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(NEW_YORK)
    anchor_days = timestamps.loc[
        (timestamps.dt.time == OPENING_DECISION_PRICE_TIME) & (timestamps.dt.date < session_day)
    ].dt.date
    return int(anchor_days.nunique())


def _default_checkpoint(session_day: date) -> dict[str, object]:
    return {"format_version": 1, "session_date": session_day.isoformat(), "completed_events": [], "ledger": []}


def _load_checkpoint(path: Path | None, session_day: date) -> dict[str, object]:
    if path is None or not path.exists():
        return _default_checkpoint(session_day)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot restore five-minute checkpoint {path}: {error}") from error
    if not isinstance(state, dict) or state.get("session_date") != session_day.isoformat():
        return _default_checkpoint(session_day)
    defaults = _default_checkpoint(session_day)
    for key, value in defaults.items():
        state.setdefault(key, value)
    if not isinstance(state["completed_events"], list) or not isinstance(state["ledger"], list):
        raise RuntimeError(f"five-minute checkpoint {path} has an invalid shape")
    return state


def _persist_checkpoint(path: Path | None, checkpoint: dict[str, object], completed: set[str], ledger: list[dict[str, object]]) -> None:
    if path is None:
        return
    checkpoint["completed_events"] = sorted(completed)
    checkpoint["ledger"] = ledger
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, indent=2, default=str) + "\n", encoding="utf-8")


def _execution_summary(report: dict[str, object]) -> str:
    submitted = report.get("submitted_orders", [])
    return (
        f"status={report.get('status', 'unknown')} market_open={report.get('market_open', 'unknown')} "
        f"planned={len(report.get('orders', []))} submitted={len(submitted) if isinstance(submitted, list) else 0}"
    )


def run_five_minute_paper_session(
    *,
    session_day: date,
    history_start: str,
    output_dir: str | Path,
    top_n: int = 20,
    max_weight: Decimal = Decimal("0.10"),
    risk_aversion: float = 10.0,
    min_order_notional: Decimal = Decimal("1"),
    min_weight_drift: Decimal = Decimal("0.0025"),
    mode: Literal["paper", "live"] = "paper",
    allow_live_trading: bool = False,
    checkpoint_path: str | Path | None = None,
    minute_cache_path: str | Path | None = None,
    historical_minute_cache_path: str | Path | None = None,
    stop_at: pd.Timestamp | None = None,
    resume: bool = False,
    now: Callable[[], pd.Timestamp] = _utc_timestamp,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, object]]:
    """Submit one fresh target at each five-minute boundary, with no intra-window rebalance."""
    if mode == "live" and not allow_live_trading:
        raise ValueError("live mode requires allow_live_trading=True")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    universe = current_sp500_universe()
    symbol_column = "source_symbol" if "source_symbol" in universe.columns else "symbol"
    symbols = universe[symbol_column].dropna().astype(str).str.upper().drop_duplicates().to_list()
    universe.to_csv(output / "current_sp500_universe.csv", index=False)
    client = AlpacaTradingClient.from_environment(mode=mode)
    state_path = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint = _load_checkpoint(state_path, session_day)
    completed = set(str(item) for item in checkpoint["completed_events"])
    ledger = [dict(item) for item in checkpoint["ledger"] if isinstance(item, dict)]
    cache_file = Path(minute_cache_path) if minute_cache_path is not None else None
    bars = _read_minute_cache(cache_file) if cache_file is not None else pd.DataFrame(columns=["timestamp", "symbol", "close"])
    if historical_minute_cache_path is not None:
        seed = _read_minute_cache(Path(historical_minute_cache_path))
        if not seed.empty:
            bars = _regular_minute_cache_projection(pd.concat([seed, bars], ignore_index=True))
    health = MinuteCacheHealth()
    stream: AlpacaMinuteBarStream | None = None
    try:
        stream = AlpacaMinuteBarStream(symbols)
        stream.start()
        print(f"FIVE MINUTE CACHE WEBSOCKET STARTED | feed=iex symbols={len(symbols)}", flush=True)
    except Exception as error:
        _record_degraded(health, "alpaca_iex_websocket", _utc_timestamp(), error)
        stream = None
    if _prior_sessions(bars, session_day) < MINIMUM_HISTORY_SESSIONS:
        try:
            historical = download_minute_bars(symbols, history_start, _utc_timestamp())
            bars = _regular_minute_cache_projection(pd.concat([bars, historical], ignore_index=True))
            if cache_file is not None:
                _write_minute_cache(cache_file, bars)
            print(f"FIVE MINUTE HISTORY BOOTSTRAPPED | {_cache_summary(bars)}", flush=True)
        except Exception as error:
            _record_degraded(health, "alpaca_iex_history", _utc_timestamp(), error)
    elif _prior_opening_anchor_sessions(bars, session_day) < MINIMUM_HISTORY_SESSIONS:
        # Older retained caches began at 09:30.  Fetch only the missing 09:29
        # endpoints rather than re-downloading the multi-million-row history,
        # so the opening model has the same causal anchor as a fresh cache.
        try:
            opening_anchors = download_minute_endpoint_bars(
                symbols,
                history_start,
                _utc_timestamp(),
                endpoint_times={OPENING_DECISION_PRICE_TIME},
            )
            bars = _regular_minute_cache_projection(pd.concat([bars, opening_anchors], ignore_index=True))
            if cache_file is not None:
                _write_minute_cache(cache_file, bars)
            print(
                "FIVE MINUTE OPENING ANCHORS BACKFILLED | "
                f"prior_sessions={_prior_opening_anchor_sessions(bars, session_day)} {_cache_summary(bars)}",
                flush=True,
            )
        except Exception as error:
            _record_degraded(health, "alpaca_iex_opening_anchor_history", _utc_timestamp(), error)
    def build_engine(forecast_horizon_minutes: int) -> RealtimeIntradayForecastEngine:
        return RealtimeIntradayForecastEngine(
            bars,
            interval_minutes=FORECAST_CADENCE_MINUTES,
            forecast_horizon_minutes=forecast_horizon_minutes,
            opening_decision=True,
            top_n=top_n,
            max_weight=float(max_weight),
            risk_aversion=risk_aversion,
        )

    engines: dict[int, RealtimeIntradayForecastEngine] = {}
    for forecast_horizon_minutes in (FORECAST_CADENCE_MINUTES, CLOSING_FORECAST_HORIZON_MINUTES):
        try:
            engines[forecast_horizon_minutes] = build_engine(forecast_horizon_minutes)
            print(
                "FIVE MINUTE FORECAST ENGINE READY | "
                f"cache_sessions={_prior_sessions(bars, session_day)} cadence=5m "
                f"horizon={forecast_horizon_minutes}m precompute_lead=60s",
                flush=True,
            )
        except Exception as error:
            # Preserve the paper account by refusing a target if a complete
            # causal panel cannot be built. Do not silently rebuild it late at
            # a decision boundary unless the exact bar repair succeeds first.
            print(
                "FIVE MINUTE FORECAST ENGINE UNAVAILABLE | "
                f"cadence=5m horizon={forecast_horizon_minutes}m reason={error}",
                flush=True,
            )
    print(
        "FIVE MINUTE SESSION STARTED | "
        f"mode={client.mode} session={session_day.isoformat()} symbols={len(symbols)} "
        f"resume={resume} completed_events={len(completed)} cache={_cache_summary(bars)}",
        flush=True,
    )

    def persist() -> None:
        _persist_checkpoint(state_path, checkpoint, completed, ledger)
        (output / "five_minute_paper_session_ledger.json").write_text(
            json.dumps(ledger, indent=2, default=str) + "\n", encoding="utf-8"
        )

    last_minute_refresh: pd.Timestamp | None = None
    prepared_event_ids: set[str] = set()
    try:
        for kind, due_at in five_minute_events(session_day):
            event_id = f"{kind}:{due_at.isoformat()}"
            if event_id in completed:
                continue
            if (
                stop_at is not None
                and now() < due_at
                and due_at > stop_at
                and not _handoff_must_wait_for_flatten(kind, due_at, stop_at)
            ):
                break
            heartbeat: pd.Timestamp | None = None
            while now() < due_at:
                current = now()
                completed_minute = _completed_bar_at(current, session_day)
                # Keep websocket bars flowing into the in-memory engine while
                # reserving the final 45 seconds for the exact decision-time
                # IEX REST repair and order submission.
                if (
                    completed_minute is not None
                    and completed_minute != last_minute_refresh
                    and due_at - current > EVENT_PRIORITY_GUARD
                ):
                    try:
                        update = _merge_current_minutes(
                            bars,
                            symbols,
                            session_day=session_day,
                            observed_at=current,
                            cache_path=cache_file,
                            minute_stream=stream,
                            health=health,
                            repair_from_rest=False,
                            persist=False,
                        )
                        bars = update.bars
                        if not update.new_rows.empty:
                            for engine in engines.values():
                                engine.update_minute_bars(update.new_rows)
                        last_minute_refresh = completed_minute
                    except Exception as error:
                        _record_degraded(health, "background_minute_precompute", current, error)
                # Training labels must end strictly before the decision, so
                # their calculation is complete one minute before the final
                # decision-price bar exists. The exact current-price screen
                # remains deferred until the boundary below.
                forecast_horizon_minutes = _forecast_horizon_for_decision(due_at, session_day)
                selected_engine = engines.get(forecast_horizon_minutes)
                if (
                    kind == "forecast_and_order"
                    and event_id not in prepared_event_ids
                    and current >= due_at - timedelta(minutes=1)
                    and selected_engine is not None
                ):
                    try:
                        selected_engine.prepare_for_decision(due_at, prepared_at=current)
                        prepared_event_ids.add(event_id)
                        print(
                            "FIVE MINUTE FORECAST PREPARED | "
                            f"decision={due_at.isoformat()} horizon={forecast_horizon_minutes}m "
                            f"prepared_at={current.isoformat()} lead_seconds="
                            f"{max(0, int((due_at - current).total_seconds()))}",
                            flush=True,
                        )
                    except (ValueError, RuntimeError) as error:
                        # A handoff can begin inside the final minute before
                        # its first decision. The boundary path still creates
                        # the engine after its exact completed-bar repair.
                        print(
                            "FIVE MINUTE FORECAST PRECOMPUTE DEFERRED | "
                            f"decision={due_at.isoformat()} reason={error}",
                            flush=True,
                        )
                        prepared_event_ids.add(event_id)
                if heartbeat is None or current - heartbeat >= WAIT_HEARTBEAT_INTERVAL:
                    print(
                        "FIVE MINUTE PAPER WAIT | "
                        f"next={kind} due={due_at.isoformat()} now={current.isoformat()} "
                        f"remaining={max(0, int((due_at-current).total_seconds()))}s {_cache_summary(bars)}",
                        flush=True,
                    )
                    heartbeat = current
                next_wakeup = due_at
                if (
                    kind == "forecast_and_order"
                    and event_id not in prepared_event_ids
                    and selected_engine is not None
                ):
                    next_wakeup = min(next_wakeup, due_at - timedelta(minutes=1))
                sleep(min(15.0, max(0.01, float((next_wakeup - current).total_seconds()))))
            observed = now()
            if stop_at is not None and observed > stop_at:
                break
            if observed > due_at + MAX_EVENT_LAG and kind != "flatten":
                entry = {"event": "skipped_stale_five_minute_event", "scheduled_at": due_at.isoformat(), "observed_at": observed.isoformat(), "kind": kind, "reason": "maximum_event_lag_exceeded"}
                ledger.append(entry)
                completed.add(event_id)
                persist()
                print(f"FIVE MINUTE EVENT SKIPPED | scheduled={due_at.isoformat()} reason=maximum_event_lag_exceeded", flush=True)
                continue
            if kind == "flatten":
                # Persist every completed minute through 15:58 before closing
                # positions.  Tomorrow's new checkpoint starts empty, while
                # this cache provides its most recent completed-session data.
                try:
                    update = _merge_current_minutes(
                        bars,
                        symbols,
                        session_day=session_day,
                        observed_at=observed,
                        cache_path=cache_file,
                        minute_stream=stream,
                        health=health,
                        repair_from_rest=True,
                        persist=True,
                    )
                    bars = update.bars
                    if not update.new_rows.empty:
                        for engine in engines.values():
                            engine.update_minute_bars(update.new_rows)
                    print(
                        "FIVE MINUTE CACHE FINALIZED | "
                        f"through={_completed_bar_at(observed, session_day).isoformat()} "
                        f"elapsed={update.elapsed_seconds:.3f}s {_cache_summary(bars)}",
                        flush=True,
                    )
                except Exception as error:
                    # A cache failure must never prevent the risk-reducing
                    # 15:59 flatten from being submitted.
                    _record_degraded(health, "final_session_cache", observed, error)
                report = run_end_of_day_flatten(client, execute=True)
                ledger.append({
                    "event": "end_of_day_flatten",
                    "scheduled_at": due_at.isoformat(),
                    "observed_at": observed.isoformat(),
                    "cache_through": _completed_bar_at(observed, session_day).isoformat(),
                    "paper_rebalance": report,
                })
                completed.add(event_id)
                persist()
                print(f"FIVE MINUTE FLATTEN RESULT | {_execution_summary(report)}", flush=True)
                continue
            try:
                forecast_horizon_minutes = _forecast_horizon_for_decision(due_at, session_day)
                update = _merge_current_minutes(
                    bars,
                    symbols,
                    session_day=session_day,
                    observed_at=observed,
                    cache_path=cache_file,
                    minute_stream=stream,
                    health=health,
                    repair_from_rest=True,
                    persist=True,
                )
                bars = update.bars
                if not update.new_rows.empty:
                    for engine in engines.values():
                        engine.update_minute_bars(update.new_rows)
                engine = engines.get(forecast_horizon_minutes)
                if engine is None:
                    engine = build_engine(forecast_horizon_minutes)
                    engines[forecast_horizon_minutes] = engine
                    print(
                        "FIVE MINUTE FORECAST ENGINE RECOVERED | "
                        f"source=exact_boundary_cache cadence=5m horizon={forecast_horizon_minutes}m",
                        flush=True,
                    )
                forecast_started = time.monotonic()
                candidate = engine.generate_candidate(due_at)
                if not candidate.status["weights_generated"]:
                    raise RuntimeError(f"causal candidate unavailable: {candidate.status.get('reason')}")
                candidate_path = output / f"candidate_{due_at.strftime('%Y%m%dT%H%MZ')}.csv"
                candidate.weights.to_csv(candidate_path, index=False)
                targets = load_target_weights(candidate_path, max_weight=max_weight)
                report = run_rebalance(
                    client, targets, min_order_notional=min_order_notional, min_weight_drift=min_weight_drift,
                    liquidate_non_target_positions=True, execute=True,
                )
                entry = {
                    "event": "five_minute_forecast_target_order", "scheduled_at": due_at.isoformat(),
                    "observed_at": observed.isoformat(), "target_end": candidate.status["target_end"],
                    "forecast_horizon_minutes": forecast_horizon_minutes,
                    "cache_refresh_seconds": update.elapsed_seconds,
                    "candidate_file": candidate_path.name, "forecast_status": candidate.status,
                    "paper_rebalance": report,
                }
                ledger.append(entry)
                completed.add(event_id)
                persist()
                print(
                    "FIVE MINUTE TARGET RESULT | "
                    f"decision={due_at.isoformat()} target_end={candidate.status['target_end']} "
                    f"horizon={forecast_horizon_minutes}m cache_elapsed={update.elapsed_seconds:.3f}s "
                    f"weights={len(targets)} forecast_elapsed={time.monotonic() - forecast_started:.3f}s "
                    f"{_execution_summary(report)}",
                    flush=True,
                )
            except Exception as error:
                # A selection that cannot use fresh, causal inputs is recorded
                # as skipped. It never reuses a prior target or submits late.
                ledger.append({"event": "skipped_unavailable_five_minute_target", "scheduled_at": due_at.isoformat(), "observed_at": observed.isoformat(), "reason": str(error)})
                completed.add(event_id)
                persist()
                print(f"FIVE MINUTE TARGET SKIPPED | decision={due_at.isoformat()} reason={error}", flush=True)
    finally:
        if stream is not None:
            stream.stop()
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one causal five-minute Alpaca portfolio session.")
    parser.add_argument("--session-date", required=True)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--minute-cache")
    parser.add_argument("--historical-minute-cache")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-weight", type=Decimal, default=Decimal("0.10"))
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--allow-live-trading", action="store_true")
    args = parser.parse_args()
    if args.mode == "live" and not args.allow_live_trading:
        parser.error("--mode live requires --allow-live-trading and ALPACA_LIVE_* credentials")
    run_five_minute_paper_session(
        session_day=date.fromisoformat(args.session_date), history_start=args.history_start,
        output_dir=args.output_dir, checkpoint_path=args.checkpoint, minute_cache_path=args.minute_cache,
        historical_minute_cache_path=args.historical_minute_cache, top_n=args.top_n,
        max_weight=args.max_weight, mode=args.mode, allow_live_trading=args.allow_live_trading,
    )


if __name__ == "__main__":
    main()
