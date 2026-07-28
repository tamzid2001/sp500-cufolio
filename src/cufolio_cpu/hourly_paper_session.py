"""Paper-first hourly portfolio forecast session with 15-minute rebalancing.

This runner is intentionally separate from the daily paper worker.  It holds
one process for a bounded regular-session window, forecasts each one-hour
target before it begins, and submits rebalances every fifteen minutes during
that target. Paper trading is the default; live mode requires a separate,
explicit acknowledgement and separate credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as clock_time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from .alpaca import (
    AlpacaMinuteBarStream,
    download_minute_bars,
    download_minute_endpoint_bars,
    download_yfinance_minute_bars,
)
from .hourly_intraday_backtest import NEW_YORK, generate_hourly_one_hour_candidate
from .paper_rebalance import AlpacaTradingClient, load_target_weights, run_end_of_day_flatten, run_rebalance
from .universe import current_sp500_universe

FORECAST_LEAD = timedelta(hours=1, minutes=10)
FINAL_TARGET_REFRESH_LEAD = timedelta(minutes=10)
TARGET_FORECAST_LEADS = (
    (10, FORECAST_LEAD),
    (11, FORECAST_LEAD),
    (12, FORECAST_LEAD),
    (13, FORECAST_LEAD),
    # The final 14:30--15:30 target is refreshed at 14:20 using the newest
    # observed minute, as requested.  It replaces the old 70-minute lead for
    # this final window; there is no duplicate target or duplicate order.
    (14, FINAL_TARGET_REFRESH_LEAD),
)
MAX_START_LAG = timedelta(minutes=10)
WAIT_HEARTBEAT_INTERVAL = timedelta(minutes=5)
EVENT_PRIORITY_GUARD = timedelta(seconds=90)
MINUTE_CACHE_FAILURE_NOTICE_INTERVAL = timedelta(minutes=5)
MINUTE_CACHE_MAX_REPAIR_BACKOFF = timedelta(minutes=5)
LIVE_MINUTE_CACHE_START = clock_time(9, 20)
LIVE_MINUTE_CACHE_END = clock_time(15, 30)
# A 120-scenario fit needs about 20 full sessions (six one-hour outcomes per
# session). Keep a buffer for holidays, sparse symbols, and the strict
# complete-case covariance step without retaining every one-minute row.
MINUTE_CACHE_SESSION_LIMIT = 35
# A 120-scenario optimization consumes six completed hourly labels per full
# session. Twenty prior sessions provide the intended trailing panel and leave
# enough room for holidays or a sparse constituent without treating an
# otherwise non-empty one-day cache as training-ready.
MINIMUM_HISTORY_SESSIONS = 20
HISTORY_LABEL_ENDPOINT_TIMES = frozenset(clock_time(hour, 30) for hour in range(9, 16))
CACHE_ENDPOINT_TIMES = frozenset(
    {
        # Exact decision minutes for the five live forecasts.
        clock_time(9, 20), clock_time(10, 20), clock_time(11, 20),
        clock_time(12, 20), clock_time(14, 20),
        # Exact regular-session endpoints needed for the historical one-hour
        # label panel: 09:30--10:30 through 14:30--15:30.
        *(clock_time(hour, 30) for hour in range(9, 16)),
    }
)


@dataclass(frozen=True)
class SessionEvent:
    kind: Literal["select", "rebalance", "flatten"]
    due_at: pd.Timestamp
    selection_at: pd.Timestamp | None = None
    target_start: pd.Timestamp | None = None


@dataclass
class MinuteCacheHealth:
    """Rate-limit transient cache-provider failures without hiding staleness.

    This state is intentionally process-local. The durable cache contains the
    data; this object only prevents a temporary IEX or Yahoo interruption from
    producing one identical error line and one full-universe fallback request
    per minute.
    """

    source: str = "cold_start"
    last_success_at: pd.Timestamp | None = None
    last_failure_at: pd.Timestamp | None = None
    retry_after: dict[str, pd.Timestamp] = field(default_factory=dict)
    failure_counts: dict[str, int] = field(default_factory=dict)
    last_notice_at: pd.Timestamp | None = None

    def may_attempt(self, source: str, observed_at: pd.Timestamp) -> bool:
        return observed_at >= self.retry_after.get(source, observed_at)

    def record_success(self, source: str, observed_at: pd.Timestamp) -> None:
        self.source = source
        self.last_success_at = observed_at
        self.failure_counts.pop(source, None)
        self.retry_after.pop(source, None)

    def record_failure(self, source: str, observed_at: pd.Timestamp) -> pd.Timestamp:
        attempts = self.failure_counts.get(source, 0) + 1
        self.failure_counts[source] = attempts
        delay = min(
            MINUTE_CACHE_MAX_REPAIR_BACKOFF,
            timedelta(seconds=30 * (2 ** min(attempts - 1, 4))),
        )
        retry_at = observed_at + delay
        self.source = f"degraded_{source}"
        self.last_failure_at = observed_at
        self.retry_after[source] = retry_at
        return retry_at

    def should_emit_notice(self, observed_at: pd.Timestamp) -> bool:
        if self.last_notice_at is None or observed_at - self.last_notice_at >= MINUTE_CACHE_FAILURE_NOTICE_INTERVAL:
            self.last_notice_at = observed_at
            return True
        return False


def _cache_health_summary(health: MinuteCacheHealth) -> str:
    successful = health.last_success_at.isoformat() if health.last_success_at is not None else "none"
    return f"cache_source={health.source} cache_last_success={successful}"


def session_events(session_day: date) -> list[SessionEvent]:
    """Return the exact paper-session schedule for a New York session date.

    Forecasts are made at 09:20, 10:20, 11:20, 12:20, and 14:20 New York
    time using only the exact observed selection-minute price. They target
    the respectively later 10:30--11:30 through 14:30--15:30 windows. The
    final 14:20 selection produces the 14:30--15:30 target requested by the
    user. Each target is first rebalanced at its start and then at +15, +30,
    and +45 minutes, followed by a 15:30 flatten.
    """
    events: list[SessionEvent] = []
    for hour, forecast_lead in TARGET_FORECAST_LEADS:
        target_start = pd.Timestamp.combine(session_day, clock_time(hour, 30)).tz_localize(NEW_YORK).tz_convert("UTC")
        selection_at = target_start - forecast_lead
        events.append(SessionEvent("select", selection_at, selection_at, target_start))
        for minutes in (0, 15, 30, 45):
            events.append(
                SessionEvent(
                    "rebalance",
                    target_start + timedelta(minutes=minutes),
                    selection_at,
                    target_start,
                )
            )
    flatten_at = pd.Timestamp.combine(session_day, clock_time(15, 30)).tz_localize(NEW_YORK).tz_convert("UTC")
    events.append(SessionEvent("flatten", flatten_at))
    return sorted(events, key=lambda event: event.due_at)


def _utc_timestamp(value: datetime | pd.Timestamp | None = None) -> pd.Timestamp:
    timestamp = pd.Timestamp(value or datetime.now(tz=NEW_YORK))
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must include a UTC offset")
    return timestamp.tz_convert("UTC")


def _cache_summary(bars: pd.DataFrame | None) -> str:
    if bars is None or bars.empty:
        return "rows=0 latest=none"
    latest = pd.to_datetime(bars["timestamp"], utc=True).max().isoformat()
    return f"rows={len(bars):,} symbols={bars['symbol'].nunique()} latest={latest}"


def _execution_summary(report: dict[str, object]) -> str:
    """Return a compact, secret-free paper-order status line."""
    planned = len(report.get("orders", [])) if isinstance(report.get("orders"), list) else 0
    submitted = len(report.get("submitted_orders", [])) if isinstance(report.get("submitted_orders"), list) else 0
    details = [
        f"status={report.get('status', 'unknown')}",
        f"market_open={report.get('market_open', 'unknown')}",
        f"planned={planned}",
        f"submitted={submitted}",
    ]
    for field in ("equity", "cash"):
        if field in report:
            details.append(f"{field}=${report[field]}")
    return " ".join(details)


def _history_through(
    bars: pd.DataFrame | None,
    symbols: list[str],
    *,
    start: str,
    decision_at: pd.Timestamp,
    cache_path: Path | None = None,
    minute_stream: AlpacaMinuteBarStream | None = None,
    force_history_bootstrap: bool = False,
) -> pd.DataFrame:
    """Fetch only missing minute bars and merge them into the durable cache.

    Older sessions retain only model-required endpoints.  The current session
    retains all completed minutes so the next forecast has current data ready
    without a multi-week re-download.
    """
    end = decision_at + timedelta(minutes=1)
    requested_symbols = {symbol.upper().strip() for symbol in symbols}
    streamed = minute_stream.completed_bars_through(decision_at) if minute_stream is not None else pd.DataFrame()
    if force_history_bootstrap:
        # The durable historical portion retains exact decision and label
        # endpoints only. Filter each IEX response before retaining it instead
        # of materializing every one-minute row for the full S&P universe.
        combined = download_minute_endpoint_bars(
            symbols,
            start,
            end,
            endpoint_times=CACHE_ENDPOINT_TIMES,
        )
        if bars is not None and not bars.empty:
            combined = pd.concat([bars, combined], ignore_index=True)
    elif bars is None or bars.empty:
        try:
            combined = download_minute_bars(symbols, start, end)
        except Exception as error:
            # A full training bootstrap needs Alpaca IEX history. Yahoo's
            # one-minute data is only a bounded, latest-bar repair path.
            if pd.Timestamp(end) - pd.Timestamp(start) > pd.Timedelta(days=8):
                raise RuntimeError(
                    "Alpaca IEX historical bootstrap was unavailable; refusing to replace the required "
                    "multi-session training panel with a short Yahoo request"
                ) from error
            combined = download_yfinance_minute_bars(symbols, start, end)
    else:
        existing = _minute_cache_projection(bars)
        # A constituent can change between sessions.  Never allow a cached
        # former constituent to enter today's optimizer or order target.
        existing = existing.loc[existing["symbol"].isin(requested_symbols)].copy()
        if existing.empty:
            combined = download_minute_bars(symbols, start, end)
        else:
            latest = existing["timestamp"].max()
            if latest >= decision_at:
                combined = existing
            else:
                latest_local_day = latest.tz_convert(NEW_YORK).date()
                decision_local_day = decision_at.tz_convert(NEW_YORK).date()
                # Do not refill the overnight gap on a new session: historical
                # labels are already cached and this decision needs only today's
                # exact observation.  Within a session, overlap the final minute
                # so a partial response is repaired and deduplication retains the
                # newest authoritative close.
                fetch_start = decision_at if latest_local_day < decision_local_day else latest
                try:
                    incremental = download_minute_bars(symbols, fetch_start, end)
                except Exception:
                    incremental = download_yfinance_minute_bars(symbols, fetch_start, end)
                combined = pd.concat([existing, incremental], ignore_index=True) if not incremental.empty else existing
    if not streamed.empty:
        combined = pd.concat([combined, streamed], ignore_index=True)
    compact = _minute_cache_projection(combined)
    if cache_path is not None:
        _write_minute_cache(cache_path, compact)
    return compact


def _complete_history_sessions(bars: pd.DataFrame | None, session_day: date) -> int:
    """Count earlier local sessions that retain every hourly label endpoint."""
    if bars is None or bars.empty:
        return 0
    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    local = timestamps.dt.tz_convert(NEW_YORK)
    earlier = local.loc[local.dt.date < session_day]
    endpoint_rows = earlier.loc[earlier.dt.time.isin(HISTORY_LABEL_ENDPOINT_TIMES)]
    if endpoint_rows.empty:
        return 0
    endpoint_sets = endpoint_rows.groupby(endpoint_rows.dt.date).agg(lambda values: set(values.dt.time))
    return sum(HISTORY_LABEL_ENDPOINT_TIMES.issubset(times) for times in endpoint_sets)


def _minute_cache_projection(bars: pd.DataFrame) -> pd.DataFrame:
    """Normalize, deduplicate, and bound the cross-handoff minute cache.

    Completed prior sessions are compacted to exact optimizer endpoints.
    The newest session preserves every 09:20--15:30 minute, allowing a
    once-per-minute updater to make current data available before later
    forecast/rebalance events without allowing the cache to grow unbounded.
    """
    required = {"timestamp", "symbol", "close"}
    if missing := required.difference(bars.columns):
        raise ValueError(f"minute-bar cache is missing columns: {sorted(missing)}")
    compact = bars.loc[:, ["timestamp", "symbol", "close"]].copy()
    compact["timestamp"] = pd.to_datetime(compact["timestamp"], utc=True, errors="coerce")
    compact["symbol"] = compact["symbol"].astype(str).str.upper().str.strip()
    compact["close"] = pd.to_numeric(compact["close"], errors="coerce")
    compact = compact.dropna(subset=["timestamp", "close"])
    compact = compact[(compact["symbol"] != "") & (compact["close"] > 0)]
    local = compact["timestamp"].dt.tz_convert(NEW_YORK)
    if compact.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])
    local_dates = local.dt.tz_localize(None).dt.normalize()
    retained_dates = sorted(local_dates.unique())[-MINUTE_CACHE_SESSION_LIMIT:]
    compact = compact.loc[local_dates.isin(retained_dates)]
    local = local.loc[compact.index]
    local_dates = local_dates.loc[compact.index]
    newest_session = max(retained_dates)
    is_endpoint = local.dt.time.isin(CACHE_ENDPOINT_TIMES)
    is_current_session_minute = (
        (local_dates == newest_session)
        & (local.dt.time >= LIVE_MINUTE_CACHE_START)
        & (local.dt.time <= LIVE_MINUTE_CACHE_END)
    )
    compact = compact.loc[is_endpoint | is_current_session_minute]
    return compact.drop_duplicates(["symbol", "timestamp"], keep="last").sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _last_completed_session_minute(observed_at: pd.Timestamp, session_day: date) -> pd.Timestamp | None:
    """Return the latest fully closed 09:20--15:30 New York minute."""
    observed = _utc_timestamp(observed_at)
    session_open = pd.Timestamp.combine(session_day, LIVE_MINUTE_CACHE_START).tz_localize(NEW_YORK).tz_convert("UTC")
    session_close = pd.Timestamp.combine(session_day, LIVE_MINUTE_CACHE_END).tz_localize(NEW_YORK).tz_convert("UTC")
    completed = observed.floor("min") - timedelta(minutes=1)
    if completed < session_open:
        return None
    return min(completed, session_close)


def _record_cache_provider_failure(
    health: MinuteCacheHealth | None,
    *,
    source: str,
    observed_at: pd.Timestamp,
    error: Exception,
) -> None:
    """Emit one bounded, secret-free cache-health notice per outage period."""
    if health is None:
        print(f"MINUTE CACHE DEGRADED | source={source} reason={error}", flush=True)
        return
    retry_at = health.record_failure(source, observed_at)
    if health.should_emit_notice(observed_at):
        print(
            "MINUTE CACHE DEGRADED | "
            f"source={source} retry_after={retry_at.isoformat()} reason={error}",
            flush=True,
        )


def _refresh_current_session_minutes(
    bars: pd.DataFrame | None,
    symbols: list[str],
    *,
    session_day: date,
    observed_at: pd.Timestamp,
    cache_path: Path | None,
    minute_stream: AlpacaMinuteBarStream | None = None,
    cache_health: MinuteCacheHealth | None = None,
) -> pd.DataFrame:
    """Append completed IEX stream bars and repair only when the stream is unavailable.

    The live order of preference is IEX websocket, explicit IEX REST, then a
    bounded Yahoo one-minute repair.  No code path leaves the feed implicit,
    so a paper account cannot accidentally retry a recent SIP request.
    """
    observed = _utc_timestamp(observed_at)
    completed = _last_completed_session_minute(observed_at, session_day)
    if completed is None:
        return bars if bars is not None else pd.DataFrame(columns=["timestamp", "symbol", "close"])
    session_open = pd.Timestamp.combine(session_day, LIVE_MINUTE_CACHE_START).tz_localize(NEW_YORK).tz_convert("UTC")
    existing = _minute_cache_projection(bars) if bars is not None and not bars.empty else pd.DataFrame(
        columns=["timestamp", "symbol", "close"]
    )
    requested_symbols = {symbol.upper().strip() for symbol in symbols}
    existing = existing.loc[existing["symbol"].isin(requested_symbols)].copy()
    if existing.empty:
        fetch_start = session_open
    else:
        local = existing["timestamp"].dt.tz_convert(NEW_YORK)
        today = local.dt.date == session_day
        current_session = existing.loc[
            today & (local.dt.time >= LIVE_MINUTE_CACHE_START) & (local.dt.time <= LIVE_MINUTE_CACHE_END)
        ]
        fetch_start = session_open if current_session.empty else current_session["timestamp"].max() + timedelta(minutes=1)
    if fetch_start > completed:
        return existing
    refresh_started = time.monotonic()
    incremental = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    source = "alpaca_iex_websocket"
    if minute_stream is not None:
        incremental = minute_stream.completed_bars_through(completed)
        incremental = incremental.loc[
            pd.to_datetime(incremental["timestamp"], utc=True, errors="coerce") >= fetch_start
        ].copy()
    stream_available = (
        minute_stream is not None
        and bool(getattr(minute_stream, "available", getattr(minute_stream, "connected", False)))
    )
    if incremental.empty and stream_available:
        # During an authenticated-startup interval or a quiet IEX minute the
        # websocket can have no completed bar yet. Wait for it instead of
        # issuing a redundant full-universe request every minute.
        return existing
    if incremental.empty:
        # A stream failure must first repair through the explicitly entitled
        # IEX REST feed.  Yahoo is a bounded last resort only after that IEX
        # request fails or returns no usable current-session row.
        iex_source = "alpaca_iex_rest_repair"
        if cache_health is None or cache_health.may_attempt(iex_source, observed):
            try:
                incremental = download_minute_bars(symbols, fetch_start, completed + timedelta(minutes=1))
                source = iex_source
                if incremental.empty and cache_health is not None:
                    cache_health.record_failure(iex_source, observed)
            except Exception as error:
                _record_cache_provider_failure(
                    cache_health,
                    source=iex_source,
                    observed_at=observed,
                    error=error,
                )
        if incremental.empty:
            yahoo_source = "yfinance_1m_fallback"
            if cache_health is None or cache_health.may_attempt(yahoo_source, observed):
                try:
                    incremental = download_yfinance_minute_bars(
                        symbols, fetch_start, completed + timedelta(minutes=1)
                    )
                    source = yahoo_source
                    if incremental.empty and cache_health is not None:
                        cache_health.record_failure(yahoo_source, observed)
                except Exception as error:
                    _record_cache_provider_failure(
                        cache_health,
                        source=yahoo_source,
                        observed_at=observed,
                        error=error,
                    )
        if incremental.empty:
            return existing
    # Treat provider end-boundary semantics conservatively: even if a response
    # happens to include the next in-progress timestamp, it cannot enter this
    # causal cache until that minute has closed on a later poll.
    if not incremental.empty:
        incremental = incremental.loc[
            pd.to_datetime(incremental["timestamp"], utc=True, errors="coerce") <= completed
        ].copy()
    combined = pd.concat([existing, incremental], ignore_index=True) if not incremental.empty else existing
    merged = _minute_cache_projection(combined)
    if cache_path is not None:
        _write_minute_cache(cache_path, merged)
    if cache_health is not None:
        cache_health.record_success(source, observed)
    print(
        "MINUTE CACHE REFRESHED | "
        f"source={source} from={fetch_start.isoformat()} through={completed.isoformat()} downloaded={len(incremental):,} "
        f"elapsed={time.monotonic() - refresh_started:.1f}s {_cache_summary(merged)}",
        flush=True,
    )
    return merged


def _read_minute_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"MINUTE CACHE MISS | path={path}", flush=True)
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])
    try:
        restored = _minute_cache_projection(pd.read_csv(path, compression="gzip"))
        print(f"MINUTE CACHE RESTORED | path={path} {_cache_summary(restored)}", flush=True)
        return restored
    except (EOFError, OSError, UnicodeDecodeError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        # A cache is an optimization only. A malformed or partial cache must
        # never become model input; the next selection will refetch safely.
        print(f"MINUTE CACHE IGNORED | path={path} reason={error}", flush=True)
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])


def _write_minute_cache(path: Path, bars: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    _minute_cache_projection(bars).to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)
    print(
        f"MINUTE CACHE SAVED | path={path} rows={len(bars):,} "
        f"latest={bars['timestamp'].max().isoformat() if not bars.empty else 'none'}",
        flush=True,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _event_key(event: SessionEvent) -> str:
    return f"{event.kind}:{event.due_at.isoformat()}"


def _target_key(target_start: pd.Timestamp) -> str:
    return _utc_timestamp(target_start).isoformat()


def _default_checkpoint(session_day: date) -> dict[str, Any]:
    return {
        "format_version": 1,
        "session_date": session_day.isoformat(),
        "completed_events": [],
        "targets_by_start": {},
        "forecast_details": {},
        "ledger": [],
    }


def _load_checkpoint(path: Path, session_day: date) -> dict[str, Any]:
    if not path.exists():
        return _default_checkpoint(session_day)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot restore hourly paper checkpoint {path}: {error}") from error
    if not isinstance(state, dict) or state.get("session_date") != session_day.isoformat():
        return _default_checkpoint(session_day)
    defaults = _default_checkpoint(session_day)
    for key, value in defaults.items():
        state.setdefault(key, value)
    if not all(isinstance(state[key], expected) for key, expected in (
        ("completed_events", list), ("targets_by_start", dict),
        ("forecast_details", dict), ("ledger", list),
    )):
        raise RuntimeError(f"hourly paper checkpoint {path} has an invalid shape")
    return state


def _restore_targets(checkpoint: dict[str, Any]) -> dict[pd.Timestamp, dict[str, Decimal]]:
    restored: dict[pd.Timestamp, dict[str, Decimal]] = {}
    for target_start, raw_weights in checkpoint["targets_by_start"].items():
        if not isinstance(raw_weights, dict):
            raise RuntimeError("hourly paper checkpoint has invalid target weights")
        restored[_utc_timestamp(pd.Timestamp(target_start))] = {
            str(symbol): Decimal(str(weight)) for symbol, weight in raw_weights.items()
        }
    return restored


def _persist_checkpoint(
    path: Path | None,
    checkpoint: dict[str, Any],
    *,
    completed_events: set[str],
    targets_by_start: dict[pd.Timestamp, dict[str, Decimal]],
    forecast_details: dict[pd.Timestamp, dict[str, object]],
    ledger: list[dict[str, object]],
) -> None:
    if path is None:
        return
    checkpoint.update({
        "completed_events": sorted(completed_events),
        "targets_by_start": {
            _target_key(target_start): {symbol: format(weight, "f") for symbol, weight in weights.items()}
            for target_start, weights in targets_by_start.items()
        },
        "forecast_details": {_target_key(target_start): details for target_start, details in forecast_details.items()},
        "ledger": ledger,
    })
    _write_json(path, checkpoint)


def run_hourly_paper_session(
    *,
    session_day: date,
    history_start: str,
    output_dir: str | Path,
    top_n: int = 20,
    lookback_scenarios: int = 120,
    min_training_scenarios: int = 20,
    max_weight: Decimal = Decimal("0.10"),
    risk_aversion: float = 10.0,
    min_order_notional: Decimal = Decimal("1"),
    min_weight_drift: Decimal = Decimal("0.0025"),
    mode: Literal["paper", "live"] = "paper",
    allow_live_trading: bool = False,
    checkpoint_path: str | Path | None = None,
    minute_cache_path: str | Path | None = None,
    minute_cache_polling: bool = True,
    stop_at: pd.Timestamp | None = None,
    resume: bool = False,
    now: Callable[[], pd.Timestamp] = _utc_timestamp,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, object]]:
    """Run the bounded selection/rebalance/flatten session.

    A delayed GitHub Actions start of more than ten minutes after a planned
    selection fails closed before submitting any stale order. ``resume`` is
    the 24/7 handoff mode: completed events and selected weights are restored
    from ``checkpoint_path``, while missed events are recorded as skipped
    rather than submitted late. ``live`` is deliberately unavailable unless
    ``allow_live_trading`` is explicit.
    """
    if mode == "live" and not allow_live_trading:
        raise ValueError("live mode requires allow_live_trading=True")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    universe = current_sp500_universe()
    symbol_column = "source_symbol" if "source_symbol" in universe.columns else "symbol"
    symbols = universe[symbol_column].dropna().astype(str).str.upper().drop_duplicates().to_list()
    universe.to_csv(output / "current_sp500_universe.csv", index=False)
    client = AlpacaTradingClient.from_environment(mode=mode)
    checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint = _load_checkpoint(checkpoint_file, session_day) if checkpoint_file is not None else _default_checkpoint(session_day)
    completed_events = set(str(item) for item in checkpoint["completed_events"])
    minute_cache_file = Path(minute_cache_path) if minute_cache_path is not None else None
    bars: pd.DataFrame | None = _read_minute_cache(minute_cache_file) if minute_cache_file is not None else None
    minute_stream: AlpacaMinuteBarStream | None = None
    minute_cache_health = MinuteCacheHealth()
    if minute_cache_polling:
        try:
            minute_stream = AlpacaMinuteBarStream(symbols)
            minute_stream.start()
            print(f"MINUTE CACHE WEBSOCKET STARTED | feed=iex symbols={len(symbols)}", flush=True)
        except Exception as error:
            # Never quietly switch to SIP. The scheduled IEX / Yahoo paths
            # below remain available and the exact failure is logged.
            minute_cache_health.record_failure("alpaca_iex_websocket", _utc_timestamp())
            print(f"MINUTE CACHE WEBSOCKET UNAVAILABLE | feed=iex reason={error}", flush=True)
    targets_by_start = _restore_targets(checkpoint)
    forecast_details = {
        _utc_timestamp(pd.Timestamp(target_start)): dict(details)
        for target_start, details in checkpoint["forecast_details"].items()
        if isinstance(details, dict)
    }
    ledger: list[dict[str, object]] = [dict(item) for item in checkpoint["ledger"] if isinstance(item, dict)]
    print(
        "HOURLY PAPER SESSION STARTED | "
        f"mode={client.mode} session={session_day.isoformat()} symbols={len(symbols)} "
        f"resume={resume} completed_events={len(completed_events)} checkpoint={checkpoint_file or 'none'} "
        f"cache={minute_cache_file or 'none'} minute_polling={minute_cache_polling} {_cache_summary(bars)}",
        flush=True,
    )

    def persist() -> None:
        _persist_checkpoint(
            checkpoint_file,
            checkpoint,
            completed_events=completed_events,
            targets_by_start=targets_by_start,
            forecast_details=forecast_details,
            ledger=ledger,
        )
        _write_json(output / "hourly_paper_session_ledger.json", ledger)

    def skip(event: SessionEvent, reason: str, observed_at: pd.Timestamp) -> None:
        completed_events.add(_event_key(event))
        ledger.append({
            "event": "skipped_stale_hourly_event",
            "scheduled_at": event.due_at.isoformat(),
            "observed_at": observed_at.isoformat(),
            "kind": event.kind,
            "target_start": event.target_start.isoformat() if event.target_start is not None else None,
            "reason": reason,
        })
        persist()
        print(
            "HOURLY PAPER EVENT SKIPPED | "
            f"kind={event.kind} scheduled={event.due_at.isoformat()} observed={observed_at.isoformat()} reason={reason}",
            flush=True,
        )
    # The full S&P one-minute history is the expensive operation. Build it
    # whenever the cache has fewer than twenty complete prior sessions, even
    # if a handoff begins after 09:20. Without this recovery path, a failed
    # first-day cache would retain a few live rows forever but never contain
    # enough completed labels for an hourly forecast.
    events = session_events(session_day)
    first_selection = events[0].selection_at
    assert first_selection is not None
    prefetch_now = now()
    if (
        _complete_history_sessions(bars, session_day) < MINIMUM_HISTORY_SESSIONS
        and (stop_at is None or prefetch_now <= stop_at)
    ):
        print(
            "HOURLY HISTORY BOOTSTRAP STARTED | "
            f"start={history_start} through={prefetch_now.isoformat()} symbols={len(symbols)} "
            f"complete_sessions={_complete_history_sessions(bars, session_day)}",
            flush=True,
        )
        bootstrap_started = time.monotonic()
        bars = _history_through(
            bars,
            symbols,
            start=history_start,
            decision_at=prefetch_now,
            cache_path=minute_cache_file,
            minute_stream=minute_stream,
            force_history_bootstrap=True,
        )
        print(
            "HOURLY HISTORY BOOTSTRAP READY | "
            f"elapsed={time.monotonic() - bootstrap_started:.1f}s {_cache_summary(bars)}",
            flush=True,
        )

    for event in events:
        event_id = _event_key(event)
        if event_id in completed_events:
            continue
        if stop_at is not None and now() < event.due_at and event.due_at > stop_at:
            break
        last_wait_heartbeat: pd.Timestamp | None = None
        last_minute_refresh: pd.Timestamp | None = None
        while now() < event.due_at:
            current = now()
            # The updater requests only completed current-session minutes and
            # yields the final 90 seconds before every selection, rebalance,
            # or flatten.  The scheduled trading event therefore always has
            # priority over data I/O.
            completed_minute = _last_completed_session_minute(current, session_day) if minute_cache_polling else None
            if (
                completed_minute is not None
                and completed_minute != last_minute_refresh
                and event.due_at - current > EVENT_PRIORITY_GUARD
            ):
                last_minute_refresh = completed_minute
                try:
                    bars = _refresh_current_session_minutes(
                        bars,
                        symbols,
                        session_day=session_day,
                        observed_at=current,
                        cache_path=minute_cache_file,
                        minute_stream=minute_stream,
                        cache_health=minute_cache_health,
                    )
                except Exception as error:
                    # This background cache is an acceleration only.  A
                    # failed minute refresh must never delay or suppress the
                    # next paper order; the scheduled forecast fetch remains
                    # the authoritative fail-closed data path.
                    _record_cache_provider_failure(
                        minute_cache_health,
                        source="cache_refresh",
                        observed_at=current,
                        error=error,
                    )
            if last_wait_heartbeat is None or current - last_wait_heartbeat >= WAIT_HEARTBEAT_INTERVAL:
                remaining = max(0, int((event.due_at - current).total_seconds()))
                print(
                    "HOURLY PAPER WAIT | "
                    f"next={event.kind} due={event.due_at.isoformat()} now={current.isoformat()} "
                    f"remaining={remaining}s {_cache_summary(bars)} {_cache_health_summary(minute_cache_health)}",
                    flush=True,
                )
                last_wait_heartbeat = current
            sleep(min(30.0, float((event.due_at - current).total_seconds())))
        observed_at = now()
        if stop_at is not None and observed_at > stop_at:
            break
        if event.kind != "flatten" and observed_at > event.due_at + MAX_START_LAG:
            if resume:
                skip(event, "runner_reached_event_after_maximum_start_lag", observed_at)
                continue
            raise RuntimeError(
                f"{event.kind} at {event.due_at.isoformat()} is too late: runner reached it at {observed_at.isoformat()}"
            )
        if event.kind == "select":
            assert event.selection_at is not None
            assert event.target_start is not None
            print(
                "HOURLY FORECAST STARTED | "
                f"selection={event.selection_at.isoformat()} target={event.target_start.isoformat()} {_cache_summary(bars)}",
                flush=True,
            )
            forecast_started = time.monotonic()
            bars = _history_through(
                bars,
                symbols,
                start=history_start,
                decision_at=event.selection_at,
                cache_path=minute_cache_file,
                minute_stream=minute_stream,
            )
            if now() > event.due_at + MAX_START_LAG:
                if resume:
                    skip(event, "history_was_not_available_before_maximum_start_lag", now())
                    continue
                raise RuntimeError(
                    f"history for {event.selection_at.isoformat()} was not available in time; "
                    "refusing a stale forecast"
                )
            candidate = generate_hourly_one_hour_candidate(
                bars,
                decision_at=event.selection_at,
                target_start_at=event.target_start,
                top_n=top_n,
                lookback_scenarios=lookback_scenarios,
                min_training_scenarios=min_training_scenarios,
                max_weight=float(max_weight),
                risk_aversion=risk_aversion,
            )
            stamp = f"{event.target_start.strftime('%H%M')}_from_{event.selection_at.strftime('%H%M')}"
            candidate_path = output / f"candidate_for_{stamp}.csv"
            candidate.weights.to_csv(candidate_path, index=False)
            _write_json(output / f"candidate_{stamp}_status.json", candidate.status)
            if not candidate.status["weights_generated"]:
                raise RuntimeError(
                    f"no exact causal forecast for {event.target_start.isoformat()} "
                    f"from {event.selection_at.isoformat()}: {candidate.status}"
                )
            targets_by_start[event.target_start] = load_target_weights(candidate_path, max_weight=max_weight)
            expected_log = float(candidate.status["expected_one_hour_log_return"])
            top_weights = candidate.weights.nlargest(5, "target_weight")
            top_weight_text = ",".join(
                f"{row.symbol}:{float(row.target_weight):.2%}" for row in top_weights.itertuples(index=False)
            )
            print(
                "HOURLY FORECAST READY | "
                f"target={event.target_start.isoformat()} weights={len(candidate.weights)} "
                f"training_rows={candidate.status.get('training_rows', 0)} "
                f"expected_one_hour_return={np.expm1(expected_log):+.4%} "
                f"elapsed={time.monotonic() - forecast_started:.1f}s top={top_weight_text}",
                flush=True,
            )
            forecast_details[event.target_start] = {
                "selection_timestamp": event.selection_at.isoformat(),
                "target_start": event.target_start.isoformat(),
                "target_end": candidate.status["target_end"],
                "trading_mode": client.mode,
                "expected_one_hour_log_return": expected_log,
                "expected_one_hour_simple_return": float(np.expm1(expected_log)),
                "candidate_file": candidate_path.name,
            }
            ledger.append(
                {
                    "event": "forecast_selected",
                    "scheduled_at": event.due_at.isoformat(),
                    "observed_at": observed_at.isoformat(),
                    **forecast_details[event.target_start],
                }
            )
            completed_events.add(event_id)
        elif event.kind == "rebalance":
            assert event.selection_at is not None
            assert event.target_start is not None
            if event.target_start not in targets_by_start:
                if resume:
                    skip(event, "selected_target_was_not_available_after_handoff", observed_at)
                    continue
                raise RuntimeError(f"missing active target for {event.due_at.isoformat()}")
            print(
                "HOURLY PAPER REBALANCE STARTED | "
                f"scheduled={event.due_at.isoformat()} target={event.target_start.isoformat()} "
                f"weights={len(targets_by_start[event.target_start])}",
                flush=True,
            )
            report = run_rebalance(
                client,
                targets_by_start[event.target_start],
                min_order_notional=min_order_notional,
                min_weight_drift=min_weight_drift,
                liquidate_non_target_positions=True,
                execute=True,
            )
            print(
                "HOURLY PAPER REBALANCE RESULT | "
                f"scheduled={event.due_at.isoformat()} {_execution_summary(report)}",
                flush=True,
            )
            ledger.append(
                {
                    "event": "quarter_hour_paper_rebalance",
                    "scheduled_at": event.due_at.isoformat(),
                    "observed_at": observed_at.isoformat(),
                    **forecast_details[event.target_start],
                    "paper_rebalance": report,
                }
            )
            completed_events.add(event_id)
        else:
            print(f"HOURLY PAPER FLATTEN STARTED | scheduled={event.due_at.isoformat()}", flush=True)
            report = run_end_of_day_flatten(client, execute=True)
            # Capture the final 14:30 and 15:30 label endpoints after the
            # close.  They are never used by an earlier forecast, but make
            # tomorrow's trailing one-hour panel as current as possible.
            bars = _history_through(
                bars,
                symbols,
                start=history_start,
                decision_at=event.due_at,
                cache_path=minute_cache_file,
                minute_stream=minute_stream,
            )
            print(
                "HOURLY PAPER FLATTEN RESULT | "
                f"scheduled={event.due_at.isoformat()} {_execution_summary(report)} {_cache_summary(bars)}",
                flush=True,
            )
            ledger.append(
                {
                    "event": "final_paper_flatten",
                    "scheduled_at": event.due_at.isoformat(),
                    "observed_at": observed_at.isoformat(),
                    "paper_rebalance": report,
                }
            )
            completed_events.add(event_id)
        persist()
        print(
            "HOURLY PAPER CHECKPOINTED | "
            f"completed_events={len(completed_events)} ledger_events={len(ledger)} checkpoint={checkpoint_file or 'none'}",
            flush=True,
        )
    print(
        "HOURLY PAPER SESSION RETURNED | "
        f"session={session_day.isoformat()} completed_events={len(completed_events)} ledger_events={len(ledger)}",
        flush=True,
    )
    if minute_stream is not None:
        minute_stream.stop()
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded causal hourly Alpaca portfolio session.")
    parser.add_argument("--session-date", required=True, help="New York session date, YYYY-MM-DD")
    parser.add_argument("--history-start", required=True, help="UTC start for one-minute training history")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--lookback-scenarios", type=int, default=120)
    parser.add_argument("--min-training-scenarios", type=int, default=20)
    parser.add_argument("--max-weight", type=Decimal, default=Decimal("0.10"))
    parser.add_argument("--risk-aversion", type=float, default=10.0)
    parser.add_argument("--min-order-notional", type=Decimal, default=Decimal("1"))
    parser.add_argument("--min-weight-drift", type=Decimal, default=Decimal("0.0025"))
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument(
        "--allow-live-trading",
        action="store_true",
        help="required with --mode live; paper is the default",
    )
    args = parser.parse_args()
    if args.mode == "live" and not args.allow_live_trading:
        parser.error("--mode live requires --allow-live-trading and ALPACA_LIVE_* credentials")
    ledger = run_hourly_paper_session(
        session_day=date.fromisoformat(args.session_date),
        history_start=args.history_start,
        output_dir=args.output_dir,
        top_n=args.top_n,
        lookback_scenarios=args.lookback_scenarios,
        min_training_scenarios=args.min_training_scenarios,
        max_weight=args.max_weight,
        risk_aversion=args.risk_aversion,
        min_order_notional=args.min_order_notional,
        min_weight_drift=args.min_weight_drift,
        mode=args.mode,
        allow_live_trading=args.allow_live_trading,
    )
    print(f"Completed {len(ledger)} {args.mode}-session events")


if __name__ == "__main__":
    main()
