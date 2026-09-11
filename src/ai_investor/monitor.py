from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, time as clock_time, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence
from zoneinfo import ZoneInfo

from .config import Settings
from .robinhood_mcp import RobinhoodReadOnlyMCPClient
from .universe import (
    UniverseEntry,
    assemble_universe,
    entries_from_json,
    entries_to_json,
    resolve_agentic_account,
    scan_event_candidates,
    scan_large_candidates,
    scan_mid_candidates,
    scan_small_candidates,
)


@dataclass(frozen=True)
class MonitorResult:
    status: str
    timestamp: str
    universe_size: int = 0
    quote_count: int = 0
    missing_symbols: tuple = ()
    triggers: tuple = ()
    mcp_tool_calls: int = 0
    log_path: str = ""
    llm_calls: int = 0


def quote_batches(symbols: Sequence[str], batch_size: int) -> List[List[str]]:
    return [
        list(symbols[index : index + batch_size])
        for index in range(0, len(symbols), batch_size)
    ]


def is_regular_market_window(now: datetime, timezone_name: str) -> bool:
    local = now.astimezone(ZoneInfo(timezone_name))
    return (
        local.weekday() < 5
        and clock_time(9, 30) <= local.time().replace(tzinfo=None) < clock_time(16, 0)
    )


def _refresh_bucket(now: datetime, timezone_name: str) -> str:
    local = now.astimezone(ZoneInfo(timezone_name))
    return "afternoon" if local.time().replace(tzinfo=None) >= clock_time(13, 0) else "morning"


class UniverseCache:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self, trading_date: str) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if raw.get("trading_date") != trading_date:
            return None
        if raw.get("cache_version") != 3:
            return None
        large_candidates = entries_from_json(raw.get("large_candidates", []))
        mid_candidates = entries_from_json(raw.get("mid_candidates", []))
        small_candidates = entries_from_json(raw.get("small_candidates", []))
        event_candidates = entries_from_json(raw.get("event_candidates", []))
        if (
            not all((large_candidates, mid_candidates, small_candidates))
            or "event_candidates" not in raw
        ):
            return None
        return {
            "event_bucket": str(raw.get("event_bucket", "")),
            "large_candidates": large_candidates,
            "mid_candidates": mid_candidates,
            "small_candidates": small_candidates,
            "event_candidates": event_candidates,
        }

    def save(
        self,
        trading_date: str,
        event_bucket: str,
        large_candidates: Sequence[UniverseEntry],
        mid_candidates: Sequence[UniverseEntry],
        small_candidates: Sequence[UniverseEntry],
        event_candidates: Sequence[UniverseEntry],
        entries: Sequence[UniverseEntry],
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_version": 3,
            "trading_date": trading_date,
            "event_bucket": event_bucket,
            "large_candidates": entries_to_json(large_candidates),
            "mid_candidates": entries_to_json(mid_candidates),
            "small_candidates": entries_to_json(small_candidates),
            "event_candidates": entries_to_json(event_candidates),
            "entries": entries_to_json(entries),
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)


def _quote_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        dict(item)
        for item in ((payload.get("data") or {}).get("results") or [])
        if item and item.get("quote")
    ]


def _quote_record(item: Dict[str, Any]) -> Dict[str, Any]:
    quote = dict(item.get("quote") or {})
    close = dict(item.get("close") or {})
    return {
        "symbol": quote.get("symbol"),
        "last_trade_price": quote.get("last_trade_price"),
        "last_trade_time": quote.get("venue_last_trade_time"),
        "bid_price": quote.get("bid_price"),
        "ask_price": quote.get("ask_price"),
        "adjusted_previous_close": quote.get("adjusted_previous_close"),
        "official_close": close.get("price"),
        "state": quote.get("state"),
        "has_traded": quote.get("has_traded"),
    }


def quotes_are_fresh(
    records: Sequence[Dict[str, Any]],
    now: datetime,
    max_age_minutes: int,
) -> bool:
    if not records:
        return False
    timestamps = []
    for record in records:
        raw = record.get("last_trade_time")
        if not raw:
            return False
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        timestamps.append(parsed.astimezone(timezone.utc))
    current = now.astimezone(timezone.utc)
    return all(
        -300 <= (current - timestamp).total_seconds() <= max_age_minutes * 60
        for timestamp in timestamps
    )


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _triggers(
    records: Iterable[Dict[str, Any]],
    entries: Sequence[UniverseEntry],
    settings: Settings,
) -> List[Dict[str, Any]]:
    buckets = {entry.symbol: entry.bucket for entry in entries}
    triggers = []
    for record in records:
        symbol = str(record.get("symbol") or "")
        last = _float(record.get("last_trade_price"))
        previous = _float(record.get("adjusted_previous_close"))
        if not symbol or last <= 0 or previous <= 0:
            continue
        change = last / previous - 1.0
        threshold = (
            settings.monitor.position_move_trigger
            if buckets.get(symbol) == "position"
            else settings.monitor.general_move_trigger
        )
        if abs(change) >= threshold:
            triggers.append(
                {
                    "type": "price_move",
                    "symbol": symbol,
                    "change_from_previous_close": round(change, 6),
                    "threshold": threshold,
                    "bucket": buckets.get(symbol, "unknown"),
                }
            )
    return sorted(
        triggers,
        key=lambda item: abs(item["change_from_previous_close"]),
        reverse=True,
    )


def _append_log(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _previous_quotes(path: Path) -> Dict[str, Dict[str, Any]]:
    """Return the most recent logged quote set before the current cycle."""
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        quotes = payload.get("quotes") or []
        result = {
            str(item.get("symbol") or "").upper(): dict(item)
            for item in quotes
            if item.get("symbol")
        }
        if result:
            return result
    return {}


def market_summary(
    records: Sequence[Dict[str, Any]],
    previous: Dict[str, Dict[str, Any]],
    fixed_etfs: Sequence[str],
) -> Dict[str, Any]:
    """Derive cheap cross-sectional and interval features from quote snapshots."""
    daily_changes: Dict[str, float] = {}
    spreads: List[float] = []
    interval_moves: List[Dict[str, Any]] = []
    for record in records:
        symbol = str(record.get("symbol") or "").upper()
        last = _float(record.get("last_trade_price"))
        prior_close = _float(record.get("adjusted_previous_close"))
        bid = _float(record.get("bid_price"))
        ask = _float(record.get("ask_price"))
        if symbol and last > 0 and prior_close > 0:
            daily_changes[symbol] = last / prior_close - 1.0
        midpoint = (bid + ask) / 2.0
        if bid > 0 and ask >= bid and midpoint > 0:
            spreads.append((ask - bid) / midpoint)
        prior = previous.get(symbol) or {}
        prior_last = _float(prior.get("last_trade_price"))
        if symbol and last > 0 and prior_last > 0:
            interval_moves.append(
                {
                    "symbol": symbol,
                    "return": round(last / prior_last - 1.0, 6),
                }
            )

    changes = list(daily_changes.values())
    advancers = sum(change > 0 for change in changes)
    decliners = sum(change < 0 for change in changes)
    unchanged = len(changes) - advancers - decliners
    mean_change = fmean(changes) if changes else 0.0
    dispersion = (
        math.sqrt(fmean((change - mean_change) ** 2 for change in changes))
        if changes
        else 0.0
    )
    interval_moves.sort(key=lambda item: abs(item["return"]), reverse=True)
    return {
        "symbols_with_valid_change": len(changes),
        "advancers": advancers,
        "decliners": decliners,
        "unchanged": unchanged,
        "positive_breadth": round(advancers / len(changes), 6) if changes else 0.0,
        "median_change_from_previous_close": round(median(changes), 6) if changes else 0.0,
        "cross_sectional_dispersion": round(dispersion, 6),
        "median_spread_fraction": round(median(spreads), 6) if spreads else None,
        "maximum_spread_fraction": round(max(spreads), 6) if spreads else None,
        "fixed_etf_changes": {
            symbol: round(daily_changes[symbol], 6)
            for symbol in fixed_etfs
            if symbol in daily_changes
        },
        "largest_interval_moves": interval_moves[:10],
    }


def run_monitor_cycle(
    settings: Settings,
    root: Path,
    force: bool = False,
    no_delay: bool = False,
    now: Optional[datetime] = None,
) -> MonitorResult:
    current = now or datetime.now(timezone.utc)
    timestamp = current.isoformat()
    if not force and not is_regular_market_window(
        current, settings.monitor.market_timezone
    ):
        return MonitorResult(status="skipped_outside_regular_market", timestamp=timestamp)

    local = current.astimezone(ZoneInfo(settings.monitor.market_timezone))
    trading_date = local.date().isoformat()
    refresh_bucket = _refresh_bucket(current, settings.monitor.market_timezone)
    cache = UniverseCache(root / ".local" / "state" / "universe.json")
    with RobinhoodReadOnlyMCPClient(
        max_calls=settings.monitor.max_mcp_calls_per_cycle
    ) as client:
        account = resolve_agentic_account(
            client, max_positions=settings.monitor.position_reserve
        )
        cached = cache.load(trading_date)
        if cached is None:
            large_candidates = scan_large_candidates(client, settings.monitor)
            mid_candidates = scan_mid_candidates(client, settings.monitor)
            small_candidates = scan_small_candidates(client, settings.monitor)
            event_candidates = scan_event_candidates(client, settings.monitor)
        else:
            large_candidates = cached["large_candidates"]
            mid_candidates = cached["mid_candidates"]
            small_candidates = cached["small_candidates"]
            event_candidates = cached["event_candidates"]
            if cached["event_bucket"] != refresh_bucket:
                event_candidates = scan_event_candidates(client, settings.monitor)
        entries = assemble_universe(
            settings.monitor,
            account.position_symbols,
            large_candidates,
            mid_candidates,
            small_candidates,
            event_candidates,
        )
        cache.save(
            trading_date,
            refresh_bucket,
            large_candidates,
            mid_candidates,
            small_candidates,
            event_candidates,
            entries,
        )

        symbols = [entry.symbol for entry in entries]
        batches = quote_batches(symbols, settings.monitor.quote_batch_size)
        if len(batches) != 3:
            raise RuntimeError("Safety check failed: quote cycle must use three batches")

        records: List[Dict[str, Any]] = []
        for index, batch in enumerate(batches):
            payload = client.call_tool("get_equity_quotes", {"symbols": batch})
            records.extend(_quote_record(item) for item in _quote_rows(payload))
            if (
                index == 0
                and not force
                and not quotes_are_fresh(
                    records,
                    current,
                    settings.monitor.max_quote_age_minutes,
                )
            ):
                returned = {
                    str(record.get("symbol") or "") for record in records
                }
                missing = tuple(sorted(set(symbols) - returned))
                log_path = root / "logs" / "market" / f"{trading_date}.jsonl"
                _append_log(
                    log_path,
                    {
                        "timestamp": timestamp,
                        "mode": settings.mode,
                        "account": account.masked_account,
                        "status": "skipped_stale_market_data",
                        "quotes": records,
                        "missing_symbols": list(missing),
                        "triggers": [],
                        "mcp_tool_calls": client.call_count,
                        "llm_calls": 0,
                    },
                )
                return MonitorResult(
                    status="skipped_stale_market_data",
                    timestamp=timestamp,
                    universe_size=len(entries),
                    quote_count=len(records),
                    missing_symbols=missing,
                    mcp_tool_calls=client.call_count,
                    log_path=str(log_path),
                    llm_calls=0,
                )
            if (
                not no_delay
                and index < len(batches) - 1
                and settings.monitor.quote_batch_delay_seconds > 0
            ):
                time.sleep(settings.monitor.quote_batch_delay_seconds)

        returned = {str(record.get("symbol") or "") for record in records}
        missing = tuple(sorted(set(symbols) - returned))
        if missing or not quotes_are_fresh(
            records, current, settings.monitor.max_quote_age_minutes
        ):
            status = "skipped_incomplete_market_data" if missing else "skipped_stale_market_data"
            log_path = root / "logs" / "market" / f"{trading_date}.jsonl"
            _append_log(
                log_path,
                {
                    "timestamp": timestamp,
                    "mode": settings.mode,
                    "account": account.masked_account,
                    "status": status,
                    "quotes": records,
                    "missing_symbols": list(missing),
                    "triggers": [],
                    "mcp_tool_calls": client.call_count,
                    "llm_calls": 0,
                },
            )
            return MonitorResult(
                status=status,
                timestamp=timestamp,
                universe_size=len(entries),
                quote_count=len(records),
                missing_symbols=missing,
                mcp_tool_calls=client.call_count,
                log_path=str(log_path),
                llm_calls=0,
            )
        log_path = root / "logs" / "market" / f"{trading_date}.jsonl"
        trigger_rows = _triggers(records, entries, settings)
        summary = market_summary(
            records,
            _previous_quotes(log_path),
            settings.monitor.fixed_etfs,
        )
        _append_log(
            log_path,
            {
                "timestamp": timestamp,
                "mode": settings.mode,
                "account": account.masked_account,
                "monitor_interval_minutes": settings.monitor.interval_minutes,
                "universe": entries_to_json(entries),
                "quotes": records,
                "market_summary": summary,
                "missing_symbols": list(missing),
                "triggers": trigger_rows,
                "mcp_tool_calls": client.call_count,
                "llm_calls": 0,
            },
        )
        return MonitorResult(
            status="completed_read_only",
            timestamp=timestamp,
            universe_size=len(entries),
            quote_count=len(records),
            missing_symbols=missing,
            triggers=tuple(trigger_rows),
            mcp_tool_calls=client.call_count,
            log_path=str(log_path),
            llm_calls=0,
        )
