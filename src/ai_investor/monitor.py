from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, time as clock_time, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from zoneinfo import ZoneInfo

from .config import Settings
from .robinhood_mcp import RobinhoodReadOnlyMCPClient
from .universe import (
    UniverseEntry,
    build_universe,
    entries_from_json,
    entries_to_json,
    resolve_agentic_account,
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

    def load(
        self,
        trading_date: str,
        refresh_bucket: str,
        position_symbols: Sequence[str],
    ) -> Optional[List[UniverseEntry]]:
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if raw.get("trading_date") != trading_date:
            return None
        if raw.get("refresh_bucket") != refresh_bucket:
            return None
        cached_positions = sorted(
            entry["symbol"]
            for entry in raw.get("entries", [])
            if entry.get("bucket") == "position"
        )
        if cached_positions != sorted(position_symbols):
            return None
        entries = entries_from_json(raw.get("entries", []))
        return entries or None

    def save(
        self,
        trading_date: str,
        refresh_bucket: str,
        entries: Sequence[UniverseEntry],
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trading_date": trading_date,
            "refresh_bucket": refresh_bucket,
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
        entries = cache.load(
            trading_date, refresh_bucket, account.position_symbols
        )
        if entries is None:
            entries = build_universe(
                client, settings.monitor, account.position_symbols
            )
            cache.save(trading_date, refresh_bucket, entries)

        symbols = [entry.symbol for entry in entries]
        batches = quote_batches(symbols, settings.monitor.quote_batch_size)
        if len(batches) != 3:
            raise RuntimeError("Safety check failed: quote cycle must use three batches")

        records: List[Dict[str, Any]] = []
        for index, batch in enumerate(batches):
            payload = client.call_tool("get_equity_quotes", {"symbols": batch})
            records.extend(_quote_record(item) for item in _quote_rows(payload))
            if (
                not no_delay
                and index < len(batches) - 1
                and settings.monitor.quote_batch_delay_seconds > 0
            ):
                time.sleep(settings.monitor.quote_batch_delay_seconds)

        returned = {str(record.get("symbol") or "") for record in records}
        missing = tuple(sorted(set(symbols) - returned))
        trigger_rows = _triggers(records, entries, settings)
        log_path = root / "logs" / "market" / f"{trading_date}.jsonl"
        _append_log(
            log_path,
            {
                "timestamp": timestamp,
                "mode": settings.mode,
                "account": account.masked_account,
                "monitor_interval_minutes": settings.monitor.interval_minutes,
                "universe": entries_to_json(entries),
                "quotes": records,
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
