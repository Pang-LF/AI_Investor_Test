from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .robinhood_mcp import RobinhoodMCPClient


@dataclass(frozen=True)
class DailyBar:
    symbol: str
    begins_at: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str = "robinhood_mcp"


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_histories(
    payload: Mapping[str, Any], decision_date: str
) -> Dict[str, List[DailyBar]]:
    histories: Dict[str, List[DailyBar]] = {}
    results = ((payload.get("data") or {}).get("results") or [])
    for item in results:
        symbol = str(item.get("symbol") or "").upper()
        parsed: List[DailyBar] = []
        for raw in item.get("bars") or []:
            begins_at = str(raw.get("begins_at") or "")
            # Intraday decisions use only fully completed prior daily bars.
            if not begins_at or begins_at[:10] >= decision_date:
                continue
            if raw.get("interpolated") is True:
                continue
            close = _float(raw.get("close_price"))
            volume = _float(raw.get("volume"))
            if close <= 0 or volume < 0:
                continue
            parsed.append(
                DailyBar(
                    symbol=symbol,
                    begins_at=begins_at,
                    open=_float(raw.get("open_price")),
                    high=_float(raw.get("high_price")),
                    low=_float(raw.get("low_price")),
                    close=close,
                    volume=volume,
                )
            )
        histories[symbol] = sorted(parsed, key=lambda bar: bar.begins_at)
    return histories


class HistoricalCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, decision_date: str) -> Path:
        return self.root / f"{decision_date}.json"

    def load(
        self,
        decision_date: str,
        symbols: Sequence[str],
        minimum_bars: int,
        history_calendar_days: int,
    ) -> Dict[str, List[DailyBar]]:
        path = self._path(decision_date)
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        expected = sorted(set(symbols))
        if (
            raw.get("symbols") != expected
            or raw.get("schema_version") != 2
            or raw.get("history_calendar_days") != history_calendar_days
        ):
            return {}
        histories = {
            symbol: [DailyBar(**bar) for bar in bars]
            for symbol, bars in (raw.get("histories") or {}).items()
        }
        # Newly listed symbols legitimately have short histories. Cache the
        # complete response and let the forecasting layer exclude only those
        # symbols instead of refetching all 60 names every 15 minutes.
        if any(symbol not in histories for symbol in expected):
            return {}
        return histories

    def save(
        self,
        decision_date: str,
        symbols: Sequence[str],
        histories: Mapping[str, Sequence[DailyBar]],
        history_calendar_days: int,
    ) -> None:
        path = self._path(decision_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "decision_date": decision_date,
            "history_calendar_days": history_calendar_days,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "symbols": sorted(set(symbols)),
            "histories": {
                symbol: [asdict(bar) for bar in bars]
                for symbol, bars in histories.items()
            },
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        temporary.replace(path)


def get_daily_histories(
    client: RobinhoodMCPClient,
    cache: HistoricalCache,
    symbols: Sequence[str],
    decision_date: str,
    history_calendar_days: int,
    minimum_bars: int,
) -> Dict[str, List[DailyBar]]:
    unique = sorted(set(str(symbol).upper() for symbol in symbols))
    cached = cache.load(
        decision_date, unique, minimum_bars, history_calendar_days
    )
    if cached:
        return cached

    # Anchor the request to the decision date. This keeps replays point-in-time
    # and prevents a replay from silently requesting a window ending "now".
    end = datetime.fromisoformat(decision_date).replace(tzinfo=timezone.utc) + timedelta(days=1)
    start = end - timedelta(days=history_calendar_days)
    histories: Dict[str, List[DailyBar]] = {}
    for index in range(0, len(unique), 10):
        batch = unique[index : index + 10]
        payload = client.call_tool(
            "get_equity_historicals",
            {
                "symbols": batch,
                "start_time": start.isoformat().replace("+00:00", "Z"),
                "end_time": end.isoformat().replace("+00:00", "Z"),
                "interval": "day",
                "bounds": "regular",
                "adjustment_type": "split",
            },
        )
        histories.update(_parse_histories(payload, decision_date))
    cache.save(decision_date, unique, histories, history_calendar_days)
    return histories
