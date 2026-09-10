from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from .config import MonitorSettings
from .robinhood_mcp import MCPError, RobinhoodReadOnlyMCPClient


@dataclass(frozen=True)
class UniverseEntry:
    symbol: str
    bucket: str
    score: float = 0.0
    sector: str = ""


@dataclass(frozen=True)
class AccountContext:
    account_number: str
    masked_account: str
    position_symbols: Tuple[str, ...]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = ((payload.get("data") or {}).get("result") or {})
    return [dict(row) for row in (result.get("results") or []) if row]


def resolve_agentic_account(
    client: RobinhoodReadOnlyMCPClient,
    max_positions: int,
) -> AccountContext:
    accounts_payload = client.call_tool("get_accounts", {})
    accounts = ((accounts_payload.get("data") or {}).get("accounts") or [])
    eligible = [
        account
        for account in accounts
        if account
        and account.get("agentic_allowed") is True
        and account.get("state") == "active"
        and not account.get("deactivated")
        and not account.get("permanently_deactivated")
    ]
    if len(eligible) != 1:
        raise MCPError(
            "Expected exactly one active agentic-allowed account; "
            f"Robinhood returned {len(eligible)}"
        )
    account_number = str(eligible[0]["account_number"])
    positions_payload = client.call_tool(
        "get_equity_positions", {"account_number": account_number}
    )
    positions = ((positions_payload.get("data") or {}).get("positions") or [])
    symbols = tuple(
        sorted(
            {
                str(position.get("symbol", "")).upper()
                for position in positions
                if position
                and str(position.get("symbol", "")).strip()
                and _number(position.get("quantity")) != 0
            }
        )
    )
    if len(symbols) > max_positions:
        raise MCPError(
            f"Hard risk check failed: {len(symbols)} positions exceeds {max_positions}"
        )
    return AccountContext(
        account_number=account_number,
        masked_account="••••" + account_number[-4:],
        position_symbols=symbols,
    )


def _common_columns() -> List[Dict[str, Any]]:
    return [
        {"display_name": name, "visible": True}
        for name in (
            "Market cap",
            "Sector",
            "Volume",
            "Average volume",
            "Relative volume",
            "% Change",
        )
    ]


def _core_rows(
    client: RobinhoodReadOnlyMCPClient, settings: MonitorSettings
) -> List[Dict[str, Any]]:
    payload = client.call_tool(
        "preview_scan",
        {
            "filters": [
                {
                    "filter_type": "FILTER_TYPE_INSTRUMENT_TYPE",
                    "predicate": "=",
                    "values": ["STOCK"],
                },
                {
                    "filter_type": "FILTER_TYPE_MARKET_CAP",
                    "predicate": ">=",
                    "values": [str(settings.core_min_market_cap)],
                },
                {
                    "filter_type": "FILTER_TYPE_LAST",
                    "predicate": ">=",
                    "values": [str(settings.core_min_price)],
                },
                {
                    "filter_type": "FILTER_TYPE_AVERAGE_VOLUME",
                    "predicate": ">=",
                    "values": [str(settings.core_min_average_volume)],
                    "interval": "1d",
                    "length": 30,
                },
            ],
            "columns": _common_columns(),
        },
    )
    return _rows(payload)


def _event_rows(
    client: RobinhoodReadOnlyMCPClient, settings: MonitorSettings
) -> List[Dict[str, Any]]:
    change = settings.event_min_absolute_change
    payload = client.call_tool(
        "preview_scan",
        {
            "filters": [
                {
                    "filter_type": "FILTER_TYPE_INSTRUMENT_TYPE",
                    "predicate": "=",
                    "values": ["STOCK"],
                },
                {
                    "filter_type": "FILTER_TYPE_MARKET_CAP",
                    "predicate": ">=",
                    "values": [str(settings.event_min_market_cap)],
                },
                {
                    "filter_type": "FILTER_TYPE_LAST",
                    "predicate": ">=",
                    "values": [str(settings.core_min_price)],
                },
                {
                    "filter_type": "FILTER_TYPE_AVERAGE_VOLUME",
                    "predicate": ">=",
                    "values": [str(settings.event_min_average_volume)],
                    "interval": "1d",
                    "length": 30,
                },
                {
                    "filter_type": "FILTER_TYPE_RELATIVE_VOLUME",
                    "predicate": ">=",
                    "values": [str(settings.event_min_relative_volume)],
                    "interval": "1d",
                    "length": 30,
                },
                {
                    "filter_type": "FILTER_TYPE_PERCENT_CHANGE_FROM_CLOSE",
                    "predicate": "OUTSIDE",
                    "values": [str(-change), str(change)],
                    "interval": "1d",
                    "plot": "Close",
                },
            ],
            "columns": _common_columns(),
        },
    )
    return _rows(payload)


def _row_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    columns = dict(row.get("columns") or {})
    last = _number(columns.get("Last"))
    average_volume = _number(columns.get("Average volume"))
    market_cap = _number(columns.get("Market cap"))
    change = _number(columns.get("% Change"))
    relative_volume = _number(columns.get("Relative volume"))
    completeness_fields = (
        last,
        average_volume,
        market_cap,
        columns.get("Sector"),
        columns.get("% Change"),
    )
    return {
        "symbol": str(row.get("ticker", "")).upper(),
        "sector": str(columns.get("Sector", "")),
        "last": last,
        "average_volume": average_volume,
        "dollar_volume": last * average_volume,
        "market_cap": market_cap,
        "change": change,
        "relative_volume": relative_volume,
        "completeness": sum(bool(value) for value in completeness_fields)
        / len(completeness_fields),
    }


def _percentile(values: Sequence[float], value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    below_or_equal = sum(candidate <= value for candidate in ordered)
    return below_or_equal / len(ordered)


def _rank_core(
    rows: Iterable[Dict[str, Any]], min_dollar_volume: float
) -> List[UniverseEntry]:
    metrics = [
        metric
        for metric in (_row_metrics(row) for row in rows)
        if metric["symbol"] and metric["dollar_volume"] >= min_dollar_volume
    ]
    liquidities = [math.log1p(metric["dollar_volume"]) for metric in metrics]
    sizes = [math.log1p(metric["market_cap"]) for metric in metrics]
    ranked = []
    for metric in metrics:
        liquidity = _percentile(liquidities, math.log1p(metric["dollar_volume"]))
        size = _percentile(sizes, math.log1p(metric["market_cap"]))
        opportunity = min(abs(metric["change"]) / 0.05, 1.0)
        score = (
            0.50 * liquidity
            + 0.25 * size
            + 0.15 * opportunity
            + 0.10 * metric["completeness"]
        )
        ranked.append(
            UniverseEntry(
                symbol=metric["symbol"],
                bucket="core",
                score=round(score, 8),
                sector=metric["sector"],
            )
        )
    return sorted(ranked, key=lambda entry: (-entry.score, entry.symbol))


def _rank_events(rows: Iterable[Dict[str, Any]]) -> List[UniverseEntry]:
    ranked = []
    for row in rows:
        metric = _row_metrics(row)
        if not metric["symbol"]:
            continue
        score = (
            abs(metric["change"])
            * max(metric["relative_volume"], 0.0)
            * math.log1p(max(metric["dollar_volume"], 0.0))
        )
        ranked.append(
            UniverseEntry(
                symbol=metric["symbol"],
                bucket="event",
                score=round(score, 8),
                sector=metric["sector"],
            )
        )
    return sorted(ranked, key=lambda entry: (-entry.score, entry.symbol))


def _take_with_sector_cap(
    candidates: Iterable[UniverseEntry],
    limit: int,
    excluded: Set[str],
    sector_cap: int,
) -> List[UniverseEntry]:
    selected: List[UniverseEntry] = []
    sector_counts: Dict[str, int] = {}
    for candidate in candidates:
        if candidate.symbol in excluded:
            continue
        sector = candidate.sector or "unknown"
        if sector_counts.get(sector, 0) >= sector_cap:
            continue
        selected.append(candidate)
        excluded.add(candidate.symbol)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def scan_core_candidates(
    client: RobinhoodReadOnlyMCPClient,
    settings: MonitorSettings,
) -> List[UniverseEntry]:
    return _rank_core(
        _core_rows(client, settings), settings.core_min_average_dollar_volume
    )


def scan_event_candidates(
    client: RobinhoodReadOnlyMCPClient,
    settings: MonitorSettings,
) -> List[UniverseEntry]:
    return _rank_events(_event_rows(client, settings))


def assemble_universe(
    settings: MonitorSettings,
    position_symbols: Sequence[str],
    core_candidates: Sequence[UniverseEntry],
    event_candidates: Sequence[UniverseEntry],
) -> List[UniverseEntry]:
    positions = [
        UniverseEntry(symbol=symbol, bucket="position")
        for symbol in position_symbols
    ]
    if len(positions) > settings.position_reserve:
        raise MCPError("Position count exceeds the configured reserve")

    excluded = {entry.symbol for entry in positions}
    fixed = []
    for symbol in settings.fixed_etfs:
        if symbol not in excluded:
            fixed.append(UniverseEntry(symbol=symbol, bucket="fixed_etf"))
            excluded.add(symbol)

    core = _take_with_sector_cap(
        core_candidates,
        settings.core_target,
        excluded,
        sector_cap=5,
    )
    events = _take_with_sector_cap(
        event_candidates,
        settings.event_target,
        excluded,
        sector_cap=3,
    )

    entries = positions + fixed + core + events
    # Unused position reserve and quiet event periods are filled from the core
    # ranking while preserving uniqueness. This keeps every quote cycle at 60.
    for candidate in core_candidates:
        if len(entries) >= settings.universe_size:
            break
        if candidate.symbol in excluded:
            continue
        entries.append(candidate)
        excluded.add(candidate.symbol)
    if len(entries) != settings.universe_size:
        raise MCPError(
            f"Universe construction produced {len(entries)} symbols, expected "
            f"{settings.universe_size}"
        )
    return entries


def build_universe(
    client: RobinhoodReadOnlyMCPClient,
    settings: MonitorSettings,
    position_symbols: Sequence[str],
) -> List[UniverseEntry]:
    return assemble_universe(
        settings,
        position_symbols,
        scan_core_candidates(client, settings),
        scan_event_candidates(client, settings),
    )


def entries_to_json(entries: Sequence[UniverseEntry]) -> List[Dict[str, Any]]:
    return [asdict(entry) for entry in entries]


def entries_from_json(raw: Sequence[Dict[str, Any]]) -> List[UniverseEntry]:
    return [UniverseEntry(**dict(item)) for item in raw]
