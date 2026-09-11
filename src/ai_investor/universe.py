from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .config import MonitorSettings
from .robinhood_mcp import MCPError, RobinhoodReadOnlyMCPClient


@dataclass(frozen=True)
class UniverseEntry:
    symbol: str
    bucket: str
    score: float = 0.0
    sector: str = ""
    investable: bool = True
    issuer: str = ""


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


def _size_rows(
    client: RobinhoodReadOnlyMCPClient,
    *,
    minimum_market_cap: int,
    maximum_market_cap: Optional[int],
    minimum_price: float,
    minimum_average_volume: int,
) -> List[Dict[str, Any]]:
    market_cap_filter = {
        "filter_type": "FILTER_TYPE_MARKET_CAP",
        "predicate": "BETWEEN" if maximum_market_cap is not None else ">=",
        "values": (
            [str(minimum_market_cap), str(maximum_market_cap - 1)]
            if maximum_market_cap is not None
            else [str(minimum_market_cap)]
        ),
    }
    filters: List[Dict[str, Any]] = [
        {
            "filter_type": "FILTER_TYPE_INSTRUMENT_TYPE",
            "predicate": "=",
            "values": ["STOCK"],
        },
        market_cap_filter,
        {
            "filter_type": "FILTER_TYPE_LAST",
            "predicate": ">=",
            "values": [str(minimum_price)],
        },
        {
            "filter_type": "FILTER_TYPE_AVERAGE_VOLUME",
            "predicate": ">=",
            "values": [str(minimum_average_volume)],
            "interval": "1d",
            "length": 30,
        },
    ]
    payload = client.call_tool(
        "preview_scan",
        {
            "filters": filters,
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
                    "values": [str(settings.small_min_price)],
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
        columns.get("Last"),
        columns.get("Average volume"),
        columns.get("Market cap"),
        columns.get("Sector"),
        columns.get("% Change"),
    )
    return {
        "symbol": str(row.get("ticker", "")).upper(),
        "sector": str(columns.get("Sector", "")),
        "issuer": str(columns.get("Name", "")),
        "last": last,
        "average_volume": average_volume,
        "dollar_volume": last * average_volume,
        "market_cap": market_cap,
        "change": change,
        "relative_volume": relative_volume,
        "completeness": sum(
            value is not None and str(value).strip() != ""
            for value in completeness_fields
        )
        / len(completeness_fields),
    }


def _percentile(values: Sequence[float], value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    below_or_equal = sum(candidate <= value for candidate in ordered)
    return below_or_equal / len(ordered)


def _rank_core(
    rows: Iterable[Dict[str, Any]],
    min_dollar_volume: float,
    bucket: str = "core",
) -> List[UniverseEntry]:
    metrics = [
        metric
        for metric in (_row_metrics(row) for row in rows)
        if metric["symbol"] and metric["dollar_volume"] >= min_dollar_volume
    ]
    liquidities = [math.log1p(metric["dollar_volume"]) for metric in metrics]
    ranked = []
    for metric in metrics:
        liquidity = _percentile(liquidities, math.log1p(metric["dollar_volume"]))
        # Size is controlled by explicit buckets, so ranking within a bucket
        # rewards execution quality rather than simply choosing its largest names.
        score = 0.75 * liquidity + 0.25 * metric["completeness"]
        ranked.append(
            UniverseEntry(
                symbol=metric["symbol"],
                bucket=bucket,
                score=round(score, 8),
                sector=metric["sector"],
                issuer=metric["issuer"],
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
                issuer=metric["issuer"],
            )
        )
    return sorted(ranked, key=lambda entry: (-entry.score, entry.symbol))


ISSUER_ALIASES = {
    "GOOG": "ALPHABET",
    "GOOGL": "ALPHABET",
    "FOX": "FOX_CORP",
    "FOXA": "FOX_CORP",
    "NWS": "NEWS_CORP",
    "NWSA": "NEWS_CORP",
    "UA": "UNDER_ARMOUR",
    "UAA": "UNDER_ARMOUR",
    "BRK.A": "BERKSHIRE_HATHAWAY",
    "BRK.B": "BERKSHIRE_HATHAWAY",
}


def _issuer_key(symbol: str, issuer: str = "") -> str:
    symbol_key = symbol.upper()
    if symbol_key in ISSUER_ALIASES:
        return ISSUER_ALIASES[symbol_key]
    cleaned = re.sub(
        r"\s+(CLASS\s+[A-Z]|COMMON STOCK|ORDINARY SHARES?)$",
        "",
        issuer.upper().strip(),
    )
    return cleaned or symbol_key


def _take_with_sector_cap(
    candidates: Iterable[UniverseEntry],
    limit: int,
    excluded: Set[str],
    sector_cap: int,
    issuer_excluded: Optional[Set[str]] = None,
    global_sector_counts: Optional[Dict[str, int]] = None,
    global_sector_cap: int = 5,
) -> List[UniverseEntry]:
    selected: List[UniverseEntry] = []
    sector_counts: Dict[str, int] = {}
    issuers = issuer_excluded if issuer_excluded is not None else set()
    global_counts = global_sector_counts if global_sector_counts is not None else {}
    for candidate in candidates:
        if candidate.symbol in excluded:
            continue
        issuer = _issuer_key(candidate.symbol, candidate.issuer)
        if issuer in issuers:
            continue
        sector = candidate.sector or "unknown"
        if sector_counts.get(sector, 0) >= sector_cap:
            continue
        if global_counts.get(sector, 0) >= global_sector_cap:
            continue
        selected.append(candidate)
        excluded.add(candidate.symbol)
        issuers.add(issuer)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        global_counts[sector] = global_counts.get(sector, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def scan_large_candidates(
    client: RobinhoodReadOnlyMCPClient,
    settings: MonitorSettings,
) -> List[UniverseEntry]:
    return _rank_core(
        _size_rows(
            client,
            minimum_market_cap=settings.large_min_market_cap,
            maximum_market_cap=None,
            minimum_price=settings.large_min_price,
            minimum_average_volume=settings.large_min_average_volume,
        ),
        settings.large_min_average_dollar_volume,
        "large",
    )


def scan_mid_candidates(
    client: RobinhoodReadOnlyMCPClient,
    settings: MonitorSettings,
) -> List[UniverseEntry]:
    return _rank_core(
        _size_rows(
            client,
            minimum_market_cap=settings.mid_min_market_cap,
            maximum_market_cap=settings.mid_max_market_cap,
            minimum_price=settings.mid_min_price,
            minimum_average_volume=settings.mid_min_average_volume,
        ),
        settings.mid_min_average_dollar_volume,
        "mid",
    )


def scan_small_candidates(
    client: RobinhoodReadOnlyMCPClient,
    settings: MonitorSettings,
) -> List[UniverseEntry]:
    return _rank_core(
        _size_rows(
            client,
            minimum_market_cap=settings.small_min_market_cap,
            maximum_market_cap=settings.small_max_market_cap,
            minimum_price=settings.small_min_price,
            minimum_average_volume=settings.small_min_average_volume,
        ),
        settings.small_min_average_dollar_volume,
        "small",
    )


def scan_event_candidates(
    client: RobinhoodReadOnlyMCPClient,
    settings: MonitorSettings,
) -> List[UniverseEntry]:
    return _rank_events(_event_rows(client, settings))


def assemble_universe(
    settings: MonitorSettings,
    position_symbols: Sequence[str],
    large_candidates: Sequence[UniverseEntry],
    mid_candidates: Sequence[UniverseEntry],
    small_candidates: Sequence[UniverseEntry],
    event_candidates: Sequence[UniverseEntry],
) -> List[UniverseEntry]:
    candidate_context = {
        entry.symbol: entry
        for candidates in (
            large_candidates,
            mid_candidates,
            small_candidates,
            event_candidates,
        )
        for entry in candidates
    }
    positions = []
    for symbol in position_symbols:
        context = candidate_context.get(symbol)
        positions.append(
            UniverseEntry(
                symbol=symbol,
                bucket="position",
                sector=context.sector if context else "",
                issuer=context.issuer if context else "",
            )
        )
    if len(positions) > settings.position_reserve:
        raise MCPError("Position count exceeds the configured reserve")

    excluded = {entry.symbol for entry in positions}
    issuer_excluded = {
        _issuer_key(entry.symbol, entry.issuer) for entry in positions
    }
    global_sector_counts: Dict[str, int] = {}
    for entry in positions:
        if entry.sector:
            global_sector_counts[entry.sector] = (
                global_sector_counts.get(entry.sector, 0) + 1
            )
    fixed = []
    for symbol in settings.fixed_etfs:
        if symbol not in excluded:
            fixed.append(
                UniverseEntry(symbol=symbol, bucket="fixed_etf", investable=False)
            )
            excluded.add(symbol)

    # Opportunity buckets receive slots before large caps so mega-cap names
    # cannot consume the global sector allowance by themselves.
    events = _take_with_sector_cap(
        event_candidates,
        settings.event_target,
        excluded,
        sector_cap=2,
        issuer_excluded=issuer_excluded,
        global_sector_counts=global_sector_counts,
    )
    small = _take_with_sector_cap(
        small_candidates,
        settings.small_target,
        excluded,
        sector_cap=2,
        issuer_excluded=issuer_excluded,
        global_sector_counts=global_sector_counts,
    )
    mid = _take_with_sector_cap(
        mid_candidates,
        settings.mid_target,
        excluded,
        sector_cap=3,
        issuer_excluded=issuer_excluded,
        global_sector_counts=global_sector_counts,
    )
    large = _take_with_sector_cap(
        large_candidates,
        settings.large_target,
        excluded,
        sector_cap=5,
        issuer_excluded=issuer_excluded,
        global_sector_counts=global_sector_counts,
    )

    entries = positions + fixed + large + mid + small + events
    # Fill unused position slots in round-robin bucket order. Fills retain the
    # same issuer and global-sector constraints as target allocation.
    fill_pools = (mid_candidates, small_candidates, event_candidates, large_candidates)
    while len(entries) < settings.universe_size:
        added = False
        for candidates in fill_pools:
            picked = _take_with_sector_cap(
                candidates,
                1,
                excluded,
                sector_cap=5,
                issuer_excluded=issuer_excluded,
                global_sector_counts=global_sector_counts,
            )
            if picked:
                entries.extend(picked)
                added = True
            if len(entries) >= settings.universe_size:
                break
        if not added:
            break
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
        scan_large_candidates(client, settings),
        scan_mid_candidates(client, settings),
        scan_small_candidates(client, settings),
        scan_event_candidates(client, settings),
    )


def entries_to_json(entries: Sequence[UniverseEntry]) -> List[Dict[str, Any]]:
    return [asdict(entry) for entry in entries]


def entries_from_json(raw: Sequence[Dict[str, Any]]) -> List[UniverseEntry]:
    return [UniverseEntry(**dict(item)) for item in raw]
