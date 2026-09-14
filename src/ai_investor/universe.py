from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
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
    listing_venue: str = ""
    trading_status: str = ""
    ipo_age_calendar_days: int = 0
    float_shares: float = 0.0
    float_ratio: float = 0.0
    average_dollar_volume: float = 0.0
    median_dollar_volume_20d: float = 0.0
    eligibility_flags: Tuple[str, ...] = ()
    event_type: str = ""


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


def _scan_rows(
    client: RobinhoodReadOnlyMCPClient, scan_id: str
) -> List[Dict[str, Any]]:
    return _rows(client.call_tool("run_scan", {"scan_id": scan_id}))


def _row_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    columns = dict(row.get("columns") or {})
    last = _number(columns.get("Last"))
    average_volume = _number(columns.get("Average volume"))
    market_cap = _number(columns.get("Market cap"))
    change = _number(columns.get("% Change"))
    relative_volume = _number(columns.get("Relative volume"))
    ipo_delta = int(_number(columns.get("IPO age days")))
    listing_venue = str(columns.get("Listing venue", ""))
    trading_status = str(columns.get("Trading status", ""))
    float_shares = _number(columns.get("Float shares"))
    float_ratio = _number(columns.get("Float ratio"))
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
        "ipo_age_calendar_days": abs(ipo_delta) if ipo_delta < 0 else 0,
        "listing_venue": listing_venue,
        "trading_status": trading_status,
        "float_shares": float_shares,
        "float_ratio": float_ratio,
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
        if metric["symbol"]
        and metric["dollar_volume"] >= min_dollar_volume
        and not _is_otc_venue(metric["listing_venue"])
        and _is_regular_trading_status(metric["trading_status"])
        and not _looks_like_spac(metric["issuer"])
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
                listing_venue=metric["listing_venue"],
                trading_status=metric["trading_status"],
                ipo_age_calendar_days=metric["ipo_age_calendar_days"],
                float_shares=metric["float_shares"],
                float_ratio=metric["float_ratio"],
                average_dollar_volume=metric["dollar_volume"],
            )
        )
    return sorted(ranked, key=lambda entry: (-entry.score, entry.symbol))


def _rank_events(rows: Iterable[Dict[str, Any]]) -> List[UniverseEntry]:
    ranked = []
    for row in rows:
        metric = _row_metrics(row)
        if not metric["symbol"]:
            continue
        if _is_otc_venue(metric["listing_venue"]):
            continue
        flags = []
        if not _is_regular_trading_status(metric["trading_status"]):
            flags.append("non_regular_trading_status")
        if _looks_like_spac(metric["issuer"]):
            flags.append("spac_or_blank_check")
        if 0 < metric["ipo_age_calendar_days"] < 252:
            flags.append("recent_ipo")
        score = (
            abs(metric["change"])
            * max(metric["relative_volume"], 0.0)
            * math.log1p(max(metric["dollar_volume"], 0.0))
            * (0.5 + 0.5 * metric["completeness"])
        )
        ranked.append(
            UniverseEntry(
                symbol=metric["symbol"],
                bucket="event",
                score=round(score, 8),
                sector=metric["sector"],
                issuer=metric["issuer"],
                listing_venue=metric["listing_venue"],
                trading_status=metric["trading_status"],
                ipo_age_calendar_days=metric["ipo_age_calendar_days"],
                float_shares=metric["float_shares"],
                float_ratio=metric["float_ratio"],
                average_dollar_volume=metric["dollar_volume"],
                eligibility_flags=tuple(flags),
                event_type=("breakout" if metric["change"] > 0 else "breakdown"),
            )
        )
    return sorted(ranked, key=lambda entry: (-entry.score, entry.symbol))


OTC_VENUES = frozenset({"OOTC", "OTCB", "OTCQ", "OTCM", "PINX"})


def _is_otc_venue(value: str) -> bool:
    return value.upper().strip() in OTC_VENUES


def _is_regular_trading_status(value: str) -> bool:
    # Missing status is recorded for downstream review rather than silently
    # excluding otherwise valid companies. Explicit halts/restrictions fail.
    normalized = value.lower().strip()
    return not normalized or normalized == "regular"


def _looks_like_spac(issuer: str) -> bool:
    normalized = issuer.upper()
    return bool(re.search(r"\b(SPAC|BLANK CHECK|ACQUISITION CORP(?:ORATION)?)\b", normalized))


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


def _interleave_event_directions(
    candidates: Sequence[UniverseEntry],
) -> List[UniverseEntry]:
    groups = {
        "breakout": [item for item in candidates if item.event_type == "breakout"],
        "breakdown": [item for item in candidates if item.event_type == "breakdown"],
    }
    remainder = [
        item for item in candidates if item.event_type not in groups
    ]
    interleaved: List[UniverseEntry] = []
    for index in range(max(len(groups["breakout"]), len(groups["breakdown"]))):
        for name in ("breakout", "breakdown"):
            if index < len(groups[name]):
                interleaved.append(groups[name][index])
    return interleaved + remainder


def scan_large_candidates(
    client: RobinhoodReadOnlyMCPClient,
    settings: MonitorSettings,
    scan_id: str,
) -> List[UniverseEntry]:
    return _rank_core(
        _scan_rows(client, scan_id),
        settings.large_min_average_dollar_volume,
        "large",
    )


def scan_mid_candidates(
    client: RobinhoodReadOnlyMCPClient,
    settings: MonitorSettings,
    scan_id: str,
) -> List[UniverseEntry]:
    return _rank_core(
        _scan_rows(client, scan_id),
        settings.mid_min_average_dollar_volume,
        "mid",
    )


def scan_small_candidates(
    client: RobinhoodReadOnlyMCPClient,
    settings: MonitorSettings,
    scan_id: str,
) -> List[UniverseEntry]:
    return _rank_core(
        _scan_rows(client, scan_id),
        settings.small_min_average_dollar_volume,
        "small",
    )


def scan_event_candidates(
    client: RobinhoodReadOnlyMCPClient,
    settings: MonitorSettings,
    scan_id: str,
) -> List[UniverseEntry]:
    return _rank_events(_scan_rows(client, scan_id))


def filter_small_candidates_by_median_liquidity(
    client: RobinhoodReadOnlyMCPClient,
    candidates: Sequence[UniverseEntry],
    settings: MonitorSettings,
    decision_date: str,
) -> List[UniverseEntry]:
    """Enforce 20 completed-day median dollar volume on a bounded shortlist.

    Robinhood's scanner exposes historical average turnover but no median.
    Pulling split-adjusted and raw completed daily bars for at most 20
    prefiltered names makes both the robust liquidity rule and recent reverse-
    split check auditable without expanding the recurring quote loop.
    """
    chosen = list(candidates[: settings.small_median_liquidity_candidate_limit])
    if not chosen:
        return []
    end = datetime.fromisoformat(decision_date).replace(tzinfo=timezone.utc)
    start = end - timedelta(days=120)
    median_by_symbol: Dict[str, float] = {}
    split_close_by_symbol: Dict[str, Dict[str, float]] = {}
    for index in range(0, len(chosen), 10):
        batch = chosen[index : index + 10]
        payload = client.call_tool(
            "get_equity_historicals",
            {
                "symbols": [item.symbol for item in batch],
                "start_time": start.isoformat().replace("+00:00", "Z"),
                "end_time": end.isoformat().replace("+00:00", "Z"),
                "interval": "day",
                "bounds": "regular",
                "adjustment_type": "split",
            },
        )
        results = ((payload.get("data") or {}).get("results") or [])
        for result in results:
            symbol = str(result.get("symbol") or "").upper()
            dollar_volumes = []
            split_closes: Dict[str, float] = {}
            for bar in result.get("bars") or []:
                begins_at = str(bar.get("begins_at") or "")
                if not begins_at or begins_at[:10] >= decision_date:
                    continue
                if bar.get("interpolated") is True:
                    continue
                close = _number(bar.get("close_price"))
                volume = _number(bar.get("volume"))
                if close > 0 and volume >= 0:
                    dollar_volumes.append(close * volume)
                    split_closes[begins_at[:10]] = close
            if len(dollar_volumes) >= 20:
                median_by_symbol[symbol] = median(dollar_volumes[-20:])
                split_close_by_symbol[symbol] = split_closes
    liquidity_pass = [
        candidate
        for candidate in chosen
        if median_by_symbol.get(candidate.symbol, 0.0)
        >= settings.small_min_average_dollar_volume
    ]
    raw_close_by_symbol: Dict[str, Dict[str, float]] = {}
    for index in range(0, len(liquidity_pass), 10):
        batch = liquidity_pass[index : index + 10]
        payload = client.call_tool(
            "get_equity_historicals",
            {
                "symbols": [item.symbol for item in batch],
                "start_time": start.isoformat().replace("+00:00", "Z"),
                "end_time": end.isoformat().replace("+00:00", "Z"),
                "interval": "day",
                "bounds": "regular",
                "adjustment_type": "none",
            },
        )
        for result in ((payload.get("data") or {}).get("results") or []):
            symbol = str(result.get("symbol") or "").upper()
            raw_closes = {}
            for bar in result.get("bars") or []:
                begins_at = str(bar.get("begins_at") or "")
                if not begins_at or begins_at[:10] >= decision_date:
                    continue
                close = _number(bar.get("close_price"))
                if close > 0 and bar.get("interpolated") is not True:
                    raw_closes[begins_at[:10]] = close
            raw_close_by_symbol[symbol] = raw_closes

    def recent_reverse_split(symbol: str) -> bool:
        adjusted = split_close_by_symbol.get(symbol) or {}
        raw = raw_close_by_symbol.get(symbol) or {}
        dates = sorted(set(adjusted) & set(raw))
        if len(dates) < 20:
            return True  # Incomplete corporate-action comparison fails closed.
        ratios = [raw[date] / adjusted[date] for date in dates if adjusted[date] > 0]
        return any(
            previous > 0 and current / previous >= 1.5
            for previous, current in zip(ratios, ratios[1:])
        )

    return [
        UniverseEntry(
            **{
                **asdict(candidate),
                "median_dollar_volume_20d": round(
                    median_by_symbol.get(candidate.symbol, 0.0), 2
                ),
            }
        )
        for candidate in liquidity_pass
        if not recent_reverse_split(candidate.symbol)
    ]


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
    if len(positions) > settings.universe_size - len(settings.fixed_etfs):
        raise MCPError("Positions leave no capacity for the investment universe")

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
    available_stock_slots = settings.universe_size - len(fixed) - len(positions)
    events = _take_with_sector_cap(
        _interleave_event_directions(event_candidates),
        min(settings.event_target, available_stock_slots),
        excluded,
        sector_cap=2,
        issuer_excluded=issuer_excluded,
        global_sector_counts=global_sector_counts,
    )
    small = _take_with_sector_cap(
        small_candidates,
        min(settings.small_target, max(0, available_stock_slots - len(events))),
        excluded,
        sector_cap=2,
        issuer_excluded=issuer_excluded,
        global_sector_counts=global_sector_counts,
    )
    mid = _take_with_sector_cap(
        mid_candidates,
        min(
            settings.mid_target,
            max(0, available_stock_slots - len(events) - len(small)),
        ),
        excluded,
        sector_cap=3,
        issuer_excluded=issuer_excluded,
        global_sector_counts=global_sector_counts,
    )
    large = _take_with_sector_cap(
        large_candidates,
        min(
            settings.large_target,
            max(0, available_stock_slots - len(events) - len(small) - len(mid)),
        ),
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
    scan_ids: Dict[str, str],
) -> List[UniverseEntry]:
    return assemble_universe(
        settings,
        position_symbols,
        scan_large_candidates(client, settings, scan_ids["large"]),
        scan_mid_candidates(client, settings, scan_ids["mid"]),
        scan_small_candidates(client, settings, scan_ids["small"]),
        scan_event_candidates(client, settings, scan_ids["event"]),
    )


def entries_to_json(entries: Sequence[UniverseEntry]) -> List[Dict[str, Any]]:
    return [asdict(entry) for entry in entries]


def entries_from_json(raw: Sequence[Dict[str, Any]]) -> List[UniverseEntry]:
    return [UniverseEntry(**dict(item)) for item in raw]
