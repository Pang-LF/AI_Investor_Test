from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .config import MonitorSettings
from .robinhood_mcp import MCPError, RobinhoodMCPClient


SCANNER_KEYS = ("large", "mid", "small", "event")


@dataclass(frozen=True)
class ScannerDefinition:
    key: str
    title_prefix: str
    filters: tuple[Dict[str, Any], ...]
    columns: tuple[Dict[str, Any], ...]

    @property
    def fingerprint(self) -> str:
        payload = {"filters": self.filters, "columns": self.columns}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def title(self) -> str:
        return f"{self.title_prefix} {self.fingerprint[:8]}"

    def create_arguments(self) -> Dict[str, Any]:
        return {
            "preset": "INITIAL",
            "title": self.title,
            "filters": list(self.filters),
            "columns": list(self.columns),
        }


@dataclass(frozen=True)
class ScannerRecord:
    scan_id: str
    title: str
    fingerprint: str
    server_fingerprint: str


def _server_fingerprint(scan: Mapping[str, Any]) -> str:
    payload = {
        "filter_summary": scan.get("filter_summary") or [],
        "columns": scan.get("columns") or [],
        "sorting": scan.get("sorting") or "",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _common_columns() -> list[Dict[str, Any]]:
    columns = [
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
    columns.extend(
        [
            {
                "display_name": "Float shares",
                "expression": "fundamental.sharesFloat",
                "visible": True,
            },
            {
                "display_name": "Float ratio",
                "expression": (
                    "fundamental.sharesFloat / fundamental.sharesOutstanding"
                ),
                "visible": True,
            },
            {
                "display_name": "IPO age days",
                "expression": "daysFromNow(fundamental.initialPublicOfferingYmd)",
                "visible": True,
            },
            {
                "display_name": "Listing venue",
                "expression": "officialPlaceOfListing",
                "visible": True,
            },
            {
                "display_name": "Trading status",
                "expression": "tradingStatus",
                "visible": True,
            },
        ]
    )
    return columns


def _size_definition(
    key: str,
    title: str,
    *,
    minimum_market_cap: int,
    maximum_market_cap: Optional[int],
    minimum_price: float,
    minimum_average_volume: int,
    minimum_ipo_age_calendar_days: int,
    minimum_float_ratio: Optional[float] = None,
) -> ScannerDefinition:
    market_cap_filter = {
        "filter_type": "FILTER_TYPE_MARKET_CAP",
        "predicate": "BETWEEN" if maximum_market_cap is not None else ">=",
        "values": (
            [str(minimum_market_cap), str(maximum_market_cap - 1)]
            if maximum_market_cap is not None
            else [str(minimum_market_cap)]
        ),
    }
    filters: list[Dict[str, Any]] = [
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
        {
            "expression": "daysFromNow(fundamental.initialPublicOfferingYmd)",
            "predicate": "<=",
            "values": [str(-minimum_ipo_age_calendar_days)],
            "display_title": "IPO age days",
        },
    ]
    if minimum_float_ratio is not None:
        filters.append(
            {
                "expression": (
                    "fundamental.sharesFloat / fundamental.sharesOutstanding"
                ),
                "predicate": ">=",
                "values": [str(minimum_float_ratio)],
                "display_title": "Float ratio",
            }
        )
    duplicate_columns = {"IPO age days"}
    if minimum_float_ratio is not None:
        duplicate_columns.add("Float ratio")
    columns = [
        column
        for column in _common_columns()
        if column["display_name"] not in duplicate_columns
    ]
    return ScannerDefinition(key, title, tuple(filters), tuple(columns))


def scanner_definitions(settings: MonitorSettings) -> Dict[str, ScannerDefinition]:
    definitions = {
        "large": _size_definition(
            "large",
            "AI Investor Large",
            minimum_market_cap=settings.large_min_market_cap,
            maximum_market_cap=None,
            minimum_price=settings.large_min_price,
            minimum_average_volume=settings.large_min_average_volume,
            minimum_ipo_age_calendar_days=settings.stable_min_ipo_age_calendar_days,
        ),
        "mid": _size_definition(
            "mid",
            "AI Investor Mid",
            minimum_market_cap=settings.mid_min_market_cap,
            maximum_market_cap=settings.mid_max_market_cap,
            minimum_price=settings.mid_min_price,
            minimum_average_volume=settings.mid_min_average_volume,
            minimum_ipo_age_calendar_days=settings.stable_min_ipo_age_calendar_days,
        ),
        "small": _size_definition(
            "small",
            "AI Investor Small",
            minimum_market_cap=settings.small_min_market_cap,
            maximum_market_cap=settings.small_max_market_cap,
            minimum_price=settings.small_min_price,
            minimum_average_volume=settings.small_min_average_volume,
            minimum_ipo_age_calendar_days=settings.stable_min_ipo_age_calendar_days,
            minimum_float_ratio=settings.small_min_float_ratio,
        ),
    }
    change = settings.event_min_absolute_change
    definitions["event"] = ScannerDefinition(
        "event",
        "AI Investor Event",
        tuple(
            [
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
            ]
        ),
        tuple(_common_columns()),
    )
    return definitions


class ScannerRegistry:
    def __init__(self, records: Mapping[str, ScannerRecord]) -> None:
        self.records = dict(records)

    def scan_id(self, key: str) -> str:
        if key not in self.records:
            raise MCPError(f"Scanner registry is missing {key}")
        return self.records[key].scan_id

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "scans": {key: asdict(value) for key, value in self.records.items()},
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(
        cls, path: Path, definitions: Mapping[str, ScannerDefinition]
    ) -> "ScannerRegistry":
        if not path.exists():
            raise MCPError("Saved scanner registry is missing; run setup-scanners")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MCPError("Saved scanner registry is unreadable") from exc
        if raw.get("schema_version") != 2:
            raise MCPError("Saved scanner registry schema is unsupported")
        records = {
            key: ScannerRecord(**dict(value))
            for key, value in (raw.get("scans") or {}).items()
        }
        if set(records) != set(SCANNER_KEYS):
            raise MCPError("Saved scanner registry is incomplete")
        for key, definition in definitions.items():
            record = records[key]
            if (
                record.fingerprint != definition.fingerprint
                or record.title != definition.title
            ):
                raise MCPError(
                    f"Saved {key} scanner does not match current configuration; "
                    "run setup-scanners explicitly"
                )
        return cls(records)

    def validate_remote(self, payload: Mapping[str, Any]) -> None:
        scans = (payload.get("data") or {}).get("scans") or []
        by_id = {
            str(item.get("scan_id")): item
            for item in scans
            if item.get("scan_id")
        }
        for key, record in self.records.items():
            remote = by_id.get(record.scan_id)
            if remote is None:
                raise MCPError(f"Saved {key} scanner no longer exists")
            if str(remote.get("title") or "") != record.title:
                raise MCPError(f"Saved {key} scanner title changed")
            if remote.get("cortex_managed") is True:
                raise MCPError(f"Saved {key} scanner became externally managed")
            if _server_fingerprint(remote) != record.server_fingerprint:
                raise MCPError(f"Saved {key} scanner configuration changed")


def provision_scanners(
    client: RobinhoodMCPClient,
    settings: MonitorSettings,
    path: Path,
) -> ScannerRegistry:
    definitions = scanner_definitions(settings)
    listed = client.call_tool("get_scans", {})
    existing = (listed.get("data") or {}).get("scans") or []
    existing_by_title = {
        str(item.get("title")): str(item.get("scan_id"))
        for item in existing
        if item.get("title") and item.get("scan_id")
    }
    scan_ids: Dict[str, str] = {}
    for key in SCANNER_KEYS:
        definition = definitions[key]
        scan_id = existing_by_title.get(definition.title, "")
        if not scan_id:
            created = client.call_tool("create_scan", definition.create_arguments())
            result = (created.get("data") or {}).get("result") or {}
            scan_id = str(result.get("scan_id") or "")
            if not scan_id:
                raise MCPError(f"Robinhood did not return a scan_id for {key}")
            if str(result.get("scan_title") or "") != definition.title:
                raise MCPError(f"Robinhood returned an unexpected title for {key}")
            if len(result.get("filters_applied") or []) != len(definition.filters):
                raise MCPError(f"Robinhood did not apply every {key} scanner filter")
        checked = client.call_tool("run_scan", {"scan_id": scan_id})
        checked_result = (checked.get("data") or {}).get("result") or {}
        if str(checked_result.get("scan_id") or "") != scan_id:
            raise MCPError(f"Robinhood could not verify the saved {key} scanner")
        scan_ids[key] = scan_id
    refreshed = client.call_tool("get_scans", {})
    refreshed_scans = (refreshed.get("data") or {}).get("scans") or []
    refreshed_by_id = {
        str(item.get("scan_id")): item
        for item in refreshed_scans
        if item.get("scan_id")
    }
    records: Dict[str, ScannerRecord] = {}
    for key in SCANNER_KEYS:
        definition = definitions[key]
        scan_id = scan_ids[key]
        remote = refreshed_by_id.get(scan_id)
        if remote is None:
            raise MCPError(f"Robinhood did not list the saved {key} scanner")
        records[key] = ScannerRecord(
            scan_id=scan_id,
            title=definition.title,
            fingerprint=definition.fingerprint,
            server_fingerprint=_server_fingerprint(remote),
        )
    registry = ScannerRegistry(records)
    registry.validate_remote(refreshed)
    registry.save(path)
    return registry
