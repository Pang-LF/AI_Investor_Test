from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .notification import DeliveryResult, send_or_queue


def classify_operational_failure(error: BaseException) -> str:
    text = f"{type(error).__module__}.{type(error).__name__}: {error}".lower()
    if "persistent live authorization" in text or "live_trading kill switch" in text:
        return "live_authorization"
    if any(marker in text for marker in ("oauth", "access token", "refresh token", "401", "403")):
        return "robinhood_authentication"
    if any(marker in text for marker in ("robinhood", "mcp", "httpx", "requesterror")):
        return "robinhood_service"
    if any(marker in text for marker in ("openai", "llm")):
        return "openai_service"
    if any(marker in text for marker in ("smtp", "email", "notification")):
        return "email_delivery"
    return "agent_runtime"


def sanitize_error(error: object) -> str:
    text = str(error)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED_API_KEY]", text)
    text = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{8,}\b", "[REDACTED_NUMBER]", text)
    return text[:1500]


def _state_path(root: Path) -> Path:
    return root / ".local" / "state" / "operational_alerts.json"


def _load_state(root: Path) -> dict[str, Any]:
    try:
        raw = json.loads(_state_path(root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "failures": {}}
    if raw.get("schema_version") != 1 or not isinstance(raw.get("failures"), dict):
        return {"schema_version": 1, "failures": {}}
    return raw


def _save_state(root: Path, state: dict[str, Any]) -> None:
    path = _state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_log(
    root: Path,
    *,
    timestamp: datetime,
    component: str,
    error: str,
    consecutive_count: int,
    notification: str,
) -> None:
    path = root / "logs" / "health" / f"{timestamp.date().isoformat()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": timestamp.isoformat(),
        "component": component,
        "status": "failure",
        "error": error,
        "consecutive_count": consecutive_count,
        "notification": notification,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def report_operational_failure(
    root: Path,
    *,
    component: str,
    error: object,
    repeat_minutes: int,
    occurrence_threshold: int = 1,
    now: Optional[datetime] = None,
) -> DeliveryResult:
    """Log a failure and email it once at threshold, then at a bounded cadence."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    safe_error = sanitize_error(error)
    state = _load_state(root)
    failures = state["failures"]
    previous = failures.get(component) or {}
    count = int(previous.get("consecutive_count", 0)) + 1
    last_notified_raw = previous.get("last_notified_at")
    last_notified = None
    if last_notified_raw:
        try:
            last_notified = datetime.fromisoformat(str(last_notified_raw))
        except ValueError:
            last_notified = None
    due = count >= occurrence_threshold and (
        last_notified is None
        or current - last_notified >= timedelta(minutes=repeat_minutes)
    )
    delivery = DeliveryResult("suppressed", "Alert threshold or repeat interval not reached")
    if due:
        delivery = send_or_queue(
            root,
            run_id=f"health-{component}-{current.strftime('%Y%m%dT%H%M%SZ')}",
            subject=f"[AI Investor] ACTION REQUIRED: {component} failure",
            body=(
                f"Time: {current.isoformat()}\n"
                f"Component: {component}\n"
                f"Consecutive detections: {count}\n"
                f"Error: {safe_error}\n\n"
                "Trading fails closed for this condition. Check the Mac, network, "
                "Robinhood OAuth/MCP, configuration, and local logs before taking action."
            ),
        )
        last_notified_raw = current.isoformat()
    failures[component] = {
        "active": True,
        "first_seen_at": previous.get("first_seen_at") or current.isoformat(),
        "last_seen_at": current.isoformat(),
        "last_notified_at": last_notified_raw,
        "consecutive_count": count,
        "last_error": safe_error,
        "last_delivery": delivery.status,
    }
    _save_state(root, state)
    _append_log(
        root,
        timestamp=current,
        component=component,
        error=safe_error,
        consecutive_count=count,
        notification=delivery.status,
    )
    return delivery


def clear_operational_failures(root: Path, components: Iterable[str]) -> None:
    """Reset alert counters silently; recovery emails are intentionally disabled."""
    state = _load_state(root)
    changed = False
    for component in components:
        existing = state["failures"].get(component)
        if existing and (existing.get("active") or existing.get("consecutive_count")):
            existing["active"] = False
            existing["consecutive_count"] = 0
            existing["first_seen_at"] = None
            existing["last_notified_at"] = None
            changed = True
    if changed:
        _save_state(root, state)
