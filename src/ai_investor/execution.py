from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .config import Settings
from .ledger import Ledger
from .risk_engine import OrderIntent, approve_order
from .robinhood_mcp import (
    DECISION_READ_ONLY_TOOLS,
    ORDER_REVIEW_TOOLS,
    ORDER_WRITE_TOOLS,
    RobinhoodMCPClient,
)


TERMINAL_ORDER_STATES = frozenset({"filled", "cancelled", "canceled", "rejected", "failed"})
NON_REPEATABLE_ORDER_STATES = frozenset({"submitted", "queued", "confirmed", "partially_filled", "filled"})


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    market_value: float


@dataclass(frozen=True)
class BrokerState:
    account_number: str
    portfolio_value: float
    cash: float
    option_value: float
    crypto_value: float
    futures_value: float
    positions: tuple[Position, ...]

    @property
    def positions_value(self) -> Dict[str, float]:
        return {position.symbol: position.market_value for position in self.positions}


@dataclass(frozen=True)
class PlannedOrder:
    ref_id: str
    decision_key: str
    symbol: str
    side: str
    planned_notional: float
    quantity: Optional[float]


def account_fingerprint(account_number: str) -> str:
    return hashlib.sha256(account_number.encode("utf-8")).hexdigest()


def arm_live(root: Path, account_number: str, trading_date: str) -> Path:
    requested = date.fromisoformat(trading_date)
    today = date.today()
    if requested not in {today, today + timedelta(days=1)}:
        raise RuntimeError("Live arming is valid only for today or tomorrow")
    path = root / ".local" / "state" / "live_arm.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trading_date": trading_date,
                "account_fingerprint": account_fingerprint(account_number),
                "armed_at": datetime.now(timezone.utc).isoformat(),
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def assert_live_armed(settings: Settings, root: Path, account_number: str, trading_date: str) -> None:
    if settings.mode != "LIVE" or not settings.live_trading:
        raise RuntimeError("LIVE_TRADING kill switch is off")
    path = root / ".local" / "state" / "live_arm.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Daily live arm file is absent or invalid") from exc
    if payload.get("trading_date") != trading_date:
        raise RuntimeError("Daily live arm has expired")
    if payload.get("account_fingerprint") != account_fingerprint(account_number):
        raise RuntimeError("Live arm does not match the selected account")


def deterministic_ref_id(decision_key: str, symbol: str, side: str, value: float) -> str:
    name = f"{decision_key}|{symbol.upper()}|{side}|{value:.2f}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def plan_orders(
    *,
    decision_key: str,
    target_weights: Mapping[str, float],
    state: BrokerState,
    prices: Mapping[str, float],
    minimum_trade_usd: float,
) -> list[PlannedOrder]:
    current = state.positions_value
    symbols = sorted(set(current) | set(target_weights))
    orders: list[PlannedOrder] = []
    for symbol in symbols:
        target_value = target_weights.get(symbol, 0.0) * state.portfolio_value
        difference = target_value - current.get(symbol, 0.0)
        if abs(difference) < minimum_trade_usd:
            continue
        side = "buy" if difference > 0 else "sell"
        notional = round(abs(difference), 2)
        quantity = None
        if side == "sell":
            price = prices.get(symbol, 0.0)
            if price <= 0:
                continue
            held = next((item.quantity for item in state.positions if item.symbol == symbol), 0.0)
            quantity = min(held, notional / price)
            notional = round(quantity * price, 2)
        orders.append(
            PlannedOrder(
                ref_id=deterministic_ref_id(decision_key, symbol, side, notional),
                decision_key=decision_key,
                symbol=symbol,
                side=side,
                planned_notional=notional,
                quantity=quantity,
            )
        )
    return sorted(orders, key=lambda item: (item.side != "sell", item.symbol))


def _data(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    value = payload.get("data") or {}
    return value if isinstance(value, Mapping) else {}


def fetch_broker_state(client: RobinhoodMCPClient) -> BrokerState:
    accounts = list(_data(client.call_tool("get_accounts", {})).get("accounts") or [])
    eligible = [item for item in accounts if item.get("agentic_allowed") is True and item.get("state") == "active"]
    if len(eligible) != 1:
        raise RuntimeError(f"Expected one active Agentic account, got {len(eligible)}")
    number = str(eligible[0]["account_number"])
    portfolio = _data(client.call_tool("get_portfolio", {"account_number": number}))
    positions_raw = list(_data(client.call_tool("get_equity_positions", {"account_number": number})).get("positions") or [])
    positions = tuple(
        Position(
            symbol=str(item.get("symbol", "")).upper(),
            quantity=_number(item.get("quantity")),
            market_value=_number(item.get("market_value") or item.get("equity_value") or item.get("quantity")) * (1.0 if item.get("market_value") or item.get("equity_value") else _number(item.get("price") or item.get("last_price"))),
        )
        for item in positions_raw
        if item.get("symbol") and _number(item.get("quantity")) > 0
    )
    return BrokerState(
        account_number=number,
        portfolio_value=_number(portfolio.get("total_value")),
        cash=_number(portfolio.get("cash")),
        option_value=_number(portfolio.get("options_value")),
        crypto_value=_number(portfolio.get("crypto_value")),
        futures_value=_number(portfolio.get("futures_value")),
        positions=positions,
    )


def execute_orders(
    *,
    settings: Settings,
    root: Path,
    ledger: Ledger,
    client: RobinhoodMCPClient,
    state: BrokerState,
    orders: Sequence[PlannedOrder],
    quotes: Mapping[str, Mapping[str, object]],
    tradability: Mapping[str, bool],
    trading_date: str,
    now: datetime,
    allowed_buy_symbols: Optional[set[str]] = None,
    decision_started_at: Optional[datetime] = None,
    reference_prices: Optional[Mapping[str, float]] = None,
) -> list[Dict[str, Any]]:
    results: list[Dict[str, Any]] = []
    for order in orders:
        summary = {
            "ref_id": order.ref_id,
            "symbol": order.symbol,
            "side": order.side,
            "planned_notional": order.planned_notional,
        }
        existing = ledger.get_order(order.ref_id)
        if existing and existing["status"] in NON_REPEATABLE_ORDER_STATES:
            results.append({**summary, "status": "duplicate_suppressed"})
            continue
        risk = approve_order(
            OrderIntent(order.symbol, order.side, order.planned_notional, order.quantity),
            policy=settings.risk,
            portfolio_value=state.portfolio_value,
            cash=state.cash,
            positions_value=state.positions_value,
            daily_order_notional=ledger.daily_order_notional(trading_date),
            daily_order_count=ledger.daily_order_count(trading_date),
            daily_open_value=ledger.daily_open_value(trading_date),
            high_watermark=ledger.high_watermark(),
            quote=quotes.get(order.symbol, {}),
            now=now,
            tradable=tradability.get(order.symbol, False),
            allowed_buy_symbols=allowed_buy_symbols,
            decision_started_at=decision_started_at,
            reference_price=(reference_prices or {}).get(order.symbol),
            option_value=state.option_value,
            crypto_value=state.crypto_value,
            futures_value=state.futures_value,
        )
        status = "approved" if risk.approved else "blocked"
        ledger.upsert_order(
            ref_id=order.ref_id, decision_key=order.decision_key,
            trading_date=trading_date, mode=settings.mode, symbol=order.symbol,
            side=order.side, order_type=settings.execution.order_type,
            quantity=f"{order.quantity:.8f}" if order.quantity is not None else None,
            dollar_amount=f"{order.planned_notional:.2f}" if order.side == "buy" else None,
            planned_notional=order.planned_notional, status=status,
            response={"risk": risk.to_dict()},
        )
        if not risk.approved:
            results.append({**summary, "status": status, "risk": risk.to_dict()})
            continue
        arguments: Dict[str, Any] = {
            "account_number": state.account_number,
            "symbol": order.symbol,
            "side": order.side,
            "type": settings.execution.order_type,
            "time_in_force": settings.execution.time_in_force,
            "market_hours": settings.execution.market_hours,
        }
        if order.side == "buy":
            arguments["dollar_amount"] = f"{order.planned_notional:.2f}"
        else:
            arguments["quantity"] = f"{order.quantity:.8f}"
        review = client.call_tool("review_equity_order", arguments)
        checks = _data(review).get("order_checks") or {}
        if _review_has_failure(checks):
            ledger.upsert_order(
                ref_id=order.ref_id, decision_key=order.decision_key, trading_date=trading_date,
                mode=settings.mode, symbol=order.symbol, side=order.side,
                order_type=settings.execution.order_type,
                quantity=arguments.get("quantity"), dollar_amount=arguments.get("dollar_amount"),
                planned_notional=order.planned_notional, status="review_rejected", response=review,
            )
            results.append({**summary, "status": "review_rejected"})
            continue
        if settings.mode != "LIVE":
            ledger.upsert_order(
                ref_id=order.ref_id, decision_key=order.decision_key, trading_date=trading_date,
                mode=settings.mode, symbol=order.symbol, side=order.side,
                order_type=settings.execution.order_type,
                quantity=arguments.get("quantity"), dollar_amount=arguments.get("dollar_amount"),
                planned_notional=order.planned_notional, status="hypothetical_reviewed", response=review,
            )
            results.append({**summary, "status": "hypothetical_reviewed"})
            continue
        assert_live_armed(settings, root, state.account_number, trading_date)
        placement_arguments = dict(arguments)
        placement_arguments["ref_id"] = order.ref_id
        placed = client.call_tool("place_equity_order", placement_arguments)
        placed_data = _data(placed)
        broker_id = str(placed_data.get("id") or placed_data.get("order_id") or "") or None
        ledger.upsert_order(
            ref_id=order.ref_id, decision_key=order.decision_key, trading_date=trading_date,
            mode=settings.mode, symbol=order.symbol, side=order.side,
            order_type=settings.execution.order_type,
            quantity=arguments.get("quantity"), dollar_amount=arguments.get("dollar_amount"),
            planned_notional=order.planned_notional, status="submitted",
            broker_order_id=broker_id, response=placed,
        )
        results.append(
            {**summary, "status": "submitted", "broker_order_id": broker_id}
        )
    return results


def reconcile_orders(client: RobinhoodMCPClient, ledger: Ledger, state: BrokerState, trading_date: str) -> int:
    payload = client.call_tool("get_equity_orders", {"account_number": state.account_number})
    rows = list(_data(payload).get("orders") or _data(payload).get("results") or [])
    changed = 0
    for row in rows:
        ref_id = str(row.get("ref_id") or row.get("client_order_id") or "")
        existing = ledger.get_order(ref_id) if ref_id else None
        if not existing:
            continue
        status = str(row.get("state") or row.get("status") or existing["status"]).lower()
        filled_quantity = _number(
            row.get("cumulative_quantity")
            or row.get("filled_quantity")
            or row.get("executed_quantity")
        )
        average_fill_price = _number(
            row.get("average_price")
            or row.get("average_fill_price")
            or row.get("executed_price")
        )
        if status == "filled" and existing["side"] == "buy":
            current_position = next(
                (position for position in state.positions if position.symbol == existing["symbol"]),
                None,
            )
            if current_position is None or current_position.quantity + 1e-8 < filled_quantity:
                status = "filled_position_mismatch"
        ledger.upsert_order(
            ref_id=ref_id, decision_key=existing["decision_key"], trading_date=trading_date,
            mode=existing["mode"], symbol=existing["symbol"], side=existing["side"],
            order_type=existing["order_type"], quantity=existing["quantity"],
            dollar_amount=existing["dollar_amount"], planned_notional=existing["planned_notional"],
            status=status, broker_order_id=str(row.get("id") or existing["broker_order_id"] or "") or None,
            filled_quantity=filled_quantity or None,
            average_fill_price=average_fill_price or None,
            response=dict(row),
        )
        changed += 1
    return changed


def execution_toolset(live: bool) -> frozenset[str]:
    tools = DECISION_READ_ONLY_TOOLS | ORDER_REVIEW_TOOLS
    return tools | ORDER_WRITE_TOOLS if live else tools


def _review_has_failure(checks: object) -> bool:
    """Fail only on explicit negative review results; unknown schemas fail closed."""
    if checks in (None, {}, []):
        return False
    text = json.dumps(checks, default=str).lower()
    negative = ("failed", "failure", "rejected", "not_allowed", "not allowed", '"passed":false')
    return any(marker in text for marker in negative)


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
