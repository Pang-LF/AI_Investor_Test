from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional, Sequence

from .config import RiskSettings


LEVERAGED_OR_INVERSE_ETFS = frozenset(
    {"TQQQ", "SQQQ", "UPRO", "SPXU", "SOXL", "SOXS", "TECL", "TECS", "FAS", "FAZ"}
)


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str
    planned_notional: float
    quantity: Optional[float] = None


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def approve_order(
    intent: OrderIntent,
    *,
    policy: RiskSettings,
    portfolio_value: float,
    cash: float,
    positions_value: Mapping[str, float],
    daily_order_notional: float,
    daily_order_count: int,
    daily_open_value: Optional[float],
    high_watermark: Optional[float],
    quote: Mapping[str, object],
    now: datetime,
    tradable: bool,
    allowed_buy_symbols: Optional[set[str]] = None,
    decision_started_at: Optional[datetime] = None,
    reference_price: Optional[float] = None,
    asset_type: str = "equity",
    option_value: float = 0.0,
    crypto_value: float = 0.0,
    futures_value: float = 0.0,
) -> RiskDecision:
    reasons: list[str] = []
    symbol = intent.symbol.upper()
    if portfolio_value <= 0:
        reasons.append("invalid_portfolio_value")
    if asset_type not in {"equity", "etf"}:
        reasons.append("asset_type_not_allowed")
    if symbol in LEVERAGED_OR_INVERSE_ETFS:
        reasons.append("leveraged_or_inverse_etf")
    if intent.side not in {"buy", "sell"}:
        reasons.append("invalid_side")
    if (
        intent.planned_notional <= 0
        or intent.planned_notional > policy.max_trade_usd
        or (
            portfolio_value > 0
            and intent.planned_notional
            > portfolio_value * policy.max_trade_fraction + 1e-6
        )
    ):
        reasons.append("trade_size_limit")
    if option_value != 0 or crypto_value != 0 or futures_value != 0:
        reasons.append("prohibited_account_exposure")
    if not tradable:
        reasons.append("instrument_not_tradable")

    last = _number(quote.get("last_trade_price") or quote.get("last_price"))
    bid = _number(quote.get("bid_price"))
    ask = _number(quote.get("ask_price"))
    if last <= 0 or bid <= 0 or ask <= 0 or ask < bid:
        reasons.append("invalid_quote")
    elif (ask - bid) / ((ask + bid) / 2.0) > policy.max_spread_fraction:
        reasons.append("spread_limit")
    raw_time = quote.get("last_trade_time") or quote.get("venue_last_trade_time")
    try:
        quote_time = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
        if quote_time.tzinfo is None:
            quote_time = quote_time.replace(tzinfo=timezone.utc)
        age = (now.astimezone(timezone.utc) - quote_time.astimezone(timezone.utc)).total_seconds()
        if age < -300 or age > policy.max_execution_quote_age_seconds:
            reasons.append("stale_quote")
    except (TypeError, ValueError):
        reasons.append("missing_quote_timestamp")
    if decision_started_at is not None:
        decision_age = (
            now.astimezone(timezone.utc)
            - decision_started_at.astimezone(timezone.utc)
        ).total_seconds()
        if decision_age < 0 or decision_age > policy.max_decision_age_seconds:
            reasons.append("decision_age_limit")
    if reference_price is not None and reference_price > 0 and last > 0:
        if (
            abs(last / reference_price - 1.0)
            > policy.max_decision_price_drift_fraction
        ):
            reasons.append("decision_price_drift_limit")

    is_buy = intent.side == "buy"
    if is_buy and allowed_buy_symbols is not None and symbol not in allowed_buy_symbols:
        reasons.append("symbol_not_in_approved_universe")
    if is_buy and intent.planned_notional > cash + 1e-6:
        reasons.append("insufficient_cash_no_margin")
    resulting = dict(positions_value)
    current = resulting.get(symbol, 0.0)
    resulting[symbol] = current + intent.planned_notional if is_buy else max(
        0.0, current - intent.planned_notional
    )
    nonzero = {key: value for key, value in resulting.items() if value > 0.01}
    if len(nonzero) > policy.max_positions:
        reasons.append("position_count_limit")
    if portfolio_value > 0 and resulting.get(symbol, 0.0) / portfolio_value > policy.max_position_fraction + 1e-6:
        reasons.append("position_size_limit")
    if daily_order_notional + intent.planned_notional > portfolio_value * policy.max_daily_turnover_fraction + 1e-6:
        reasons.append("daily_turnover_limit")
    if daily_order_count >= policy.max_live_orders_per_day:
        reasons.append("daily_order_count_limit")
    if is_buy and daily_open_value and portfolio_value / daily_open_value - 1 <= -policy.max_daily_loss_fraction:
        reasons.append("daily_loss_circuit_breaker")
    if is_buy and high_watermark and portfolio_value / high_watermark - 1 <= -policy.max_portfolio_drawdown_fraction:
        reasons.append("drawdown_circuit_breaker")
    return RiskDecision(approved=not reasons, reasons=tuple(sorted(set(reasons))))


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
