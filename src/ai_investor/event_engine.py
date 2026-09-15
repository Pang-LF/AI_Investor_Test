from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import fmean, median
from typing import Any, Dict, Mapping, Sequence

from .config import EventEngineSettings
from .forecasting import _aligned_series, _quantile, _rolling_beta
from .market_data import DailyBar


EVENT_ENGINE_VERSION = "empirical_event_analog_v0.1"


@dataclass(frozen=True)
class EventOutcomeDistribution:
    horizon_days: int
    expected_excess_return: float
    median_excess_return: float
    probability_positive: float
    uncertainty: float
    p10: float
    p90: float


@dataclass(frozen=True)
class EventShadowForecast:
    symbol: str
    observed_at: str
    event_type: str
    change_from_previous_close: float
    observed_price: float
    benchmark_price: float
    beta_to_spy: float
    discovery_sources: tuple[str, ...]
    analog_scope: str
    analog_samples: int
    independent_event_dates: int
    confidence: str
    sample_support: str
    one_day: EventOutcomeDistribution
    five_day: EventOutcomeDistribution
    live_entry_enabled: bool = False
    engine_version: str = EVENT_ENGINE_VERSION
    limitations: tuple[str, ...] = (
        "Current signal is intraday versus prior close; analogs are completed close-to-close events.",
        "No point-in-time earnings, guidance, analyst-revision, or news classification is used yet.",
        "Historical analogs use today's selected universe and therefore contain selection bias.",
        "Output is shadow research evidence and cannot create a portfolio weight or order.",
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def classify_price_event(change: float, strong_threshold: float) -> str:
    strength = "strong" if abs(change) >= strong_threshold else "standard"
    direction = "up" if change > 0 else "down"
    return f"{strength}_{direction}"


def _forward_excess(
    closes: Sequence[float], benchmark: Sequence[float], index: int, horizon: int
) -> float:
    beta = _rolling_beta(closes, benchmark, index)
    future = index + horizon
    return (
        closes[future] / closes[index]
        - 1.0
        - beta * (benchmark[future] / benchmark[index] - 1.0)
    )


def _historical_analogs(
    histories: Mapping[str, Sequence[DailyBar]],
    settings: EventEngineSettings,
) -> Dict[str, list[tuple[str, float, float]]]:
    benchmark_bars = histories.get("SPY") or []
    analogs: Dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for symbol, bars in histories.items():
        if symbol == "SPY" or not bars:
            continue
        dates, closes, _, benchmark = _aligned_series(bars, benchmark_bars)
        for index in range(60, len(dates) - 5):
            if closes[index - 1] <= 0:
                continue
            change = closes[index] / closes[index - 1] - 1.0
            if abs(change) < settings.minimum_absolute_move:
                continue
            event_type = classify_price_event(
                change, settings.strong_move_threshold
            )
            analogs[event_type].append((
                dates[index],
                _forward_excess(closes, benchmark, index, 1),
                _forward_excess(closes, benchmark, index, 5),
            ))
    return analogs


def _winsorized(values: Sequence[float], horizon: int) -> list[float]:
    if not values:
        return []
    hard_cap = 0.20 if horizon == 1 else 0.50
    bounded = [max(-hard_cap, min(value, hard_cap)) for value in values]
    lower = _quantile(bounded, 0.10)
    upper = _quantile(bounded, 0.90)
    return [max(lower, min(value, upper)) for value in bounded]


def _outcome_distribution(
    analogs: Sequence[tuple[str, float, float]],
    horizon: int,
    prior_samples: int,
) -> EventOutcomeDistribution:
    value_index = 1 if horizon == 1 else 2
    raw = [float(item[value_index]) for item in analogs]
    values = _winsorized(raw, horizon)
    by_date: Dict[str, list[float]] = defaultdict(list)
    for item, value in zip(analogs, values):
        by_date[item[0]].append(value)
    date_values = [fmean(items) for items in by_date.values()]
    strength = len(date_values) / (len(date_values) + prior_samples)
    # Cross-sectional events on the same date share market and regime shocks.
    # Treat each date as one independent block so event-heavy sessions cannot
    # dominate the center, probability, uncertainty, or tails.
    average = fmean(date_values) if date_values else 0.0
    robust_center = (
        0.5 * average + 0.5 * median(date_values) if date_values else 0.0
    )
    expected = strength * robust_center
    variance = (
        fmean((value - average) ** 2 for value in date_values)
        if date_values else 0.0
    )
    positive_dates = sum(value > 0 for value in date_values)
    probability = (
        (positive_dates + prior_samples * 0.5)
        / (len(date_values) + prior_samples)
        if date_values else 0.5
    )
    return EventOutcomeDistribution(
        horizon_days=horizon,
        expected_excess_return=expected,
        median_excess_return=median(date_values) if date_values else 0.0,
        probability_positive=probability,
        uncertainty=math.sqrt(variance),
        p10=_quantile(date_values, 0.10) if date_values else 0.0,
        p90=_quantile(date_values, 0.90) if date_values else 0.0,
    )


def build_event_shadow_forecasts(
    *,
    histories: Mapping[str, Sequence[DailyBar]],
    quotes: Mapping[str, Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    triggers: Sequence[Mapping[str, Any]],
    settings: EventEngineSettings,
    observed_at: str,
) -> list[EventShadowForecast]:
    if not settings.enabled:
        return []
    investable = {
        str(item.get("symbol") or "").upper()
        for item in entries
        if item.get("symbol") and item.get("investable", True)
    }
    event_bucket = {
        str(item.get("symbol") or "").upper()
        for item in entries
        if item.get("symbol") and item.get("bucket") == "event"
    }
    trigger_symbols = {
        str(item.get("symbol") or "").upper()
        for item in triggers
        if item.get("symbol") and item.get("type") == "price_move"
    }
    sources: Dict[str, set[str]] = defaultdict(set)
    for symbol in event_bucket:
        sources[symbol].add("dynamic_event_bucket")
    for symbol in trigger_symbols:
        sources[symbol].add("intraday_price_trigger")

    current_events = []
    for symbol in sorted((event_bucket | trigger_symbols) & investable):
        quote = quotes.get(symbol) or {}
        last = float(quote.get("last_trade_price") or 0.0)
        previous = float(quote.get("adjusted_previous_close") or 0.0)
        if last <= 0 or previous <= 0:
            continue
        change = last / previous - 1.0
        if abs(change) < settings.minimum_absolute_move:
            continue
        current_events.append((symbol, change))
    current_events.sort(key=lambda item: (-abs(item[1]), item[0]))
    current_events = current_events[:settings.maximum_candidates]

    analogs_by_type = _historical_analogs(histories, settings)
    benchmark_quote = quotes.get("SPY") or {}
    benchmark_price = float(benchmark_quote.get("last_trade_price") or 0.0)
    forecasts = []
    for symbol, change in current_events:
        event_type = classify_price_event(change, settings.strong_move_threshold)
        analogs = analogs_by_type.get(event_type, [])
        analog_scope = event_type
        if len(analogs) < settings.minimum_analog_samples:
            direction = event_type.rsplit("_", 1)[-1]
            analogs = [
                item
                for name, group in analogs_by_type.items()
                if name.endswith("_" + direction)
                for item in group
            ]
            analog_scope = f"all_{direction}_events_fallback"
        independent_dates = len({item[0] for item in analogs})
        sample_support = (
            "INSUFFICIENT"
            if len(analogs) < settings.minimum_analog_samples
            else "LOW"
            if independent_dates < 20
            else "MEDIUM"
            if independent_dates < 50
            else "HIGH"
        )
        beta = 1.0
        symbol_history = histories.get(symbol) or []
        benchmark_history = histories.get("SPY") or []
        if symbol_history and benchmark_history:
            _, closes, _, benchmark_closes = _aligned_series(
                symbol_history, benchmark_history
            )
            if len(closes) >= 20:
                beta = _rolling_beta(closes, benchmark_closes, len(closes) - 1)
        quote = quotes.get(symbol) or {}
        forecasts.append(EventShadowForecast(
            symbol=symbol,
            observed_at=observed_at,
            event_type=event_type,
            change_from_previous_close=change,
            observed_price=float(quote.get("last_trade_price") or 0.0),
            benchmark_price=benchmark_price,
            beta_to_spy=beta,
            discovery_sources=tuple(sorted(sources[symbol])),
            analog_scope=analog_scope,
            analog_samples=len(analogs),
            independent_event_dates=independent_dates,
            confidence="SHADOW_UNVALIDATED",
            sample_support=sample_support,
            one_day=_outcome_distribution(
                analogs, 1, settings.shrinkage_prior_samples
            ),
            five_day=_outcome_distribution(
                analogs, 5, settings.shrinkage_prior_samples
            ),
        ))
    return forecasts
