from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

from .config import ForecastSettings
from .forecasting import AssetForecast


def build_candidate_funnel(
    entries: Sequence[Mapping[str, Any]],
    forecasts: Sequence[AssetForecast],
    settings: ForecastSettings,
    *,
    event_symbols: Iterable[str],
    research_symbols: Iterable[str],
    holding_symbols: Iterable[str],
) -> list[Dict[str, Any]]:
    by_symbol = {item.symbol: item for item in forecasts}
    event_set = {symbol.upper() for symbol in event_symbols}
    research_set = {symbol.upper() for symbol in research_symbols}
    holding_set = {symbol.upper() for symbol in holding_symbols}
    ranked = sorted(
        (
            forecast
            for forecast in forecasts
            if (
                forecast.raw_expected_excess_return_20d
                >= settings.research_min_raw_expected_excess_return_20d
                and forecast.raw_probability_positive_excess_20d
                >= settings.research_min_raw_probability_positive
            )
            or forecast.symbol in event_set
        ),
        key=lambda forecast: (
            forecast.raw_expected_excess_return_20d
            / max(forecast.uncertainty_20d, 1e-9),
            forecast.raw_expected_excess_return_20d,
        ),
        reverse=True,
    )
    ranks = {forecast.symbol: index + 1 for index, forecast in enumerate(ranked)}
    rows: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        symbol = str(entry.get("symbol", "")).upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        forecast = by_symbol.get(symbol)
        investable = bool(entry.get("investable", True))
        raw_gate_passed = bool(
            forecast
            and forecast.raw_expected_excess_return_20d
            >= settings.research_min_raw_expected_excess_return_20d
            and forecast.raw_probability_positive_excess_20d
            >= settings.research_min_raw_probability_positive
        )
        event_bypass = bool(forecast and symbol in event_set)
        selected = symbol in research_set
        if not investable:
            reason = "context_only_not_investable"
        elif forecast is None:
            reason = "missing_or_short_history"
        elif symbol in holding_set:
            reason = "current_holding_priority"
        elif selected:
            reason = "selected_by_quant_rank_or_event"
        elif raw_gate_passed or event_bypass:
            reason = "qualified_but_below_research_capacity"
        else:
            reason = "research_threshold_not_met"
        rows.append(
            {
                "symbol": symbol,
                "bucket": entry.get("bucket"),
                "investable": investable,
                "current_holding": symbol in holding_set,
                "event_candidate": symbol in event_set,
                "forecast_available": forecast is not None,
                "raw_expected_excess_return_20d": (
                    forecast.raw_expected_excess_return_20d if forecast else None
                ),
                "raw_probability_positive_excess_20d": (
                    forecast.raw_probability_positive_excess_20d if forecast else None
                ),
                "edge_ratio_20d": (
                    forecast.raw_expected_excess_return_20d
                    / max(forecast.uncertainty_20d, 1e-9)
                    if forecast
                    else None
                ),
                "research_gate_passed": raw_gate_passed,
                "research_rank": ranks.get(symbol),
                "selected_for_research": selected,
                "research_selection_reason": reason,
                "llm_verdict": None,
                "investment_eligibility_failures": [],
                "target_weight": 0.0,
                "final_disposition": "not_researched",
            }
        )
    return rows


def finalize_candidate_funnel(
    rows: Sequence[Mapping[str, Any]],
    *,
    assessments: Mapping[str, Mapping[str, Any]],
    investment_failures: Mapping[str, Sequence[str]],
    holding_decisions: Mapping[str, Mapping[str, Any]],
    target_weights: Mapping[str, float],
    orders: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    order_statuses = {
        str(order.get("symbol") or order.get("ticker") or "").upper(): str(
            order.get("status") or "unknown"
        )
        for order in orders
    }
    finalized: list[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        symbol = str(row["symbol"])
        assessment = assessments.get(symbol, {})
        verdict = assessment.get("verdict")
        failures = list(investment_failures.get(symbol, ()))
        target = float(target_weights.get(symbol, 0.0))
        holding = holding_decisions.get(symbol, {})
        if symbol in order_statuses:
            disposition = f"order_{order_statuses[symbol]}"
        elif target > 0 and holding:
            disposition = "retained_no_rebalance"
        elif target > 0:
            disposition = "selected_target_no_order"
        elif verdict and verdict != "allow":
            disposition = f"llm_{verdict}"
        elif failures:
            disposition = "investment_eligibility_failed"
        elif row.get("selected_for_research"):
            disposition = "optimizer_zero_weight"
        else:
            disposition = str(row.get("final_disposition"))
        row.update(
            {
                "llm_verdict": verdict,
                "investment_eligibility_failures": failures,
                "holding_decision": holding or None,
                "target_weight": target,
                "final_disposition": disposition,
            }
        )
        finalized.append(row)
    return finalized
