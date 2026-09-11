from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import fmean
from typing import Dict, Mapping, Optional, Sequence

from .config import PortfolioSettings, RiskSettings
from .forecasting import AssetForecast
from .market_data import DailyBar


@dataclass(frozen=True)
class TargetPortfolio:
    weights: Dict[str, float]
    cash_weight: float
    objective_value: float
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _daily_returns(bars: Sequence[DailyBar], count: int = 60) -> Dict[str, float]:
    chosen = bars[-(count + 1) :]
    return {
        chosen[index].begins_at[:10]: chosen[index].close / chosen[index - 1].close - 1
        for index in range(1, len(chosen))
        if chosen[index - 1].close > 0
    }


def shrinkage_covariance(
    symbols: Sequence[str], histories: Mapping[str, Sequence[DailyBar]], shrinkage: float
) -> list[list[float]]:
    returns = {symbol: _daily_returns(histories[symbol]) for symbol in symbols}
    dates = sorted(set.intersection(*(set(values) for values in returns.values())))
    if len(dates) < 20:
        raise ValueError("At least 20 aligned return observations are required")
    rows = [[returns[symbol][date] for date in dates] for symbol in symbols]
    means = [fmean(row) for row in rows]
    denominator = max(len(dates) - 1, 1)
    covariance = []
    for left, left_row in enumerate(rows):
        covariance.append([])
        for right, right_row in enumerate(rows):
            sample = sum(
                (a - means[left]) * (b - means[right])
                for a, b in zip(left_row, right_row)
            ) / denominator
            if left != right:
                sample *= 1.0 - shrinkage
            covariance[-1].append(sample)
    return covariance


def _project(
    weights: list[float],
    cap: float,
    total_cap: float,
    groups: Sequence[str],
    group_cap: float,
    floors: Optional[Sequence[float]] = None,
) -> list[float]:
    minimums = list(floors or [0.0] * len(weights))
    if len(minimums) != len(weights):
        raise ValueError("Weight floors must align with portfolio weights")
    minimums = [max(0.0, min(value, cap)) for value in minimums]
    projected = [
        max(floor_value, min(value, cap))
        for value, floor_value in zip(weights, minimums)
    ]

    def reduce_to_cap(indexes: Sequence[int], limit: float) -> None:
        group_total = sum(projected[index] for index in indexes)
        if group_total <= limit + 1e-12:
            return
        slack = sum(
            projected[index] - minimums[index] for index in indexes
        )
        required = group_total - limit
        if slack + 1e-12 < required:
            raise ValueError("Current-position floors exceed a portfolio constraint")
        scale = max(0.0, (slack - required) / max(slack, 1e-12))
        for index in indexes:
            projected[index] = minimums[index] + (
                projected[index] - minimums[index]
            ) * scale

    for group in set(groups):
        indexes = [index for index, value in enumerate(groups) if value == group]
        reduce_to_cap(indexes, group_cap)
    reduce_to_cap(list(range(len(projected))), total_cap)
    return projected


def optimize_portfolio(
    forecasts: Sequence[AssetForecast],
    histories: Mapping[str, Sequence[DailyBar]],
    portfolio: PortfolioSettings,
    risk: RiskSettings,
    sectors: Optional[Mapping[str, str]] = None,
    current_weights: Optional[Mapping[str, float]] = None,
    minimum_weights: Optional[Mapping[str, float]] = None,
) -> TargetPortfolio:
    chosen = sorted(
        forecasts,
        key=lambda value: value.expected_excess_return_20d,
        reverse=True,
    )[: risk.max_positions]
    if not chosen:
        return TargetPortfolio(weights={}, cash_weight=1.0, objective_value=0.0)
    symbols = [item.symbol for item in chosen]
    sector_groups = [
        (sectors or {}).get(symbol) or f"unknown:{symbol}" for symbol in symbols
    ]
    covariance_daily = shrinkage_covariance(symbols, histories, shrinkage=0.50)
    covariance_20d = [
        [20.0 * value for value in row] for row in covariance_daily
    ]
    expected_20d = [item.expected_excess_return_20d for item in chosen]
    uncertainty_20d = [item.uncertainty_20d for item in chosen]
    floors = [
        max(0.0, (minimum_weights or {}).get(symbol, 0.0))
        for symbol in symbols
    ]
    floors = _project(
        floors,
        min(risk.max_position_fraction, portfolio.soft_max_position_fraction),
        portfolio.max_invested_fraction,
        sector_groups,
        portfolio.soft_max_sector_fraction,
    )
    baseline = _project(
        [max(0.0, (current_weights or {}).get(symbol, 0.0)) for symbol in symbols],
        min(risk.max_position_fraction, portfolio.soft_max_position_fraction),
        portfolio.max_invested_fraction,
        sector_groups,
        portfolio.soft_max_sector_fraction,
        floors,
    )
    weights = baseline[:]
    for iteration in range(portfolio.optimizer_iterations):
        step = portfolio.optimizer_step_size / math.sqrt(iteration + 1.0)
        gradient = []
        for index in range(len(chosen)):
            risk_gradient = sum(
                covariance_20d[index][other] * weights[other]
                for other in range(len(chosen))
            )
            difference = weights[index] - baseline[index]
            cost_gradient = (
                portfolio.reallocation_cost_fraction
                if difference > 1e-9
                else -portfolio.reallocation_cost_fraction
                if difference < -1e-9
                else 0.0
            )
            gradient.append(
                expected_20d[index]
                - portfolio.risk_aversion * risk_gradient
                - 2.0
                * portfolio.uncertainty_penalty
                * uncertainty_20d[index] ** 2
                * weights[index]
                - cost_gradient
            )
        weights = _project(
            [weight + step * value for weight, value in zip(weights, gradient)],
            min(risk.max_position_fraction, portfolio.soft_max_position_fraction),
            portfolio.max_invested_fraction,
            sector_groups,
            portfolio.soft_max_sector_fraction,
            floors,
        )
    def objective(values: Sequence[float], include_cost: bool) -> float:
        variance = sum(
            values[left] * covariance_20d[left][right] * values[right]
            for left in range(len(values))
            for right in range(len(values))
        )
        estimation_risk = sum(
            (weight * uncertainty) ** 2
            for weight, uncertainty in zip(values, uncertainty_20d)
        )
        turnover = sum(
            abs(weight - old) for weight, old in zip(values, baseline)
        )
        return (
            sum(a * b for a, b in zip(values, expected_20d))
            - 0.5 * portfolio.risk_aversion * variance
            - portfolio.uncertainty_penalty * estimation_risk
            - (portfolio.reallocation_cost_fraction * turnover if include_cost else 0.0)
        )

    candidate_optimized_objective = objective(weights, include_cost=True)
    optimized_objective = candidate_optimized_objective
    baseline_objective = objective(baseline, include_cost=False)
    if optimized_objective <= baseline_objective + 1e-9:
        weights = baseline
        optimized_objective = baseline_objective
    mapped = {
        symbol: round(weight, 8)
        for symbol, weight in zip(symbols, weights)
        if weight > 1e-6
    }
    portfolio_variance = sum(
        weights[left] * covariance_20d[left][right] * weights[right]
        for left in range(len(weights))
        for right in range(len(weights))
    )
    diagnostics: Dict[str, object] = {
        "horizon_days": 20,
        "risk_aversion": portfolio.risk_aversion,
        "uncertainty_penalty": portfolio.uncertainty_penalty,
        "reallocation_cost_fraction": portfolio.reallocation_cost_fraction,
        "baseline_objective": baseline_objective,
        "candidate_optimized_net_objective": candidate_optimized_objective,
        "optimized_net_objective": optimized_objective,
        "objective_improvement": optimized_objective - baseline_objective,
        "portfolio_variance_20d": portfolio_variance,
        "symbols": {},
    }
    symbol_diagnostics = diagnostics["symbols"]
    assert isinstance(symbol_diagnostics, dict)
    for index, symbol in enumerate(symbols):
        covariance_marginal = sum(
            covariance_20d[index][other] * weights[other]
            for other in range(len(weights))
        )
        symbol_diagnostics[symbol] = {
            "expected_excess_return_20d": expected_20d[index],
            "uncertainty_20d": uncertainty_20d[index],
            "variance_20d": covariance_20d[index][index],
            "baseline_weight": baseline[index],
            "minimum_weight": floors[index],
            "final_weight": weights[index],
            "expected_return_contribution": weights[index] * expected_20d[index],
            "covariance_penalty_contribution": (
                0.5 * portfolio.risk_aversion * weights[index] * covariance_marginal
            ),
            "estimation_penalty_contribution": (
                portfolio.uncertainty_penalty
                * (weights[index] * uncertainty_20d[index]) ** 2
            ),
            "reallocation_cost_contribution": (
                portfolio.reallocation_cost_fraction
                * abs(weights[index] - baseline[index])
            ),
            "initial_marginal_alpha_after_cost": (
                expected_20d[index]
                - (
                    portfolio.reallocation_cost_fraction
                    if baseline[index] <= 1e-9
                    else 0.0
                )
            ),
            "final_marginal_objective": (
                expected_20d[index]
                - portfolio.risk_aversion * covariance_marginal
                - 2.0
                * portfolio.uncertainty_penalty
                * uncertainty_20d[index] ** 2
                * weights[index]
            ),
        }
    return TargetPortfolio(
        weights=mapped,
        cash_weight=round(max(0.0, 1.0 - sum(mapped.values())), 8),
        objective_value=optimized_objective,
        diagnostics=diagnostics,
    )
