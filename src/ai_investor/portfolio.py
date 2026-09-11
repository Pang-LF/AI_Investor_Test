from __future__ import annotations

import math
from dataclasses import asdict, dataclass
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
) -> list[float]:
    projected = [max(0.0, min(value, cap)) for value in weights]
    for group in set(groups):
        indexes = [index for index, value in enumerate(groups) if value == group]
        group_total = sum(projected[index] for index in indexes)
        if group_total > group_cap:
            scale = group_cap / group_total
            for index in indexes:
                projected[index] *= scale
    total = sum(projected)
    if total > total_cap:
        projected = [value * total_cap / total for value in projected]
    return projected


def optimize_portfolio(
    forecasts: Sequence[AssetForecast],
    histories: Mapping[str, Sequence[DailyBar]],
    portfolio: PortfolioSettings,
    risk: RiskSettings,
    sectors: Optional[Mapping[str, str]] = None,
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
    covariance = shrinkage_covariance(symbols, histories, shrinkage=0.50)
    expected_daily = [item.expected_excess_return_20d / 20.0 for item in chosen]
    uncertainty_daily = [item.uncertainty_20d / math.sqrt(20.0) for item in chosen]
    weights = [0.0] * len(chosen)
    for iteration in range(portfolio.optimizer_iterations):
        step = portfolio.optimizer_step_size / math.sqrt(iteration + 1.0)
        gradient = []
        for index in range(len(chosen)):
            risk_gradient = sum(
                covariance[index][other] * weights[other]
                for other in range(len(chosen))
            )
            gradient.append(
                expected_daily[index]
                - portfolio.risk_aversion * risk_gradient
                - portfolio.uncertainty_penalty * uncertainty_daily[index] ** 2
            )
        weights = _project(
            [weight + step * value for weight, value in zip(weights, gradient)],
            min(risk.max_position_fraction, portfolio.soft_max_position_fraction),
            portfolio.max_invested_fraction,
            sector_groups,
            portfolio.soft_max_sector_fraction,
        )
    variance = sum(
        weights[left] * covariance[left][right] * weights[right]
        for left in range(len(weights))
        for right in range(len(weights))
    )
    objective = sum(a * b for a, b in zip(weights, expected_daily)) - (
        0.5 * portfolio.risk_aversion * variance
    )
    mapped = {
        symbol: round(weight, 8)
        for symbol, weight in zip(symbols, weights)
        if weight > 1e-6
    }
    return TargetPortfolio(
        weights=mapped,
        cash_weight=round(max(0.0, 1.0 - sum(mapped.values())), 8),
        objective_value=objective,
    )
