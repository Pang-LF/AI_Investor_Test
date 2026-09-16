from __future__ import annotations

import math
import json
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Dict, Mapping, Sequence

from .forecasting import AssetForecast, _aligned_series, _beta_adjusted_return
from .market_data import DailyBar


ENGINE_VERSION = "cross_sectional_factor_rank_v0.1"
COMPONENT_FAMILIES = {
    "residual_trend": "trend",
    "breakout_channel": "trend",
    "reversal_5d": "reversal",
    "relative_volume_confirmation": "volume",
    "obv_confirmation": "volume",
    "low_downside_volatility": "risk",
    "low_drawdown_pressure": "risk",
}
FAMILY_NAMES = ("trend", "reversal", "volume", "risk")


@dataclass(frozen=True)
class FactorRankSignal:
    symbol: str
    data_as_of: str
    formation_price: float
    benchmark_price: float
    raw_factors: Dict[str, float]
    winsorized_factors: Dict[str, float]
    factor_percentile_ranks: Dict[str, float]
    family_ranks: Dict[str, float]
    composite_score: float
    composite_rank: int
    ridge_score: float
    ridge_rank: int
    engine_version: str = ENGINE_VERSION
    live_eligible: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FactorRankSnapshot:
    engine_version: str
    data_as_of: str
    signals: tuple[FactorRankSignal, ...]
    top_25: tuple[str, ...]
    ridge_top_25: tuple[str, ...]
    top_25_overlap: float
    pairwise_family_spearman: Dict[str, float]
    pairwise_family_top_bucket_overlap: Dict[str, float]
    live_eligible: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _returns(values: Sequence[float]) -> list[float]:
    return [
        values[index] / values[index - 1] - 1.0
        for index in range(1, len(values))
        if values[index - 1] > 0
    ]


def _raw_components(
    bars: Sequence[DailyBar], benchmark: Sequence[DailyBar]
) -> tuple[str, float, float, Dict[str, float]] | None:
    dates, closes, volumes, benchmark_closes = _aligned_series(bars, benchmark)
    asset_by_date = {bar.begins_at[:10]: bar for bar in bars}
    if len(dates) < 80 or dates[-1] != benchmark[-1].begins_at[:10]:
        return None
    index = len(dates) - 1
    residual_5 = _beta_adjusted_return(closes, benchmark_closes, index, 5)
    residual_20 = _beta_adjusted_return(closes, benchmark_closes, index, 20)
    residual_60 = _beta_adjusted_return(closes, benchmark_closes, index, 60)
    recent_bars = [asset_by_date[date] for date in dates[-60:]]
    valid_lows = [bar.low for bar in recent_bars if bar.low > 0]
    valid_highs = [bar.high for bar in recent_bars if bar.high > 0]
    channel_low = min(valid_lows) if valid_lows else min(closes[-60:])
    channel_high = max(valid_highs) if valid_highs else max(closes[-60:])
    channel = (
        2.0 * (closes[-1] - channel_low) / (channel_high - channel_low) - 1.0
        if channel_high > channel_low > 0 else 0.0
    )
    average_volume = fmean(volumes[-21:-1]) if len(volumes) >= 21 else 0.0
    relative_volume = volumes[-1] / average_volume if average_volume > 0 else 1.0
    directional_move = math.tanh(residual_5 / 0.03)
    relative_volume_confirmation = directional_move * math.log(
        max(0.20, min(relative_volume, 5.0))
    )
    recent_closes = closes[-21:]
    recent_volumes = volumes[-20:]
    obv_numerator = sum(
        (1.0 if current > previous else -1.0 if current < previous else 0.0) * volume
        for previous, current, volume in zip(
            recent_closes[:-1], recent_closes[1:], recent_volumes
        )
    )
    obv_confirmation = obv_numerator / max(sum(recent_volumes), 1.0)
    daily = _returns(closes[-21:])
    downside = [min(value, 0.0) for value in daily]
    downside_vol = math.sqrt(fmean(value * value for value in downside))
    peak = closes[-60]
    max_drawdown = 0.0
    for close in closes[-60:]:
        peak = max(peak, close)
        max_drawdown = min(max_drawdown, close / peak - 1.0)
    return (
        dates[-1],
        closes[-1],
        benchmark_closes[-1],
        {
            "residual_trend": 0.5 * residual_20 / 20.0 + 0.5 * residual_60 / 60.0,
            "breakout_channel": channel,
            "reversal_5d": -residual_5 / 5.0,
            "relative_volume_confirmation": relative_volume_confirmation,
            "obv_confirmation": obv_confirmation,
            "low_downside_volatility": -downside_vol,
            "low_drawdown_pressure": max_drawdown,
        },
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _percentile_ranks(values: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) <= 1:
        return {symbol: 0.5 for symbol in values}
    result: Dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_position = (index + end - 1) / 2.0
        rank = average_position / (len(ordered) - 1)
        for cursor in range(index, end):
            result[ordered[cursor][0]] = rank
        index = end
    return result


def _spearman(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    common = sorted(set(left) & set(right))
    if len(common) < 3:
        return 0.0
    a = _percentile_ranks({symbol: left[symbol] for symbol in common})
    b = _percentile_ranks({symbol: right[symbol] for symbol in common})
    mean_a = fmean(a.values())
    mean_b = fmean(b.values())
    numerator = sum((a[s] - mean_a) * (b[s] - mean_b) for s in common)
    denominator = math.sqrt(
        sum((a[s] - mean_a) ** 2 for s in common)
        * sum((b[s] - mean_b) ** 2 for s in common)
    )
    return numerator / denominator if denominator > 0 else 0.0


def _top_overlap(
    left: Mapping[str, float], right: Mapping[str, float], fraction: float
) -> float:
    common = set(left) & set(right)
    count = max(1, math.ceil(len(common) * fraction))
    top_left = set(sorted(common, key=lambda s: (-left[s], s))[:count])
    top_right = set(sorted(common, key=lambda s: (-right[s], s))[:count])
    return len(top_left & top_right) / count


def build_factor_rank_snapshot(
    histories: Mapping[str, Sequence[DailyBar]],
    ridge_forecasts: Sequence[AssetForecast],
    *,
    shortlist_size: int = 25,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    top_bucket_fraction: float = 0.10,
) -> FactorRankSnapshot:
    benchmark = histories.get("SPY") or []
    if not benchmark:
        raise ValueError("SPY history is required for factor challenger")
    raw_by_symbol: Dict[str, Dict[str, float]] = {}
    prices: Dict[str, tuple[str, float, float]] = {}
    for symbol, bars in histories.items():
        if symbol == "SPY":
            continue
        components = _raw_components(bars, benchmark)
        if components is None:
            continue
        date, price, benchmark_price, raw = components
        raw_by_symbol[symbol] = raw
        prices[symbol] = (date, price, benchmark_price)
    if not raw_by_symbol:
        raise ValueError("No point-in-time factor cross-section is available")
    dates = {item[0] for item in prices.values()}
    if len(dates) != 1:
        raise ValueError("Factor cross-section must share one completed data date")
    winsorized: Dict[str, Dict[str, float]] = {symbol: {} for symbol in raw_by_symbol}
    factor_ranks: Dict[str, Dict[str, float]] = {symbol: {} for symbol in raw_by_symbol}
    for factor in COMPONENT_FAMILIES:
        values = {symbol: raw[factor] for symbol, raw in raw_by_symbol.items()}
        lower = _quantile(list(values.values()), winsor_lower)
        upper = _quantile(list(values.values()), winsor_upper)
        bounded = {symbol: max(lower, min(value, upper)) for symbol, value in values.items()}
        ranked = _percentile_ranks(bounded)
        for symbol in raw_by_symbol:
            winsorized[symbol][factor] = bounded[symbol]
            factor_ranks[symbol][factor] = ranked[symbol]
    family_ranks: Dict[str, Dict[str, float]] = {symbol: {} for symbol in raw_by_symbol}
    for family in FAMILY_NAMES:
        members = [name for name, group in COMPONENT_FAMILIES.items() if group == family]
        for symbol in raw_by_symbol:
            family_ranks[symbol][family] = fmean(
                factor_ranks[symbol][name] for name in members
            )
    composite_scores = {
        symbol: fmean(family_ranks[symbol][family] for family in FAMILY_NAMES)
        for symbol in raw_by_symbol
    }
    composite_order = sorted(composite_scores, key=lambda s: (-composite_scores[s], s))
    composite_rank = {symbol: index + 1 for index, symbol in enumerate(composite_order)}
    ridge_scores = {
        forecast.symbol: forecast.expected_excess_return_20d
        for forecast in ridge_forecasts
        if forecast.symbol in raw_by_symbol
    }
    ridge_order = sorted(ridge_scores, key=lambda s: (-ridge_scores[s], s))
    ridge_rank = {symbol: index + 1 for index, symbol in enumerate(ridge_order)}
    signals = tuple(
        FactorRankSignal(
            symbol=symbol,
            data_as_of=prices[symbol][0],
            formation_price=prices[symbol][1],
            benchmark_price=prices[symbol][2],
            raw_factors=raw_by_symbol[symbol],
            winsorized_factors=winsorized[symbol],
            factor_percentile_ranks=factor_ranks[symbol],
            family_ranks=family_ranks[symbol],
            composite_score=composite_scores[symbol],
            composite_rank=composite_rank[symbol],
            ridge_score=ridge_scores.get(symbol, 0.0),
            ridge_rank=ridge_rank.get(symbol, len(ridge_order) + 1),
        )
        for symbol in composite_order
    )
    family_series = {
        family: {symbol: family_ranks[symbol][family] for symbol in raw_by_symbol}
        for family in FAMILY_NAMES
    }
    correlations: Dict[str, float] = {}
    overlaps: Dict[str, float] = {}
    for left_index, left in enumerate(FAMILY_NAMES):
        for right in FAMILY_NAMES[left_index + 1:]:
            key = f"{left}|{right}"
            correlations[key] = _spearman(family_series[left], family_series[right])
            overlaps[key] = _top_overlap(
                family_series[left], family_series[right], top_bucket_fraction
            )
    top = tuple(composite_order[:shortlist_size])
    ridge_top = tuple(ridge_order[:shortlist_size])
    overlap = len(set(top) & set(ridge_top)) / max(len(top), 1)
    return FactorRankSnapshot(
        engine_version=ENGINE_VERSION,
        data_as_of=next(iter(dates)),
        signals=signals,
        top_25=top,
        ridge_top_25=ridge_top,
        top_25_overlap=overlap,
        pairwise_family_spearman=correlations,
        pairwise_family_top_bucket_overlap=overlaps,
    )


def evaluate_factor_rank_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Evaluate only resolved point-in-time rows, clustered by formation date."""
    output: Dict[str, Any] = {"engine_version": ENGINE_VERSION, "horizons": {}}
    for horizon in (5, 10, 20):
        outcome_field = f"realized_excess_{horizon}d"
        by_date: Dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            if row.get(outcome_field) is None:
                continue
            by_date.setdefault(str(row["formation_date"]), []).append(row)
        ridge_ics: list[float] = []
        composite_ics: list[float] = []
        family_ics: Dict[str, list[float]] = {name: [] for name in FAMILY_NAMES}
        bucket_values: Dict[str, Dict[str, list[float]]] = {
            "composite": {name: [] for name in ("1_5", "6_10", "11_15", "16_25")},
            "ridge": {name: [] for name in ("1_5", "6_10", "11_15", "16_25")},
        }
        tail_metrics: Dict[str, Dict[str, list[float]]] = {
            model: {
                "top_decile_return": [],
                "bottom_decile_return": [],
                "long_short_spread": [],
                "top_decile_worst_name": [],
                "winner_capture": [],
            }
            for model in ("composite", "ridge")
        }
        for date_rows in by_date.values():
            if len(date_rows) < 10:
                continue
            realized = {
                str(row["symbol"]): float(row[outcome_field]) for row in date_rows
            }
            composite = {
                str(row["symbol"]): -float(row["composite_rank"])
                for row in date_rows
            }
            ridge = {
                str(row["symbol"]): -float(row["ridge_rank"])
                for row in date_rows
            }
            composite_ics.append(_spearman(composite, realized))
            ridge_ics.append(_spearman(ridge, realized))
            for family in FAMILY_NAMES:
                values: Dict[str, float] = {}
                for row in date_rows:
                    try:
                        payload = json.loads(str(row["factors_json"]))
                        value = (payload.get("family_ranks") or {}).get(family)
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        value = None
                    if value is not None:
                        values[str(row["symbol"])] = float(value)
                if len(values) >= 10:
                    family_ics[family].append(_spearman(values, realized))
            for model, rank_field in (
                ("composite", "composite_rank"), ("ridge", "ridge_rank")
            ):
                date_buckets: Dict[str, list[float]] = {
                    name: [] for name in ("1_5", "6_10", "11_15", "16_25")
                }
                for row in date_rows:
                    rank = int(row[rank_field])
                    bucket = (
                        "1_5" if rank <= 5 else
                        "6_10" if rank <= 10 else
                        "11_15" if rank <= 15 else
                        "16_25" if rank <= 25 else None
                    )
                    if bucket:
                        date_buckets[bucket].append(float(row[outcome_field]))
                for bucket, values in date_buckets.items():
                    if values:
                        bucket_values[model][bucket].append(fmean(values))
                count = max(1, math.ceil(len(date_rows) * 0.10))
                predicted = sorted(
                    date_rows, key=lambda row: (int(row[rank_field]), str(row["symbol"]))
                )
                top = predicted[:count]
                bottom = predicted[-count:]
                top_returns = [float(row[outcome_field]) for row in top]
                bottom_returns = [float(row[outcome_field]) for row in bottom]
                realized_winners = {
                    str(row["symbol"])
                    for row in sorted(
                        date_rows,
                        key=lambda row: -float(row[outcome_field]),
                    )[:count]
                }
                predicted_top_25 = {
                    str(row["symbol"])
                    for row in predicted[: min(25, len(predicted))]
                }
                tail_metrics[model]["top_decile_return"].append(fmean(top_returns))
                tail_metrics[model]["bottom_decile_return"].append(
                    fmean(bottom_returns)
                )
                tail_metrics[model]["long_short_spread"].append(
                    fmean(top_returns) - fmean(bottom_returns)
                )
                tail_metrics[model]["top_decile_worst_name"].append(min(top_returns))
                tail_metrics[model]["winner_capture"].append(
                    len(realized_winners & predicted_top_25) / len(realized_winners)
                )
        output["horizons"][str(horizon)] = {
            "independent_date_blocks": len(composite_ics),
            "composite_mean_rank_ic": (
                fmean(composite_ics) if composite_ics else None
            ),
            "ridge_mean_rank_ic": fmean(ridge_ics) if ridge_ics else None,
            "composite_rank_ic_hit_rate": (
                sum(value > 0 for value in composite_ics) / len(composite_ics)
                if composite_ics else None
            ),
            "ridge_rank_ic_hit_rate": (
                sum(value > 0 for value in ridge_ics) / len(ridge_ics)
                if ridge_ics else None
            ),
            "factor_family_mean_rank_ic": {
                family: fmean(values) if values else None
                for family, values in family_ics.items()
            },
            "factor_family_rank_ic_hit_rate": {
                family: (
                    sum(value > 0 for value in values) / len(values)
                    if values else None
                )
                for family, values in family_ics.items()
            },
            "mean_excess_return_by_rank_bucket": {
                model: {
                    bucket: fmean(values) if values else None
                    for bucket, values in buckets.items()
                }
                for model, buckets in bucket_values.items()
            },
            "tail_and_capture": {
                model: {
                    metric: fmean(values) if values else None
                    for metric, values in metrics.items()
                }
                for model, metrics in tail_metrics.items()
            },
        }
    formation_dates = sorted({str(row["formation_date"]) for row in rows})
    turnover: list[float] = []
    previous: set[str] | None = None
    for date in formation_dates:
        current = {
            str(row["symbol"])
            for row in rows
            if str(row["formation_date"]) == date
            and int(row["composite_rank"]) <= 25
        }
        if previous and current:
            turnover.append(1.0 - len(previous & current) / max(len(previous), 1))
        previous = current
    output["top_25_turnover"] = fmean(turnover) if turnover else None
    output["top_25_stability"] = (
        1.0 - fmean(turnover) if turnover else None
    )
    output["signal_dates"] = len(formation_dates)
    output["rows"] = len(rows)
    return output
