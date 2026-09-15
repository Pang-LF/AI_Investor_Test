from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Dict, Mapping, Sequence

from .config import Settings
from .forecasting import (
    FEATURE_NAMES,
    _aligned_series,
    _date_block_positive_probability,
    _features,
    _quantile,
    _raw_positive_probability,
    _rolling_beta,
    build_training_samples,
    fit_ridge_model,
    history_integrity_issues,
)
from .market_data import DailyBar, HistoricalCache, get_daily_histories
from .robinhood_mcp import RobinhoodMCPClient


@dataclass(frozen=True)
class AuditDefinition:
    signal_days: int = 60
    winner_quantile: float = 0.90
    minimum_excess_return_5d: float = 0.05
    minimum_excess_return_20d: float = 0.08
    history_calendar_days: int = 550


def _universe_symbols(payload: Mapping[str, Any]) -> tuple[list[str], set[str]]:
    broad: list[str] = []
    for key in (
        "large_candidates", "mid_candidates", "small_candidates",
        "event_candidates",
    ):
        for item in payload.get(key) or []:
            symbol = str(item.get("symbol") or "").upper()
            if symbol and symbol not in broad:
                broad.append(symbol)
    for item in payload.get("entries") or []:
        symbol = str(item.get("symbol") or "").upper()
        if item.get("investable", True) and symbol and symbol not in broad:
            broad.append(symbol)
    active = {
        str(item.get("symbol") or "").upper()
        for item in payload.get("entries") or []
        if item.get("symbol") and item.get("investable", True)
    }
    if "SPY" not in broad:
        broad.append("SPY")
    active.add("SPY")
    return broad, active


def _universe_metadata(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for key, bucket in (
        ("large_candidates", "large"),
        ("mid_candidates", "mid"),
        ("small_candidates", "small"),
        ("event_candidates", "event"),
    ):
        for item in payload.get(key) or []:
            symbol = str(item.get("symbol") or "").upper()
            if not symbol:
                continue
            existing = metadata.get(symbol, {})
            metadata[symbol] = {
                "bucket": existing.get("bucket", bucket),
                "sector": str(
                    item.get("sector") or existing.get("sector") or "unknown"
                ),
                "average_dollar_volume": float(
                    item.get("average_dollar_volume")
                    or existing.get("average_dollar_volume")
                    or 0.0
                ),
            }
    for item in payload.get("entries") or []:
        if not item.get("investable", True):
            continue
        symbol = str(item.get("symbol") or "").upper()
        if not symbol:
            continue
        existing = metadata.get(symbol, {})
        metadata[symbol] = {
            "bucket": existing.get("bucket") or str(item.get("bucket") or "other"),
            "sector": str(
                item.get("sector") or existing.get("sector") or "unknown"
            ),
            "average_dollar_volume": float(
                item.get("average_dollar_volume")
                or existing.get("average_dollar_volume")
                or 0.0
            ),
        }
    return metadata


def fetch_audit_histories(
    root: Path, settings: Settings, definition: AuditDefinition
) -> tuple[Dict[str, list[DailyBar]], Dict[str, Any]]:
    universe = json.loads((root / ".local/state/universe.json").read_text())
    broad, active = _universe_symbols(universe)
    required_calls = math.ceil(len(broad) / 10)
    if required_calls > 48:
        raise RuntimeError(f"Audit universe needs too many broker calls: {required_calls}")
    with RobinhoodMCPClient(
        max_calls=required_calls + 2,
        allowed_tools={"get_equity_historicals"},
        allow_order_submission=False,
    ) as client:
        histories = get_daily_histories(
            client,
            HistoricalCache(root / ".local/state/audits/historicals"),
            broad,
            str(universe["trading_date"]),
            definition.history_calendar_days,
            settings.forecast.min_history_bars,
        )
    return histories, {
        "universe_date": universe["trading_date"],
        "broad_static_symbols": len(broad) - 1,
        "active_static_symbols": len(active) - 1,
        "historical_calls": required_calls,
    }


def _forward_excess(
    closes: Sequence[float], benchmark: Sequence[float], index: int, horizon: int
) -> float:
    beta = _rolling_beta(closes, benchmark, index)
    return sum(
        closes[future] / closes[future - 1] - 1.0
        - beta * (benchmark[future] / benchmark[future - 1] - 1.0)
        for future in range(index + 1, index + horizon + 1)
    )


def _event_proxy(closes: Sequence[float], volumes: Sequence[float], index: int) -> bool:
    if index < 20 or closes[index - 1] <= 0:
        return False
    daily_return = closes[index] / closes[index - 1] - 1.0
    average_volume = fmean(volumes[index - 20:index])
    relative_volume = volumes[index] / average_volume if average_volume > 0 else 0.0
    return abs(daily_return) >= 0.04 or (
        abs(daily_return) >= 0.02 and relative_volume >= 2.0
    )


def _live_funnel_summary(root: Path) -> Dict[str, Any]:
    dispositions: Counter[str] = Counter()
    decision_runs = 0
    for path in sorted((root / "logs/decisions").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            decision_runs += 1
            for row in payload.get("candidate_funnel") or []:
                dispositions[str(row.get("final_disposition") or "unknown")] += 1
    return {
        "decision_runs": decision_runs,
        "dispositions": dict(sorted(dispositions.items())),
        "limitation": "Only actual logged runs contain historical LLM and optimizer outcomes.",
    }


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"sample_size": 0}
    average = fmean(ordered)
    variance = fmean((value - average) ** 2 for value in ordered)
    return {
        "sample_size": len(ordered),
        "mean": average,
        "median": median(ordered),
        "win_rate": fmean(value > 0 for value in ordered),
        "p10": _quantile(ordered, 0.10),
        "p25": _quantile(ordered, 0.25),
        "p75": _quantile(ordered, 0.75),
        "p90": _quantile(ordered, 0.90),
        "worst": ordered[0],
        "standard_deviation": math.sqrt(variance),
    }


def _forecast_bucket(rank_fraction: float) -> str:
    if rank_fraction <= 0.05:
        return "top_5pct"
    if rank_fraction <= 0.10:
        return "pct_5_10"
    if rank_fraction <= 0.20:
        return "pct_10_20"
    if rank_fraction <= 0.40:
        return "pct_20_40"
    if rank_fraction <= 0.60:
        return "pct_40_60"
    return "bottom_40pct"


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 3:
        return 0.0
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = fmean(left_ranks)
    right_mean = fmean(right_ranks)
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks))
    if left_scale <= 0 or right_scale <= 0:
        return 0.0
    return covariance / (left_scale * right_scale)


def _block_bootstrap_mean_interval(
    dated_values: Sequence[tuple[str, float]], block_length: int, samples: int = 1000
) -> tuple[float, float]:
    values = [value for _, value in sorted(dated_values)]
    if not values:
        return (0.0, 0.0)
    rng = random.Random(20260915 + block_length)
    bootstrapped = []
    for _ in range(samples):
        draw: list[float] = []
        while len(draw) < len(values):
            start = rng.randrange(len(values))
            draw.extend(
                values[(start + offset) % len(values)]
                for offset in range(block_length)
            )
        bootstrapped.append(fmean(draw[:len(values)]))
    return (_quantile(bootstrapped, 0.025), _quantile(bootstrapped, 0.975))


def _rank_ic_summary(
    rows_by_date: Mapping[str, Sequence[Mapping[str, Any]]], horizon: int
) -> Dict[str, Any]:
    dated = []
    by_month: Dict[str, list[float]] = defaultdict(list)
    for date, rows in sorted(rows_by_date.items()):
        usable = [row for row in rows if row.get("raw_expected_20d") is not None]
        if len(usable) < 3:
            continue
        value = _spearman(
            [float(row["raw_expected_20d"]) for row in usable],
            [float(row[f"realized_excess_{horizon}d"]) for row in usable],
        )
        dated.append((date, value))
        by_month[date[:7]].append(value)
    values = [value for _, value in dated]
    nonoverlapping = values[::horizon]
    return {
        "date_blocks": len(values),
        "mean": fmean(values) if values else 0.0,
        "median": median(values) if values else 0.0,
        "hit_rate": fmean(value > 0 for value in values) if values else 0.0,
        "block_bootstrap_95_interval": _block_bootstrap_mean_interval(
            dated, min(horizon, max(1, len(values)))
        ),
        "nonoverlapping_blocks": len(nonoverlapping),
        "nonoverlapping_mean": fmean(nonoverlapping) if nonoverlapping else 0.0,
        "by_month": {
            month: {
                "date_blocks": len(month_values),
                "mean": fmean(month_values),
                "hit_rate": fmean(value > 0 for value in month_values),
            }
            for month, month_values in sorted(by_month.items())
        },
    }


def _stratified_sample(
    candidates: Sequence[str],
    active: Sequence[str],
    strata: Mapping[str, tuple[Any, ...]],
    rng: random.Random,
) -> set[str]:
    available = set(candidates)
    sampled: set[str] = set()
    groups: Dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for symbol in candidates:
        groups[strata[symbol]].append(symbol)
    for symbol in active:
        choices = [candidate for candidate in groups[strata[symbol]] if candidate in available]
        if not choices:
            choices = sorted(available)
        selected = rng.choice(choices)
        sampled.add(selected)
        available.remove(selected)
    return sampled


def _matched_universe_benchmarks(
    all_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    active_symbols: set[str],
    horizon: int,
    simulations: int = 1000,
) -> Dict[str, Any]:
    available_symbols = sorted({
        str(row["symbol"]) for row in all_rows if str(row["symbol"]) in metadata
    })
    active = sorted((active_symbols - {"SPY"}) & set(available_symbols))
    liquidities = sorted(
        float(metadata[symbol].get("average_dollar_volume") or 0.0)
        for symbol in available_symbols
    )

    def liquidity_decile(symbol: str) -> int:
        value = float(metadata[symbol].get("average_dollar_volume") or 0.0)
        if value <= 0 or not liquidities:
            return -1
        percentile = sum(candidate <= value for candidate in liquidities) / len(liquidities)
        return min(9, int(percentile * 10))

    dimensions = {
        "random": lambda symbol: ("all",),
        "market_cap_bucket_matched": lambda symbol: (
            metadata[symbol].get("bucket", "unknown"),
        ),
        "sector_matched": lambda symbol: (
            metadata[symbol].get("sector", "unknown"),
        ),
        "liquidity_matched": lambda symbol: (liquidity_decile(symbol),),
        "market_cap_sector_liquidity_matched": lambda symbol: (
            metadata[symbol].get("bucket", "unknown"),
            metadata[symbol].get("sector", "unknown"),
            liquidity_decile(symbol),
        ),
    }
    winners = [row for row in all_rows if row[f"winner_{horizon}d"]]
    denominator = len(winners)
    observed = (
        sum(str(row["symbol"]) in active for row in winners) / denominator
        if denominator else 0.0
    )
    output: Dict[str, Any] = {
        "active_investable_symbols": len(active),
        "eligible_current_snapshot_symbols": len(available_symbols),
        "observed_capture": observed,
        "market_cap_limitation": (
            "Exact point-in-time market cap is unavailable; large/mid/small scanner "
            "membership is used as the market-cap bucket proxy."
        ),
    }
    for name, key_function in dimensions.items():
        strata = {symbol: key_function(symbol) for symbol in available_symbols}
        rng = random.Random(20260915 + horizon + sum(map(ord, name)))
        captures = []
        for _ in range(simulations):
            selected = _stratified_sample(available_symbols, active, strata, rng)
            captures.append(
                sum(str(row["symbol"]) in selected for row in winners) / denominator
                if denominator else 0.0
            )
        baseline = fmean(captures) if captures else 0.0
        output[name] = {
            "simulations": simulations,
            "mean_capture": baseline,
            "p05": _quantile(captures, 0.05),
            "p95": _quantile(captures, 0.95),
            "observed_lift": observed / baseline if baseline else 0.0,
        }
    return output


def _forecast_bucket_analysis(
    all_rows: Sequence[Mapping[str, Any]], horizon: int
) -> Dict[str, Any]:
    order = (
        "top_5pct", "pct_5_10", "pct_10_20",
        "pct_20_40", "pct_40_60", "bottom_40pct",
    )
    output: Dict[str, Any] = {}
    for bucket in order:
        rows = [row for row in all_rows if row.get("forecast_bucket") == bucket]
        output[bucket] = {
            "average_predicted_alpha_20d": (
                fmean(float(row["raw_expected_20d"]) for row in rows)
                if rows else 0.0
            ),
            "realized_excess": _distribution(
                [float(row[f"realized_excess_{horizon}d"]) for row in rows]
            ),
        }
    return output


def _extreme_forecast_autopsy(
    all_rows: Sequence[Mapping[str, Any]], horizon: int
) -> Dict[str, Any]:
    groups = {"top_5pct": "top_5pct", "pct_10_20": "pct_10_20"}
    output: Dict[str, Any] = {}
    for name, bucket in groups.items():
        rows = [row for row in all_rows if row.get("forecast_bucket") == bucket]
        numeric_fields = (
            "feature_l2_distance",
            "maximum_absolute_feature_zscore",
            "trailing_volatility_20d",
            "maximum_absolute_return_5d",
            "event_days_20d",
            "average_dollar_volume_20d",
        )
        feature_values = {
            feature: _distribution([
                float((row.get("features") or {}).get(feature, 0.0)) for row in rows
            ])
            for feature in FEATURE_NAMES
        }
        zscore_values = {
            feature: _distribution([
                float((row.get("feature_zscores") or {}).get(feature, 0.0))
                for row in rows
            ])
            for feature in FEATURE_NAMES
        }
        output[name] = {
            "realized_excess": _distribution([
                float(row[f"realized_excess_{horizon}d"]) for row in rows
            ]),
            "numeric_diagnostics": {
                field: _distribution([float(row.get(field, 0.0)) for row in rows])
                for field in numeric_fields
            },
            "features": feature_values,
            "feature_zscores": zscore_values,
            "universe_buckets": dict(sorted(Counter(
                str(row.get("universe_bucket") or "unknown") for row in rows
            ).items())),
            "sectors": dict(Counter(
                str(row.get("sector") or "unknown") for row in rows
            ).most_common()),
        }
    by_date: Dict[str, Dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in all_rows:
        bucket = str(row.get("forecast_bucket") or "")
        if bucket in groups.values():
            by_date[str(row["date"])][bucket].append(
                float(row[f"realized_excess_{horizon}d"])
            )
    dated_differences = []
    for date, buckets in sorted(by_date.items()):
        if buckets["top_5pct"] and buckets["pct_10_20"]:
            dated_differences.append((
                date,
                fmean(buckets["pct_10_20"]) - fmean(buckets["top_5pct"]),
            ))
    differences = [value for _, value in dated_differences]
    nonoverlapping = differences[::horizon]
    output["date_clustered_pct_10_20_minus_top_5"] = {
        "date_blocks": len(differences),
        "mean_difference": fmean(differences) if differences else 0.0,
        "block_bootstrap_95_interval": _block_bootstrap_mean_interval(
            dated_differences, min(horizon, max(1, len(differences)))
        ),
        "nonoverlapping_blocks": len(nonoverlapping),
        "nonoverlapping_mean_difference": (
            fmean(nonoverlapping) if nonoverlapping else 0.0
        ),
    }
    return output


def _conditional_probability_analysis(
    all_rows: Sequence[Mapping[str, Any]], horizon: int
) -> Dict[str, Any]:
    rank_groups = {
        "top_10pct": {"top_5pct", "pct_5_10"},
        "pct_10_20": {"pct_10_20"},
        "pct_20_40": {"pct_20_40"},
        "bottom_60pct": {"pct_40_60", "bottom_40pct"},
    }

    def probability_band(value: float) -> str:
        if value < 0.50:
            return "below_50pct"
        if value < 0.55:
            return "pct_50_55"
        if value < 0.60:
            return "pct_55_60"
        return "at_least_60pct"

    output: Dict[str, Any] = {}
    for rank_name, buckets in rank_groups.items():
        selected = [
            row for row in all_rows
            if row.get("forecast_bucket") in buckets
            and row.get("calibrated_probability") is not None
        ]
        output[rank_name] = {}
        for band in ("below_50pct", "pct_50_55", "pct_55_60", "at_least_60pct"):
            rows = [
                row for row in selected
                if probability_band(float(row["calibrated_probability"])) == band
            ]
            output[rank_name][band] = _distribution([
                float(row[f"realized_excess_{horizon}d"]) for row in rows
            ])
    return output


def _funnel_return_attribution(
    all_rows: Sequence[Mapping[str, Any]], horizon: int
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for stage in sorted({str(row.get("quant_stage")) for row in all_rows}):
        rows = [row for row in all_rows if row.get("quant_stage") == stage]
        output[stage] = _distribution([
            float(row[f"realized_excess_{horizon}d"]) for row in rows
        ])
    return output


def _research_depth_analysis(
    all_rows: Sequence[Mapping[str, Any]], horizon: int
) -> Dict[str, Any]:
    cohorts = {
        "rank_1_3": (1, 3),
        "rank_4_5": (4, 5),
        "rank_6_10": (6, 10),
        "rank_11_15": (11, 15),
        "rank_16_20": (16, 20),
        "rank_21_plus": (21, 10 ** 9),
    }
    cohort_output = {}
    for name, (lower, upper) in cohorts.items():
        rows = [
            row for row in all_rows
            if row.get("research_rank") is not None
            and lower <= int(row["research_rank"]) <= upper
        ]
        cohort_output[name] = _distribution([
            float(row[f"realized_excess_{horizon}d"]) for row in rows
        ])
    counterfactual = {}
    for depth in (3, 5, 10, 15, 20):
        rows = [
            row for row in all_rows
            if row.get("research_rank") is not None
            and int(row["research_rank"]) <= depth
        ]
        counterfactual[f"top_{depth}"] = _distribution([
            float(row[f"realized_excess_{horizon}d"]) for row in rows
        ])
    return {"cohorts": cohort_output, "cumulative_depths": counterfactual}


def run_opportunity_capture_audit(
    root: Path,
    settings: Settings,
    histories: Mapping[str, Sequence[DailyBar]],
    definition: AuditDefinition,
    source_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    universe = json.loads((root / ".local/state/universe.json").read_text())
    broad_symbols, active_symbols = _universe_symbols(universe)
    universe_metadata = _universe_metadata(universe)
    integrity = history_integrity_issues(
        histories, settings.forecast.max_daily_close_ratio
    )
    if "SPY" in integrity:
        raise RuntimeError("Audit benchmark failed history integrity checks")
    clean = {symbol: bars for symbol, bars in histories.items() if symbol not in integrity}
    benchmark_bars = clean["SPY"]
    benchmark_dates = [bar.begins_at[:10] for bar in benchmark_bars]
    eligible_dates = benchmark_dates[
        settings.forecast.min_history_bars:-20
    ]
    signal_dates = eligible_dates[-definition.signal_days:]
    if len(signal_dates) < definition.signal_days:
        raise RuntimeError("Insufficient completed dates for a 60-day audit")
    audit_start_index = benchmark_dates.index(signal_dates[0])
    training_cutoff = benchmark_dates[audit_start_index - 20]
    active_histories = {
        symbol: bars for symbol, bars in clean.items() if symbol in active_symbols
    }
    training = [
        sample
        for sample in build_training_samples(
            active_histories, 20, settings.forecast.max_daily_close_ratio
        )
        if sample[0] < training_cutoff
    ]
    model = fit_ridge_model(
        training,
        20,
        settings.forecast.ridge_penalty,
        settings.forecast.min_training_samples,
        settings.forecast.shrinkage,
        settings.forecast.minimum_bias_correction_date_blocks,
        settings.forecast.bias_bootstrap_samples,
    )

    rows_by_date: Dict[str, list[Dict[str, Any]]] = {date: [] for date in signal_dates}
    for symbol in broad_symbols:
        bars = clean.get(symbol)
        if not bars or symbol == "SPY":
            continue
        dates, closes, volumes, benchmark = _aligned_series(bars, benchmark_bars)
        positions = {date: index for index, date in enumerate(dates)}
        for date in signal_dates:
            index = positions.get(date)
            if index is None or index < settings.forecast.min_history_bars or index + 20 >= len(dates):
                continue
            row: Dict[str, Any] = {
                "date": date,
                "symbol": symbol,
                "in_active_investable_universe": symbol in active_symbols,
                "event_proxy": _event_proxy(closes, volumes, index),
                "realized_excess_5d": _forward_excess(closes, benchmark, index, 5),
                "realized_excess_20d": _forward_excess(closes, benchmark, index, 20),
            }
            if symbol in active_symbols:
                features = _features(closes, volumes, benchmark, index)
                feature_zscores = tuple(
                    (value - feature_mean) / feature_scale
                    for value, feature_mean, feature_scale in zip(
                        features, model.feature_means, model.feature_scales
                    )
                )
                model_prediction = model.predict(features)
                raw_expected = settings.forecast.shrinkage * model_prediction
                uncapped_adjusted = raw_expected + model.applied_validation_bias
                adjusted = max(
                    -settings.forecast.max_abs_forecast_20d,
                    min(uncapped_adjusted, settings.forecast.max_abs_forecast_20d),
                )
                calibrated_probability = _date_block_positive_probability(
                    raw_expected,
                    model.validation_errors,
                    model.validation_dates,
                    settings.forecast.calibration_prior_date_blocks,
                )
                row.update({
                    "model_prediction_20d": model_prediction,
                    "raw_expected_20d": raw_expected,
                    "adjusted_expected_20d": adjusted,
                    "uncertainty_20d": model.residual_std,
                    "raw_probability": _raw_positive_probability(
                        raw_expected, model.residual_std
                    ),
                    "calibrated_probability": calibrated_probability,
                    "edge_ratio": adjusted / max(model.residual_std, 1e-9),
                    "features": dict(zip(FEATURE_NAMES, features)),
                    "feature_zscores": dict(zip(FEATURE_NAMES, feature_zscores)),
                    "feature_l2_distance": math.sqrt(
                        sum(value * value for value in feature_zscores)
                    ),
                    "maximum_absolute_feature_zscore": max(
                        abs(value) for value in feature_zscores
                    ),
                    "trailing_volatility_20d": math.sqrt(
                        252
                        * fmean(
                            (
                                closes[position] / closes[position - 1] - 1.0
                            ) ** 2
                            for position in range(index - 19, index + 1)
                        )
                    ),
                    "maximum_absolute_return_5d": max(
                        abs(closes[position] / closes[position - 1] - 1.0)
                        for position in range(index - 4, index + 1)
                    ),
                    "event_days_20d": sum(
                        _event_proxy(closes, volumes, position)
                        for position in range(index - 19, index + 1)
                    ),
                    "average_dollar_volume_20d": fmean(
                        closes[position] * volumes[position]
                        for position in range(index - 19, index + 1)
                    ),
                    "universe_bucket": universe_metadata.get(symbol, {}).get(
                        "bucket", "unknown"
                    ),
                    "sector": universe_metadata.get(symbol, {}).get(
                        "sector", "unknown"
                    ),
                })
            rows_by_date[date].append(row)

    all_rows: list[Dict[str, Any]] = []
    for date, rows in rows_by_date.items():
        for horizon, minimum in ((5, definition.minimum_excess_return_5d),
                                 (20, definition.minimum_excess_return_20d)):
            key = f"realized_excess_{horizon}d"
            threshold = max(
                minimum,
                _quantile([float(row[key]) for row in rows], definition.winner_quantile),
            )
            for row in rows:
                row[f"winner_{horizon}d"] = float(row[key]) >= threshold
                row[f"winner_threshold_{horizon}d"] = threshold
        forecast_ranked = sorted(
            (
                row for row in rows
                if row["in_active_investable_universe"]
                and row.get("raw_expected_20d") is not None
            ),
            key=lambda item: item["raw_expected_20d"],
            reverse=True,
        )
        for rank, row in enumerate(forecast_ranked, 1):
            row["daily_forecast_rank"] = rank
            row["daily_forecast_rank_fraction"] = (
                (rank - 0.5) / len(forecast_ranked)
            )
            row["forecast_bucket"] = _forecast_bucket(
                row["daily_forecast_rank_fraction"]
            )
        qualified = [
            row for row in rows
            if row["in_active_investable_universe"] and (
                row["event_proxy"] or (
                    row.get("raw_expected_20d", -math.inf)
                    >= settings.forecast.research_min_raw_expected_excess_return_20d
                    and row.get("raw_probability", 0.0)
                    >= settings.forecast.research_min_raw_probability_positive
                )
            )
        ]
        ranked_qualified = sorted(
            qualified,
            key=lambda item: (
                item.get("raw_expected_20d", -math.inf)
                / max(item.get("uncertainty_20d", 1e-9), 1e-9),
                item.get("raw_expected_20d", -math.inf),
            ),
            reverse=True,
        )
        research_ranks = {
            row["symbol"]: rank
            for rank, row in enumerate(ranked_qualified, 1)
        }
        selected = {
            row["symbol"]
            for row in ranked_qualified[:settings.forecast.research_candidate_count]
        }
        for row in rows:
            if not row["in_active_investable_universe"]:
                stage = "outside_active_investable_universe"
            elif row["symbol"] not in selected:
                stage = (
                    "research_capacity"
                    if row in qualified else "quant_research_gate"
                )
            elif model.validation_date_blocks < settings.forecast.execution_min_calibration_date_blocks:
                stage = "insufficient_calibration"
            elif row["adjusted_expected_20d"] <= settings.forecast.execution_min_bias_adjusted_excess_return_20d:
                stage = "investment_alpha_gate"
            elif row["calibrated_probability"] < settings.forecast.execution_min_calibrated_probability_positive:
                stage = "investment_probability_gate"
            elif row["edge_ratio"] < settings.forecast.execution_min_edge_ratio_20d:
                stage = "investment_edge_gate"
            else:
                stage = "quant_pipeline_pass"
            row["quant_stage"] = stage
            row["selected_for_research_capacity"] = row["symbol"] in selected
            row["research_rank"] = research_ranks.get(row["symbol"])
            all_rows.append(row)

    horizon_summaries: Dict[str, Any] = {}
    for horizon in (5, 20):
        winners = [row for row in all_rows if row[f"winner_{horizon}d"]]
        active_winners = [
            row for row in winners if row["in_active_investable_universe"]
        ]
        stage_counts = Counter(row["quant_stage"] for row in winners)
        event_count = sum(bool(row["event_proxy"]) for row in winners)
        passed = [row for row in winners if row["quant_stage"] == "quant_pipeline_pass"]
        all_passed = [
            row for row in all_rows if row["quant_stage"] == "quant_pipeline_pass"
        ]
        active_rows = [
            row for row in all_rows if row["in_active_investable_universe"]
        ]
        active_winner_rate = len(active_winners) / len(active_rows) if active_rows else 0.0
        pass_precision = len(passed) / len(all_passed) if all_passed else 0.0
        largest_by_symbol: Dict[str, Dict[str, Any]] = {}
        for row in winners:
            if row["quant_stage"] == "quant_pipeline_pass":
                continue
            prior = largest_by_symbol.get(row["symbol"])
            if prior is None or row[f"realized_excess_{horizon}d"] > prior[f"realized_excess_{horizon}d"]:
                largest_by_symbol[row["symbol"]] = row
        horizon_summaries[f"{horizon}d"] = {
            "winner_opportunities": len(winners),
            "unique_winner_symbols": len({row["symbol"] for row in winners}),
            "static_active_pool_winner_opportunities": len(active_winners),
            "static_active_pool_coverage": (
                len(active_winners) / len(winners) if winners else 0.0
            ),
            "event_proxy_detected": event_count,
            "event_proxy_recall": event_count / len(winners) if winners else 0.0,
            "quant_pipeline_passed": len(passed),
            "quant_pipeline_recall": len(passed) / len(winners) if winners else 0.0,
            "conditional_quant_recall_inside_static_active_pool": (
                len(passed) / len(active_winners) if active_winners else 0.0
            ),
            "quant_pass_opportunities": len(all_passed),
            "quant_pass_winner_precision": pass_precision,
            "active_pool_winner_base_rate": active_winner_rate,
            "winner_precision_lift": (
                pass_precision / active_winner_rate if active_winner_rate else 0.0
            ),
            "quant_pass_mean_realized_excess": (
                fmean(row[f"realized_excess_{horizon}d"] for row in all_passed)
                if all_passed else 0.0
            ),
            "quant_pass_positive_rate": (
                fmean(row[f"realized_excess_{horizon}d"] > 0 for row in all_passed)
                if all_passed else 0.0
            ),
            "stage_counts": dict(sorted(stage_counts.items())),
            "largest_missed": sorted(
                (
                    {
                        "date": row["date"], "symbol": row["symbol"],
                        "realized_excess": row[f"realized_excess_{horizon}d"],
                        "stage": row["quant_stage"],
                        "event_proxy": row["event_proxy"],
                        "raw_expected_20d": row.get("raw_expected_20d"),
                        "calibrated_probability": row.get("calibrated_probability"),
                        "edge_ratio": row.get("edge_ratio"),
                    }
                    for row in largest_by_symbol.values()
                ),
                key=lambda item: item["realized_excess"],
                reverse=True,
            )[:20],
        }

    return {
        "audit_type": "static-universe_purged_oos_opportunity_capture",
        "definition": asdict(definition),
        "source": dict(source_metadata),
        "limitations": [
            "Broad and active universes are the current scanner snapshot, not historical constituents.",
            "This has survivorship and selection bias and is not a performance backtest.",
            "Exact historical market caps are unavailable; scanner size buckets are the market-cap proxy.",
            "Daily event_proxy is not the production intraday event scanner.",
            "Overlapping signal dates count repeated opportunities and are not independent events.",
            "Historical news, LLM verdicts, portfolio state, and optimizer decisions are unavailable before logging began.",
        ],
        "signal_window": {
            "first": signal_dates[0], "last": signal_dates[-1],
            "training_cutoff_exclusive": training_cutoff,
            "purge_days": 20,
        },
        "training": {
            "samples": len(training),
            "validation_date_blocks": model.validation_date_blocks,
            "uncertainty_20d_decimal": model.residual_std,
            "applied_validation_bias_decimal": model.applied_validation_bias,
        },
        "history_integrity_quarantine": integrity,
        "horizons": horizon_summaries,
        "alpha_attribution": {
            f"{horizon}d": {
                "forecast_buckets": _forecast_bucket_analysis(
                    all_rows, horizon
                ),
                "extreme_forecast_autopsy": _extreme_forecast_autopsy(
                    all_rows, horizon
                ),
                "conditional_probability": _conditional_probability_analysis(
                    all_rows, horizon
                ),
                "funnel_returns": _funnel_return_attribution(
                    all_rows, horizon
                ),
                "research_depth": _research_depth_analysis(
                    all_rows, horizon
                ),
                "rank_ic": _rank_ic_summary(rows_by_date, horizon),
                "matched_universe_benchmarks": _matched_universe_benchmarks(
                    all_rows, universe_metadata, active_symbols, horizon
                ),
            }
            for horizon in (5, 20)
        },
        "actual_logged_funnel": _live_funnel_summary(root),
        "rows": all_rows,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Opportunity Capture Audit",
        "",
        f"Signal window: {report['signal_window']['first']} through {report['signal_window']['last']}",
        "",
        "This is a diagnostic of opportunity coverage, not a return backtest. The broad and",
        "active universes are frozen from the current scanner snapshot, so results contain",
        "survivorship and selection bias. Repeated overlapping signal dates are opportunity",
        "observations, not independent events.",
        "",
        "## Funnel results",
        "",
        "| Horizon | Winner observations | Active-investable coverage | Conditional quant recall | Pass precision/lift | Pass mean excess |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for horizon in ("5d", "20d"):
        item = report["horizons"][horizon]
        lines.append(
            f"| {horizon} | {item['winner_opportunities']} | "
            f"{item['static_active_pool_coverage']:.1%} | "
            f"{item['conditional_quant_recall_inside_static_active_pool']:.1%} | "
            f"{item['quant_pass_winner_precision']:.1%} / {item['winner_precision_lift']:.2f}x | "
            f"{item['quant_pass_mean_realized_excess']:.2%} |"
        )
    lines.extend([
        "",
        "Active-investable coverage measures discovery, while conditional quant recall measures",
        "what the model/funnel retained after a name was already inside the active pool.",
        "Pass precision alone is not profitability: the realized return distribution also matters.",
        "",
        "| Horizon | Event-proxy recall | Quant passes | Positive rate | Active-pool base rate |",
        "|---|---:|---:|---:|---:|",
    ])
    for horizon in ("5d", "20d"):
        item = report["horizons"][horizon]
        lines.append(
            f"| {horizon} | {item['event_proxy_recall']:.1%} | "
            f"{item['quant_pass_opportunities']} | "
            f"{item['quant_pass_positive_rate']:.1%} | "
            f"{item['active_pool_winner_base_rate']:.1%} |"
        )
    attribution = report["alpha_attribution"]
    lines.extend([
        "",
        "## Forecast ranking and economic attribution",
        "",
        "Daily cross-sectional 20d forecast buckets:",
        "",
        "| Forecast bucket | N | Avg predicted | Avg realized 20d | Median | Win rate | P10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    bucket_labels = {
        "top_5pct": "Top 5%", "pct_5_10": "5-10%",
        "pct_10_20": "10-20%", "pct_20_40": "20-40%",
        "pct_40_60": "40-60%", "bottom_40pct": "Bottom 40%",
    }
    for key, label in bucket_labels.items():
        item = attribution["20d"]["forecast_buckets"][key]
        distribution = item["realized_excess"]
        lines.append(
            f"| {label} | {distribution['sample_size']} | "
            f"{item['average_predicted_alpha_20d']:.2%} | "
            f"{distribution['mean']:.2%} | {distribution['median']:.2%} | "
            f"{distribution['win_rate']:.1%} | {distribution['p10']:.2%} |"
        )
    lines.extend([
        "",
        "Research-rank cohorts:",
        "",
        "| Rank cohort | N | Avg realized 5d | Avg realized 20d | 20d win rate | 20d worst |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    cohort_labels = {
        "rank_1_3": "1-3", "rank_4_5": "4-5", "rank_6_10": "6-10",
        "rank_11_15": "11-15", "rank_16_20": "16-20",
        "rank_21_plus": "21+",
    }
    for key, label in cohort_labels.items():
        five = attribution["5d"]["research_depth"]["cohorts"][key]
        twenty = attribution["20d"]["research_depth"]["cohorts"][key]
        if not twenty.get("sample_size"):
            continue
        lines.append(
            f"| {label} | {twenty['sample_size']} | {five['mean']:.2%} | "
            f"{twenty['mean']:.2%} | {twenty['win_rate']:.1%} | "
            f"{twenty['worst']:.2%} |"
        )
    lines.extend([
        "",
        "Funnel return attribution:",
        "",
        "| Stage | N | Avg realized 20d | Median | Win rate | P10 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for stage, distribution in attribution["20d"]["funnel_returns"].items():
        if not distribution.get("sample_size"):
            continue
        lines.append(
            f"| {stage} | {distribution['sample_size']} | "
            f"{distribution['mean']:.2%} | {distribution['median']:.2%} | "
            f"{distribution['win_rate']:.1%} | {distribution['p10']:.2%} |"
        )
    lines.extend([
        "",
        "Rank IC (daily Spearman correlation):",
        "",
        "| Horizon | Date blocks | Mean IC | Median IC | IC hit rate | Block-bootstrap 95% CI | Non-overlap mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for horizon in ("5d", "20d"):
        item = attribution[horizon]["rank_ic"]
        interval = item["block_bootstrap_95_interval"]
        lines.append(
            f"| {horizon} | {item['date_blocks']} | {item['mean']:.3f} | "
            f"{item['median']:.3f} | {item['hit_rate']:.1%} | "
            f"[{interval[0]:.3f}, {interval[1]:.3f}] | "
            f"{item['nonoverlapping_mean']:.3f} ({item['nonoverlapping_blocks']} blocks) |"
        )
    lines.extend([
        "",
        "Matched-universe capture benchmarks:",
        "",
        "| Horizon | Benchmark | Mean capture | 5-95% range | Observed capture | Observed lift |",
        "|---|---|---:|---:|---:|---:|",
    ])
    benchmark_labels = {
        "random": "Random",
        "market_cap_bucket_matched": "Market-cap bucket",
        "sector_matched": "Sector",
        "liquidity_matched": "Liquidity",
        "market_cap_sector_liquidity_matched": "Size + sector + liquidity",
    }
    for horizon in ("5d", "20d"):
        benchmarks = attribution[horizon]["matched_universe_benchmarks"]
        for key, label in benchmark_labels.items():
            item = benchmarks[key]
            lines.append(
                f"| {horizon} | {label} | {item['mean_capture']:.1%} | "
                f"{item['p05']:.1%}-{item['p95']:.1%} | "
                f"{benchmarks['observed_capture']:.1%} | {item['observed_lift']:.2f}x |"
            )
    lines.extend([
        "",
        "Conditional calibrated-probability results within forecast-rank bands (20d):",
        "",
        "| Forecast rank | Probability band | N | Avg realized | Median | Win rate |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for rank_group, bands in attribution["20d"]["conditional_probability"].items():
        for probability_band, distribution in bands.items():
            if not distribution.get("sample_size"):
                continue
            lines.append(
                f"| {rank_group} | {probability_band} | "
                f"{distribution['sample_size']} | {distribution['mean']:.2%} | "
                f"{distribution['median']:.2%} | {distribution['win_rate']:.1%} |"
            )
    autopsy = attribution["20d"]["extreme_forecast_autopsy"]
    lines.extend([
        "",
        "Extreme-forecast autopsy overview:",
        "",
        "| Metric | Top 5% mean | 10-20% mean |",
        "|---|---:|---:|",
    ])
    for field in (
        "feature_l2_distance", "maximum_absolute_feature_zscore",
        "trailing_volatility_20d", "maximum_absolute_return_5d",
        "event_days_20d", "average_dollar_volume_20d",
    ):
        top = autopsy["top_5pct"]["numeric_diagnostics"][field]["mean"]
        middle = autopsy["pct_10_20"]["numeric_diagnostics"][field]["mean"]
        lines.append(f"| {field} | {top:.4g} | {middle:.4g} |")
    bucket_difference = autopsy["date_clustered_pct_10_20_minus_top_5"]
    difference_interval = bucket_difference["block_bootstrap_95_interval"]
    lines.extend([
        "",
        "The date-clustered 10-20% minus Top-5% realized-return difference was "
        f"{bucket_difference['mean_difference']:.2%}; moving-block-bootstrap 95% CI "
        f"[{difference_interval[0]:.2%}, {difference_interval[1]:.2%}]. "
        f"The non-overlapping estimate was "
        f"{bucket_difference['nonoverlapping_mean_difference']:.2%} across "
        f"{bucket_difference['nonoverlapping_blocks']} blocks.",
    ])
    five_day = report["horizons"]["5d"]
    twenty_day = report["horizons"]["20d"]
    random_five = attribution["5d"]["matched_universe_benchmarks"]["random"]
    random_twenty = attribution["20d"]["matched_universe_benchmarks"]["random"]
    rank_ic_twenty = attribution["20d"]["rank_ic"]
    lines.extend([
        "",
        "## Diagnostic interpretation",
        "",
        "- Within this biased current-snapshot diagnostic, the investable observation "
        "slots captured "
        f"{five_day['static_active_pool_coverage']:.1%} of 5d and "
        f"{twenty_day['static_active_pool_coverage']:.1%} of 20d winner observations. "
        "The corresponding random-selection lifts were only "
        f"{random_five['observed_lift']:.2f}x and "
        f"{random_twenty['observed_lift']:.2f}x; discovery value must be judged against "
        "the matched baselines rather than against 100% coverage.",
        "- Inside the active pool, the complete quantitative funnel retained only "
        f"{five_day['conditional_quant_recall_inside_static_active_pool']:.1%} of 5d and "
        f"{twenty_day['conditional_quant_recall_inside_static_active_pool']:.1%} of 20d winners; "
        "research capacity is a mechanical rejection stage, not proof that expanding "
        "research depth creates alpha.",
        "- Quantitative passes enriched the winner rate by about "
        f"{five_day['winner_precision_lift']:.2f}x / "
        f"{twenty_day['winner_precision_lift']:.2f}x, but their mean realized excess was "
        f"{five_day['quant_pass_mean_realized_excess']:.2%} / "
        f"{twenty_day['quant_pass_mean_realized_excess']:.2%}. "
        f"Only {twenty_day['quant_pass_opportunities']} observations passed the complete "
        "20d funnel, so pass precision is statistically uninformative and does not justify "
        "loosening gates.",
        "- The 20d daily Rank IC was "
        f"{rank_ic_twenty['mean']:.3f} with block-bootstrap interval "
        f"[{rank_ic_twenty['block_bootstrap_95_interval'][0]:.3f}, "
        f"{rank_ic_twenty['block_bootstrap_95_interval'][1]:.3f}]. "
        "The corrected investable-universe audit therefore does not establish even weak "
        "cross-sectional ranking skill in this window.",
    ])
    for horizon in ("5d", "20d"):
        item = report["horizons"][horizon]
        lines.extend(["", f"## {horizon} attribution", "", "| Stage | Count |", "|---|---:|"])
        for stage, count in item["stage_counts"].items():
            lines.append(f"| {stage} | {count} |")
        lines.extend(["", "Largest missed winners:", "", "| Date | Symbol | Excess return | Stage | Event proxy |", "|---|---|---:|---|---|"])
        for row in item["largest_missed"][:15]:
            lines.append(
                f"| {row['date']} | {row['symbol']} | {row['realized_excess']:.2%} | "
                f"{row['stage']} | {row['event_proxy']} |"
            )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    actual = report["actual_logged_funnel"]
    lines.extend([
        "",
        "## Actual logged production funnel",
        "",
        f"Decision runs available: {actual['decision_runs']}",
        "",
        "| Disposition | Count |",
        "|---|---:|",
    ])
    for disposition, count in actual["dispositions"].items():
        lines.append(f"| {disposition} | {count} |")
    lines.extend(["", actual["limitation"]])
    return "\n".join(lines) + "\n"
