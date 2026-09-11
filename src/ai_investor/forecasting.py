from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import fmean, median
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .config import ForecastSettings
from .market_data import DailyBar


FEATURE_NAMES = (
    "residual_return_20d",
    "residual_return_60d",
    "negative_residual_return_5d",
    "residual_return_1d",
    "volume_acceleration_20d",
)


@dataclass(frozen=True)
class RidgeModel:
    horizon: int
    intercept: float
    coefficients: Tuple[float, ...]
    feature_means: Tuple[float, ...]
    feature_scales: Tuple[float, ...]
    residual_std: float
    validation_errors: Tuple[float, ...]
    training_samples: int
    validation_samples: int
    validation_date_blocks: int
    validation_bias: float
    validation_dates: Tuple[str, ...]
    validation_predictions: Tuple[float, ...]
    validation_targets: Tuple[float, ...]

    def predict(self, features: Sequence[float]) -> float:
        standardized = [
            (value - mean) / scale
            for value, mean, scale in zip(
                features, self.feature_means, self.feature_scales
            )
        ]
        return self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, standardized)
        )


@dataclass(frozen=True)
class AssetForecast:
    symbol: str
    data_as_of: str
    expected_excess_return_5d: float
    expected_excess_return_20d: float
    probability_positive_excess_5d: float
    probability_positive_excess_20d: float
    uncertainty_5d: float
    uncertainty_20d: float
    signals: Dict[str, float]
    model_prediction_excess_return_5d: float = 0.0
    model_prediction_excess_return_20d: float = 0.0
    forecast_shrinkage: float = 1.0
    raw_expected_excess_return_5d: float = 0.0
    raw_expected_excess_return_20d: float = 0.0
    raw_probability_positive_excess_5d: float = 0.5
    raw_probability_positive_excess_20d: float = 0.5
    validation_bias_5d: float = 0.0
    validation_bias_20d: float = 0.0
    calibration_observations_5d: int = 0
    calibration_observations_20d: int = 0
    probability_positive_excess_5d_interval: Tuple[float, float] = (0.0, 1.0)
    probability_positive_excess_20d_interval: Tuple[float, float] = (0.0, 1.0)
    calibration_date_blocks_5d: int = 0
    calibration_date_blocks_20d: int = 0
    model_version: str = "pooled_beta_ridge_v0.3.1"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MarketRegime:
    label: str
    spy_return_20d: float
    spy_return_60d: float
    annualized_volatility_20d: float
    positive_breadth_20d: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _return(values: Sequence[float], end: int, lookback: int) -> float:
    if end - lookback < 0 or values[end - lookback] <= 0:
        return 0.0
    return values[end] / values[end - lookback] - 1.0


def _rolling_beta(
    values: Sequence[float], benchmark: Sequence[float], end: int, lookback: int = 60
) -> float:
    start = max(1, end - lookback + 1)
    asset_returns = [
        values[index] / values[index - 1] - 1.0
        for index in range(start, end + 1)
        if values[index - 1] > 0 and benchmark[index - 1] > 0
    ]
    benchmark_returns = [
        benchmark[index] / benchmark[index - 1] - 1.0
        for index in range(start, end + 1)
        if values[index - 1] > 0 and benchmark[index - 1] > 0
    ]
    if len(asset_returns) < 20:
        return 1.0
    market_mean = fmean(benchmark_returns)
    asset_mean = fmean(asset_returns)
    market_variance = fmean(
        (value - market_mean) ** 2 for value in benchmark_returns
    )
    if market_variance <= 1e-12:
        return 1.0
    covariance = fmean(
        (asset - asset_mean) * (market - market_mean)
        for asset, market in zip(asset_returns, benchmark_returns)
    )
    return max(-1.0, min(covariance / market_variance, 3.0))


def _beta_adjusted_return(
    values: Sequence[float], benchmark: Sequence[float], end: int, lookback: int
) -> float:
    beta = _rolling_beta(values, benchmark, end)
    start = end - lookback + 1
    if start < 1:
        return 0.0
    return sum(
        values[index] / values[index - 1]
        - 1.0
        - beta * (benchmark[index] / benchmark[index - 1] - 1.0)
        for index in range(start, end + 1)
    )


def _features(
    closes: Sequence[float],
    volumes: Sequence[float],
    benchmark: Sequence[float],
    index: int,
) -> Tuple[float, ...]:
    residual_20 = _beta_adjusted_return(closes, benchmark, index, 20)
    residual_60 = _beta_adjusted_return(closes, benchmark, index, 60)
    residual_5 = _beta_adjusted_return(closes, benchmark, index, 5)
    residual_1 = _beta_adjusted_return(closes, benchmark, index, 1)
    average_volume = fmean(volumes[index - 20 : index]) if index >= 20 else 0.0
    volume_acceleration = (
        volumes[index] / average_volume - 1.0 if average_volume > 0 else 0.0
    )
    return (
        residual_20,
        residual_60,
        -residual_5,
        residual_1,
        max(-3.0, min(volume_acceleration, 5.0)),
    )


def _aligned_series(
    bars: Sequence[DailyBar], benchmark_bars: Sequence[DailyBar]
) -> Tuple[List[str], List[float], List[float], List[float]]:
    asset = {bar.begins_at[:10]: bar for bar in bars}
    benchmark = {bar.begins_at[:10]: bar for bar in benchmark_bars}
    dates = sorted(set(asset) & set(benchmark))
    return (
        dates,
        [asset[date].close for date in dates],
        [asset[date].volume for date in dates],
        [benchmark[date].close for date in dates],
    )


def _solve(matrix: List[List[float]], vector: List[float]) -> List[float]:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return [0.0] * size
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * base
                for current, base in zip(augmented[row], augmented[column])
            ]
    return [augmented[index][-1] for index in range(size)]


def _fit_core(
    samples: Sequence[Tuple[str, Tuple[float, ...], float]], penalty: float
) -> Tuple[float, Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
    width = len(FEATURE_NAMES)
    feature_means = tuple(
        fmean(sample[1][column] for sample in samples) for column in range(width)
    )
    feature_scales = []
    for column in range(width):
        variance = fmean(
            (sample[1][column] - feature_means[column]) ** 2 for sample in samples
        )
        feature_scales.append(max(math.sqrt(variance), 1e-6))
    target_mean = fmean(sample[2] for sample in samples)
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]
    for _, features, target in samples:
        row = [
            (features[column] - feature_means[column]) / feature_scales[column]
            for column in range(width)
        ]
        centered_target = target - target_mean
        for left in range(width):
            xty[left] += row[left] * centered_target
            for right in range(width):
                xtx[left][right] += row[left] * row[right]
    scaled_penalty = penalty * max(len(samples) / 1000.0, 1.0)
    for index in range(width):
        xtx[index][index] += scaled_penalty
    coefficients = tuple(_solve(xtx, xty))
    return target_mean, coefficients, feature_means, tuple(feature_scales)


def _predict_components(
    core: Tuple[float, Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]],
    features: Sequence[float],
) -> float:
    intercept, coefficients, means, scales = core
    return intercept + sum(
        coefficient * ((value - mean) / scale)
        for coefficient, value, mean, scale in zip(
            coefficients, features, means, scales
        )
    )


def fit_ridge_model(
    samples: Sequence[Tuple[str, Tuple[float, ...], float]],
    horizon: int,
    penalty: float,
    minimum_samples: int,
    shrinkage: float = 1.0,
) -> RidgeModel:
    if len(samples) < minimum_samples:
        raise ValueError(
            f"Insufficient training samples for {horizon}d model: {len(samples)}"
        )
    dates = sorted({sample[0] for sample in samples})
    split_index = max(1, int(len(dates) * 0.80))
    validation_start = dates[min(split_index, len(dates) - 1)]
    train_end_index = max(0, split_index - horizon)
    train_end = dates[train_end_index]
    train = [sample for sample in samples if sample[0] < train_end]
    validation = [sample for sample in samples if sample[0] >= validation_start]
    if len(train) < minimum_samples // 2 or not validation:
        raise ValueError("Insufficient samples after purged temporal split")
    validation_core = _fit_core(train, penalty)
    # Future-horizon labels from consecutive dates overlap. Retain only one
    # cross-section per horizon-length block so calibration does not count the
    # same subsequent market path twenty times for the 20-day model.
    validation_dates = sorted({sample[0] for sample in validation})
    calibration_dates = set(validation_dates[::horizon])
    calibration = [sample for sample in validation if sample[0] in calibration_dates]
    validation_predictions = [
        shrinkage * _predict_components(validation_core, sample[1])
        for sample in calibration
    ]
    errors = [
        sample[2] - prediction
        for sample, prediction in zip(calibration, validation_predictions)
    ]
    residual_std = max(
        math.sqrt(fmean(error * error for error in errors)), 0.002
    )
    final = _fit_core(samples, penalty)
    return RidgeModel(
        horizon=horizon,
        intercept=final[0],
        coefficients=final[1],
        feature_means=final[2],
        feature_scales=final[3],
        residual_std=residual_std,
        validation_errors=tuple(errors),
        training_samples=len(samples),
        validation_samples=len(calibration),
        validation_date_blocks=len(calibration_dates),
        validation_bias=median(errors),
        validation_dates=tuple(sample[0] for sample in calibration),
        validation_predictions=tuple(validation_predictions),
        validation_targets=tuple(sample[2] for sample in calibration),
    )


def build_training_samples(
    histories: Mapping[str, Sequence[DailyBar]], horizon: int
) -> List[Tuple[str, Tuple[float, ...], float]]:
    benchmark = histories.get("SPY") or []
    samples: List[Tuple[str, Tuple[float, ...], float]] = []
    for symbol, bars in histories.items():
        if symbol == "SPY":
            continue
        dates, closes, volumes, benchmark_closes = _aligned_series(bars, benchmark)
        for index in range(60, len(dates) - horizon):
            features = _features(closes, volumes, benchmark_closes, index)
            beta = _rolling_beta(closes, benchmark_closes, index)
            target = sum(
                closes[future] / closes[future - 1]
                - 1.0
                - beta
                * (
                    benchmark_closes[future] / benchmark_closes[future - 1]
                    - 1.0
                )
                for future in range(index + 1, index + horizon + 1)
            )
            if all(math.isfinite(value) for value in features) and math.isfinite(target):
                samples.append((dates[index], features, target))
    return sorted(samples, key=lambda sample: (sample[0], sample[1]))


def _empirical_positive_probability(
    prediction: float, validation_errors: Sequence[float]
) -> float:
    if not validation_errors:
        return 0.5
    successes = sum(prediction + error > 0 for error in validation_errors)
    return (successes + 1.0) / (len(validation_errors) + 2.0)


def _raw_positive_probability(prediction: float, uncertainty: float) -> float:
    if uncertainty <= 0:
        return 1.0 if prediction > 0 else 0.0
    z = prediction / (uncertainty * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _date_clustered_probability_interval(
    prediction: float,
    validation_errors: Sequence[float],
    validation_dates: Sequence[str],
) -> Tuple[float, float]:
    by_date: Dict[str, List[float]] = {}
    for date, error in zip(validation_dates, validation_errors):
        by_date.setdefault(date, []).append(float(prediction + error > 0))
    rates = [fmean(values) for values in by_date.values() if values]
    if len(rates) < 2:
        return (0.0, 1.0)
    center = fmean(rates)
    variance = sum((value - center) ** 2 for value in rates) / (len(rates) - 1)
    margin = 1.96 * math.sqrt(variance / len(rates))
    return (max(0.0, center - margin), min(1.0, center + margin))


def forecast_assets(
    histories: Mapping[str, Sequence[DailyBar]], settings: ForecastSettings
) -> Tuple[List[AssetForecast], Dict[str, RidgeModel]]:
    samples_5 = build_training_samples(histories, 5)
    samples_20 = build_training_samples(histories, 20)
    models = {
        "5d": fit_ridge_model(
            samples_5, 5, settings.ridge_penalty, settings.min_training_samples,
            settings.shrinkage,
        ),
        "20d": fit_ridge_model(
            samples_20, 20, settings.ridge_penalty, settings.min_training_samples,
            settings.shrinkage,
        ),
    }
    benchmark = histories.get("SPY") or []
    forecasts: List[AssetForecast] = []
    for symbol, bars in histories.items():
        if symbol == "SPY" or len(bars) < settings.min_history_bars:
            continue
        dates, closes, volumes, benchmark_closes = _aligned_series(bars, benchmark)
        if len(dates) < settings.min_history_bars:
            continue
        features = _features(closes, volumes, benchmark_closes, len(dates) - 1)
        raw_5 = models["5d"].predict(features)
        raw_20 = models["20d"].predict(features)
        raw_expected_5 = settings.shrinkage * raw_5
        raw_expected_20 = settings.shrinkage * raw_20
        expected_5 = raw_expected_5 + models["5d"].validation_bias
        expected_20 = raw_expected_20 + models["20d"].validation_bias
        cap = settings.max_abs_forecast_20d
        expected_20 = max(-cap, min(expected_20, cap))
        expected_5 = max(-cap / 2.0, min(expected_5, cap / 2.0))
        uncertainty_5 = models["5d"].residual_std
        uncertainty_20 = models["20d"].residual_std
        signals = dict(zip(FEATURE_NAMES, features))
        signals["rolling_beta_60d"] = _rolling_beta(
            closes, benchmark_closes, len(dates) - 1
        )
        forecasts.append(
            AssetForecast(
                symbol=symbol,
                data_as_of=dates[-1],
                expected_excess_return_5d=expected_5,
                expected_excess_return_20d=expected_20,
                probability_positive_excess_5d=_empirical_positive_probability(
                    raw_expected_5, models["5d"].validation_errors
                ),
                probability_positive_excess_20d=_empirical_positive_probability(
                    raw_expected_20, models["20d"].validation_errors
                ),
                uncertainty_5d=uncertainty_5,
                uncertainty_20d=uncertainty_20,
                signals=signals,
                model_prediction_excess_return_5d=raw_5,
                model_prediction_excess_return_20d=raw_20,
                forecast_shrinkage=settings.shrinkage,
                raw_expected_excess_return_5d=raw_expected_5,
                raw_expected_excess_return_20d=raw_expected_20,
                raw_probability_positive_excess_5d=_raw_positive_probability(
                    raw_expected_5, uncertainty_5
                ),
                raw_probability_positive_excess_20d=_raw_positive_probability(
                    raw_expected_20, uncertainty_20
                ),
                validation_bias_5d=models["5d"].validation_bias,
                validation_bias_20d=models["20d"].validation_bias,
                calibration_observations_5d=models["5d"].validation_samples,
                calibration_observations_20d=models["20d"].validation_samples,
                probability_positive_excess_5d_interval=(
                    _date_clustered_probability_interval(
                        raw_expected_5,
                        models["5d"].validation_errors,
                        models["5d"].validation_dates,
                    )
                ),
                probability_positive_excess_20d_interval=(
                    _date_clustered_probability_interval(
                        raw_expected_20,
                        models["20d"].validation_errors,
                        models["20d"].validation_dates,
                    )
                ),
                calibration_date_blocks_5d=models["5d"].validation_date_blocks,
                calibration_date_blocks_20d=models["20d"].validation_date_blocks,
            )
        )
    return forecasts, models


def candidate_forecasts(
    forecasts: Iterable[AssetForecast],
    settings: ForecastSettings,
    event_symbols: Iterable[str] = (),
) -> List[AssetForecast]:
    event_set = {symbol.upper() for symbol in event_symbols}
    eligible = [
        forecast
        for forecast in forecasts
        if (
            forecast.raw_expected_excess_return_20d
            >= settings.research_min_raw_expected_excess_return_20d
            and forecast.raw_probability_positive_excess_20d
            >= settings.research_min_raw_probability_positive
        )
        or forecast.symbol in event_set
    ]
    return sorted(
        eligible,
        key=lambda forecast: (
            forecast.raw_expected_excess_return_20d
            / max(forecast.uncertainty_20d, 1e-9),
            forecast.raw_expected_excess_return_20d,
        ),
        reverse=True,
    )[: settings.research_candidate_count]


def execution_candidate_forecasts(
    forecasts: Iterable[AssetForecast], settings: ForecastSettings
) -> List[AssetForecast]:
    """Apply only calibrated execution requirements.

    The thresholds remain disabled until a reviewed OOS calibration report is
    explicitly approved in configuration. Research can continue meanwhile.
    """
    if not settings.execution_calibration_approved:
        return []
    return [
        forecast for forecast in forecasts
        if not execution_gate_failures(forecast, settings)
    ]


def execution_gate_failures(
    forecast: AssetForecast, settings: ForecastSettings
) -> Tuple[str, ...]:
    failures: List[str] = []
    if not settings.execution_calibration_approved:
        failures.append("forecast_calibration_disabled")
    if forecast.calibration_date_blocks_20d < settings.execution_min_calibration_date_blocks:
        failures.append(
            f"calibration_blocks={forecast.calibration_date_blocks_20d}"
            f"<{settings.execution_min_calibration_date_blocks}"
        )
    if forecast.expected_excess_return_20d <= settings.execution_min_bias_adjusted_excess_return_20d:
        failures.append(
            f"adjusted_alpha={forecast.expected_excess_return_20d:.4f}"
            f"<={settings.execution_min_bias_adjusted_excess_return_20d:.4f}"
        )
    if forecast.probability_positive_excess_20d < settings.execution_min_calibrated_probability_positive:
        failures.append(
            f"calibrated_probability={forecast.probability_positive_excess_20d:.4f}"
            f"<{settings.execution_min_calibrated_probability_positive:.4f}"
        )
    edge_ratio = forecast.expected_excess_return_20d / max(
        forecast.uncertainty_20d, 1e-9
    )
    if edge_ratio < settings.execution_min_edge_ratio_20d:
        failures.append(
            f"edge_ratio={edge_ratio:.4f}<{settings.execution_min_edge_ratio_20d:.4f}"
        )
    return tuple(failures)


def holding_gate_failures(
    forecast: AssetForecast, settings: ForecastSettings
) -> Tuple[str, ...]:
    failures: List[str] = []
    if forecast.calibration_date_blocks_20d < settings.execution_min_calibration_date_blocks:
        failures.append("insufficient_calibration_blocks")
    if forecast.expected_excess_return_20d <= settings.holding_min_bias_adjusted_excess_return_20d:
        failures.append("holding_adjusted_alpha")
    if forecast.probability_positive_excess_20d < settings.holding_min_calibrated_probability_positive:
        failures.append("holding_calibrated_probability")
    edge_ratio = forecast.expected_excess_return_20d / max(
        forecast.uncertainty_20d, 1e-9
    )
    if edge_ratio < settings.holding_min_edge_ratio_20d:
        failures.append("holding_edge_ratio")
    return tuple(failures)


def calibration_diagnostics(model: RidgeModel) -> Dict[str, object]:
    """Summarize non-overlapping OOS performance by adjusted edge ratio."""
    buckets = (
        ("<0", float("-inf"), 0.0),
        ("0-0.05", 0.0, 0.05),
        ("0.05-0.10", 0.05, 0.10),
        ("0.10-0.15", 0.10, 0.15),
        ("0.15-0.25", 0.15, 0.25),
        (">=0.25", 0.25, float("inf")),
    )
    observations = []
    for date, prediction, target in zip(
        model.validation_dates,
        model.validation_predictions,
        model.validation_targets,
    ):
        adjusted = prediction + model.validation_bias
        observations.append(
            {
                "date": date,
                "adjusted_prediction": adjusted,
                "target": target,
                "edge_ratio": adjusted / max(model.residual_std, 1e-9),
            }
        )
    rows = []
    for label, lower, upper in buckets:
        chosen = [
            item
            for item in observations
            if lower <= item["edge_ratio"] < upper
        ]
        rows.append(
            {
                "edge_ratio_bucket": label,
                "observations": len(chosen),
                "date_blocks": len({item["date"] for item in chosen}),
                "mean_bias_adjusted_prediction": (
                    fmean(item["adjusted_prediction"] for item in chosen)
                    if chosen
                    else None
                ),
                "mean_realized_excess_return": (
                    fmean(item["target"] for item in chosen) if chosen else None
                ),
                "realized_win_rate": (
                    fmean(float(item["target"] > 0) for item in chosen)
                    if chosen
                    else None
                ),
            }
        )
    return {
        "horizon_days": model.horizon,
        "residual_definition": "realized_minus_shrunk_prediction",
        "overlap_control": "one_cross_section_per_horizon_date_block",
        "validation_samples": model.validation_samples,
        "validation_date_blocks": model.validation_date_blocks,
        "calibration_confidence": (
            "LOW" if model.validation_date_blocks < 10
            else "MEDIUM" if model.validation_date_blocks < 20
            else "HIGH"
        ),
        "median_validation_bias": model.validation_bias,
        "residual_std": model.residual_std,
        "edge_ratio_buckets": rows,
    }


def infer_market_regime(
    histories: Mapping[str, Sequence[DailyBar]]
) -> MarketRegime:
    spy = histories.get("SPY") or []
    if len(spy) < 61:
        raise ValueError("SPY history is insufficient for regime inference")
    closes = [bar.close for bar in spy]
    return_20 = closes[-1] / closes[-21] - 1.0
    return_60 = closes[-1] / closes[-61] - 1.0
    daily = [closes[index] / closes[index - 1] - 1 for index in range(1, len(closes))]
    recent = daily[-20:]
    mean = fmean(recent)
    volatility = math.sqrt(
        fmean((value - mean) ** 2 for value in recent)
    ) * math.sqrt(252)
    positive = 0
    total = 0
    for symbol, bars in histories.items():
        if symbol == "SPY":
            continue
        _, asset_closes, _, benchmark_closes = _aligned_series(bars, spy)
        if len(asset_closes) < 21:
            continue
        residual = _beta_adjusted_return(
            asset_closes, benchmark_closes, len(asset_closes) - 1, 20
        )
        positive += int(residual > 0)
        total += 1
    breadth = positive / total if total else 0.0
    if return_20 > 0 and breadth >= 0.55 and volatility < 0.30:
        label = "broad_risk_on"
    elif return_20 > 0 and breadth < 0.45:
        label = "narrow_risk_on"
    elif return_20 < 0 and volatility >= 0.30:
        label = "high_volatility_risk_off"
    elif return_20 < 0:
        label = "risk_off"
    else:
        label = "mixed"
    return MarketRegime(
        label=label,
        spy_return_20d=return_20,
        spy_return_60d=return_60,
        annualized_volatility_20d=volatility,
        positive_breadth_20d=breadth,
    )
