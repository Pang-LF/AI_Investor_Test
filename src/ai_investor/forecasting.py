from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import fmean
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
    training_samples: int
    validation_samples: int

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
    model_version: str = "pooled_ridge_v0.1.0"

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


def _features(
    closes: Sequence[float],
    volumes: Sequence[float],
    benchmark: Sequence[float],
    index: int,
) -> Tuple[float, ...]:
    residual_20 = _return(closes, index, 20) - _return(benchmark, index, 20)
    residual_60 = _return(closes, index, 60) - _return(benchmark, index, 60)
    residual_5 = _return(closes, index, 5) - _return(benchmark, index, 5)
    residual_1 = _return(closes, index, 1) - _return(benchmark, index, 1)
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
) -> RidgeModel:
    if len(samples) < minimum_samples:
        raise ValueError(
            f"Insufficient training samples for {horizon}d model: {len(samples)}"
        )
    dates = sorted({sample[0] for sample in samples})
    split_date = dates[max(1, int(len(dates) * 0.80)) - 1]
    train = [sample for sample in samples if sample[0] <= split_date]
    validation = [sample for sample in samples if sample[0] > split_date]
    if len(train) < minimum_samples // 2 or not validation:
        train = list(samples)
        validation = list(samples[-max(1, len(samples) // 5) :])
    validation_core = _fit_core(train, penalty)
    errors = [
        sample[2] - _predict_components(validation_core, sample[1])
        for sample in validation
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
        training_samples=len(samples),
        validation_samples=len(validation),
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
            target = (
                closes[index + horizon] / closes[index] - 1.0
                - (benchmark_closes[index + horizon] / benchmark_closes[index] - 1.0)
            )
            if all(math.isfinite(value) for value in features) and math.isfinite(target):
                samples.append((dates[index], features, target))
    return sorted(samples, key=lambda sample: (sample[0], sample[1]))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def forecast_assets(
    histories: Mapping[str, Sequence[DailyBar]], settings: ForecastSettings
) -> Tuple[List[AssetForecast], Dict[str, RidgeModel]]:
    samples_5 = build_training_samples(histories, 5)
    samples_20 = build_training_samples(histories, 20)
    models = {
        "5d": fit_ridge_model(
            samples_5, 5, settings.ridge_penalty, settings.min_training_samples
        ),
        "20d": fit_ridge_model(
            samples_20, 20, settings.ridge_penalty, settings.min_training_samples
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
        expected_5 = settings.shrinkage * raw_5
        expected_20 = settings.shrinkage * raw_20
        cap = settings.max_abs_forecast_20d
        expected_20 = max(-cap, min(expected_20, cap))
        expected_5 = max(-cap / 2.0, min(expected_5, cap / 2.0))
        uncertainty_5 = models["5d"].residual_std
        uncertainty_20 = models["20d"].residual_std
        forecasts.append(
            AssetForecast(
                symbol=symbol,
                data_as_of=dates[-1],
                expected_excess_return_5d=expected_5,
                expected_excess_return_20d=expected_20,
                probability_positive_excess_5d=_normal_cdf(
                    expected_5 / uncertainty_5
                ),
                probability_positive_excess_20d=_normal_cdf(
                    expected_20 / uncertainty_20
                ),
                uncertainty_5d=uncertainty_5,
                uncertainty_20d=uncertainty_20,
                signals=dict(zip(FEATURE_NAMES, features)),
            )
        )
    return forecasts, models


def candidate_forecasts(
    forecasts: Iterable[AssetForecast], settings: ForecastSettings
) -> List[AssetForecast]:
    eligible = [
        forecast
        for forecast in forecasts
        if forecast.expected_excess_return_20d
        >= settings.min_expected_excess_return_20d
        and forecast.probability_positive_excess_20d
        >= settings.min_probability_positive
    ]
    return sorted(
        eligible,
        key=lambda forecast: (
            forecast.expected_excess_return_20d
            / max(forecast.uncertainty_20d, 1e-9),
            forecast.expected_excess_return_20d,
        ),
        reverse=True,
    )[: settings.candidate_count]


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
        residual = (
            asset_closes[-1] / asset_closes[-21] - 1
            - (benchmark_closes[-1] / benchmark_closes[-21] - 1)
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
