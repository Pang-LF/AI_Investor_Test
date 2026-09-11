#!/usr/bin/env python3
"""Reproducible non-overlapping 20-day walk-forward threshold diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, median

from ai_investor.config import Settings
from ai_investor.forecasting import (
    _aligned_series,
    _features,
    _fit_core,
    _predict_components,
    _rolling_beta,
    build_training_samples,
)
from ai_investor.market_data import DailyBar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Historical cache date, YYYY-MM-DD")
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    settings = Settings.load(root / "config" / "settings.toml")
    raw = json.loads(
        (root / ".local" / "state" / "historicals" / f"{args.date}.json").read_text()
    )
    histories = {
        symbol: [DailyBar(**bar) for bar in bars]
        for symbol, bars in raw["histories"].items()
    }
    benchmark = histories["SPY"]
    benchmark_dates = [bar.begins_at[:10] for bar in benchmark]
    samples = build_training_samples(histories, 20)
    oos: list[dict] = []
    blocks: list[dict] = []
    for index in range(300, len(benchmark_dates) - 20, 20):
        date = benchmark_dates[index]
        last_safe_training_signal_date = benchmark_dates[index - 20]
        training = [
            sample for sample in samples
            if sample[0] <= last_safe_training_signal_date
        ]
        if len(training) < settings.forecast.min_training_samples:
            continue
        model = _fit_core(training, settings.forecast.ridge_penalty)
        prior_errors = [item["realized"] - item["raw"] for item in oos]
        prior_blocks = len({item["date"] for item in oos})
        bias = median(prior_errors) if prior_errors else 0.0
        uncertainty = (
            max(math.sqrt(fmean(error * error for error in prior_errors)), 0.002)
            if prior_errors else 0.10
        )
        forecasts = []
        for symbol, bars in histories.items():
            if symbol == "SPY":
                continue
            dates, closes, volumes, benchmark_closes = _aligned_series(bars, benchmark)
            if date not in dates:
                continue
            position = dates.index(date)
            if position < settings.forecast.min_history_bars or position + 20 >= len(dates):
                continue
            features = _features(closes, volumes, benchmark_closes, position)
            prediction = settings.forecast.shrinkage * _predict_components(model, features)
            beta = _rolling_beta(closes, benchmark_closes, position)
            realized = sum(
                closes[future] / closes[future - 1] - 1.0
                - beta * (
                    benchmark_closes[future] / benchmark_closes[future - 1] - 1.0
                )
                for future in range(position + 1, position + 21)
            )
            raw_probability = 0.5 * (
                1.0 + math.erf(prediction / (uncertainty * math.sqrt(2.0)))
            )
            calibrated_probability = (
                (sum(prediction + error > 0 for error in prior_errors) + 1.0)
                / (len(prior_errors) + 2.0)
                if prior_errors else 0.5
            )
            adjusted = prediction + bias
            forecasts.append(
                {
                    "date": date, "symbol": symbol, "raw": prediction,
                    "realized": realized, "raw_probability": raw_probability,
                    "calibrated_probability": calibrated_probability,
                    "adjusted": adjusted, "edge_ratio": adjusted / uncertainty,
                }
            )
        research = sorted(
            [
                item for item in forecasts
                if item["raw"] >= settings.forecast.research_min_raw_expected_excess_return_20d
                and item["raw_probability"] >= settings.forecast.research_min_raw_probability_positive
            ],
            key=lambda item: (item["raw"] / uncertainty, item["raw"]),
            reverse=True,
        )[: settings.forecast.research_candidate_count]
        blocks.append({"date": date, "prior_blocks": prior_blocks, "research": research})
        oos.extend(forecasts)

    grid = []
    for probability in (0.50, 0.525, 0.55, 0.575):
        for edge_ratio in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
            block_returns = []
            names = 0
            for block in blocks:
                if block["prior_blocks"] < settings.forecast.execution_min_calibration_date_blocks:
                    continue
                selected = [
                    item for item in block["research"]
                    if item["adjusted"]
                    > settings.forecast.execution_min_bias_adjusted_excess_return_20d
                    and item["calibrated_probability"] >= probability
                    and item["edge_ratio"] >= edge_ratio
                ]
                if selected:
                    block_returns.append(
                        fmean(
                            item["realized"] - settings.portfolio.reallocation_cost_fraction
                            for item in selected
                        )
                    )
                    names += len(selected)
            if block_returns:
                grid.append(
                    {
                        "calibrated_probability": probability,
                        "edge_ratio": edge_ratio,
                        "date_blocks": len(block_returns),
                        "selected_names": names,
                        "mean_net_excess_return": fmean(block_returns),
                        "median_net_excess_return": median(block_returns),
                        "block_win_rate": fmean(value > 0 for value in block_returns),
                        "worst_block_return": min(block_returns),
                    }
                )
    print(json.dumps({
        "source_cache_date": args.date,
        "warning": "Static current universe creates survivorship and selection bias.",
        "evaluation_blocks": len(blocks),
        "usable_blocks": sum(
            block["prior_blocks"] >= settings.forecast.execution_min_calibration_date_blocks
            for block in blocks
        ),
        "round_trip_cost_fraction": settings.portfolio.reallocation_cost_fraction,
        "configured_policy": {
            "calibrated_probability": settings.forecast.execution_min_calibrated_probability_positive,
            "edge_ratio": settings.forecast.execution_min_edge_ratio_20d,
            "bias_adjusted_alpha": settings.forecast.execution_min_bias_adjusted_excess_return_20d,
        },
        "grid": grid,
    }, indent=2))


if __name__ == "__main__":
    main()
