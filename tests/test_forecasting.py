import unittest
from dataclasses import replace
from pathlib import Path

from ai_investor.config import Settings
from ai_investor.forecasting import (
    AssetForecast,
    _beta_adjusted_return,
    _empirical_positive_probability,
    _rolling_beta,
    candidate_forecasts,
    calibration_diagnostics,
    execution_candidate_forecasts,
    fit_ridge_model,
)


class ForecastingTests(unittest.TestCase):
    def test_beta_adjustment_removes_scaled_market_move(self) -> None:
        market = [100.0]
        asset = [100.0]
        for index in range(1, 101):
            market_return = 0.003 + (index % 7 - 3) * 0.001
            asset_return = 2.0 * market_return
            market.append(market[-1] * (1.0 + market_return))
            asset.append(asset[-1] * (1.0 + asset_return))
        beta = _rolling_beta(asset, market, 100)
        residual = _beta_adjusted_return(asset, market, 100, 20)
        self.assertAlmostEqual(beta, 2.0, places=6)
        self.assertLess(abs(residual), 0.002)

    def test_validation_probability_is_empirical(self) -> None:
        self.assertEqual(_empirical_positive_probability(0.0, [-1.0, 1.0]), 0.5)
        self.assertGreater(
            _empirical_positive_probability(2.0, [-1.0, 1.0]), 0.5
        )

    def test_temporal_validation_retains_a_purged_gap(self) -> None:
        samples = []
        for day in range(200):
            features = tuple((day + offset) / 100.0 for offset in range(5))
            samples.append((f"2026-{day:03d}", features, day / 1000.0))
        model = fit_ridge_model(samples, horizon=20, penalty=8.0, minimum_samples=50)
        self.assertGreater(model.validation_samples, 0)
        self.assertEqual(len(model.validation_errors), model.validation_samples)
        self.assertLess(model.validation_date_blocks, 40)

    def test_validation_error_is_realized_minus_predicted_and_corrects_bias(self) -> None:
        samples = []
        for day in range(160):
            target = 0.02 if day >= 128 else 0.0
            samples.append((f"{day:03d}", (0.0,) * 5, target))
        model = fit_ridge_model(
            samples, horizon=5, penalty=8.0, minimum_samples=50, shrinkage=1.0
        )
        self.assertGreater(model.validation_bias, 0.015)
        diagnostics = calibration_diagnostics(model)
        self.assertEqual(
            diagnostics["validation_date_blocks"], model.validation_date_blocks
        )
        self.assertEqual(len(diagnostics["edge_ratio_buckets"]), 6)

    def test_research_and_execution_gates_are_separate(self) -> None:
        settings = Settings.load(Path("config/settings.toml")).forecast
        forecast = AssetForecast(
            symbol="AAA", data_as_of="2026-09-10",
            expected_excess_return_5d=-0.01,
            expected_excess_return_20d=-0.01,
            probability_positive_excess_5d=0.40,
            probability_positive_excess_20d=0.40,
            uncertainty_5d=0.05, uncertainty_20d=0.10, signals={},
            raw_expected_excess_return_5d=0.01,
            raw_expected_excess_return_20d=0.02,
            raw_probability_positive_excess_5d=0.51,
            raw_probability_positive_excess_20d=0.51,
            calibration_date_blocks_20d=10,
        )
        self.assertEqual(candidate_forecasts([forecast], settings), [forecast])
        self.assertEqual(execution_candidate_forecasts([forecast], settings), [])
        approved = replace(settings, execution_calibration_approved=True)
        self.assertEqual(execution_candidate_forecasts([forecast], approved), [])

    def test_event_can_enter_research_without_creating_execution_edge(self) -> None:
        settings = Settings.load(Path("config/settings.toml")).forecast
        forecast = AssetForecast(
            symbol="EVENT", data_as_of="2026-09-10",
            expected_excess_return_5d=-0.01,
            expected_excess_return_20d=-0.01,
            probability_positive_excess_5d=0.40,
            probability_positive_excess_20d=0.40,
            uncertainty_5d=0.05, uncertainty_20d=0.10, signals={},
            raw_expected_excess_return_5d=-0.01,
            raw_expected_excess_return_20d=-0.01,
            raw_probability_positive_excess_5d=0.40,
            raw_probability_positive_excess_20d=0.40,
        )
        self.assertEqual(
            candidate_forecasts([forecast], settings, event_symbols={"EVENT"}),
            [forecast],
        )
        self.assertEqual(execution_candidate_forecasts([forecast], settings), [])


if __name__ == "__main__":
    unittest.main()
