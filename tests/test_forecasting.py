import unittest
from dataclasses import replace
from pathlib import Path

from ai_investor.config import Settings
from ai_investor.forecasting import (
    AssetForecast,
    _beta_adjusted_return,
    _date_block_positive_probability,
    _date_clustered_probability_interval,
    _empirical_positive_probability,
    _rolling_beta,
    _supported_bias_correction,
    candidate_forecasts,
    calibration_diagnostics,
    investment_candidate_forecasts,
    investment_eligibility_failures,
    fit_ridge_model,
    history_integrity_issues,
    model_integrity_issues,
)
from ai_investor.forecasting import RidgeModel
from ai_investor.market_data import DailyBar


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

    def test_probability_counts_dates_not_cross_section_rows(self) -> None:
        errors = [1.0] * 100 + [-1.0]
        dates = ["2026-01-01"] * 100 + ["2026-02-01"]
        probability = _date_block_positive_probability(
            0.0, errors, dates, prior_date_blocks=20
        )
        self.assertEqual(probability, 0.5)

    def test_probability_interval_uses_same_prior_shrinkage_as_point(self) -> None:
        errors = [1.0, 1.0, -1.0, 1.0]
        dates = ["a", "b", "c", "d"]
        point = _date_block_positive_probability(
            0.0, errors, dates, prior_date_blocks=20
        )
        lower, upper = _date_clustered_probability_interval(
            0.0, errors, dates, prior_date_blocks=20
        )
        self.assertLessEqual(lower, point)
        self.assertLessEqual(point, upper)

    def test_history_integrity_quarantines_split_like_discontinuity(self) -> None:
        bars = [
            DailyBar("BAD", "2026-01-01", 1, 1, 1, 1, 100),
            DailyBar("BAD", "2026-01-02", 50, 50, 50, 50, 100),
        ]
        issues = history_integrity_issues({"BAD": bars}, 3.0)
        self.assertIn("BAD", issues)
        self.assertIn("close_ratio=50", issues["BAD"][0])

    def test_model_integrity_blocks_implausible_uncertainty_and_probability_ci(self) -> None:
        settings = Settings.load(Path("config/settings.toml")).forecast
        model = RidgeModel(
            horizon=20, intercept=0.0, coefficients=(0.0,) * 5,
            feature_means=(0.0,) * 5, feature_scales=(1.0,) * 5,
            residual_std=12.7, validation_errors=(0.0, 0.0),
            training_samples=1000, validation_samples=2,
            validation_date_blocks=2, validation_bias=0.0,
            validation_bias_interval=(-0.1, 0.1), applied_validation_bias=0.0,
            validation_dates=("a", "b"), validation_predictions=(0.0, 0.0),
            validation_targets=(0.0, 0.0),
        )
        forecast = AssetForecast(
            symbol="BAD", data_as_of="2026-01-01",
            expected_excess_return_5d=0.01, expected_excess_return_20d=0.02,
            probability_positive_excess_5d=0.55,
            probability_positive_excess_20d=0.70,
            uncertainty_5d=12.7, uncertainty_20d=12.7, signals={},
            probability_positive_excess_20d_interval=(0.80, 0.90),
        )
        five_day = replace(model, horizon=5)
        issues = model_integrity_issues(
            [forecast], {"5d": five_day, "20d": model}, settings
        )
        self.assertTrue(any("uncertainty_20d_out_of_bounds" in item for item in issues))
        self.assertTrue(any("probability_ci_20d" in item for item in issues))

    def test_bias_correction_requires_supported_date_blocks(self) -> None:
        self.assertEqual(
            _supported_bias_correction(0.02, (0.01, 0.03), 7, 20), 0.0
        )
        self.assertEqual(
            _supported_bias_correction(0.02, (0.01, 0.03), 20, 20), 0.01
        )
        self.assertEqual(
            _supported_bias_correction(0.02, (-0.01, 0.03), 20, 20), 0.0
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

    def test_validation_error_is_realized_minus_walk_forward_prediction(self) -> None:
        samples = []
        for day in range(160):
            target = 0.02 if day >= 128 else 0.0
            samples.append((f"{day:03d}", (0.0,) * 5, target))
        model = fit_ridge_model(
            samples, horizon=5, penalty=8.0, minimum_samples=50, shrinkage=1.0
        )
        for error, prediction, target in zip(
            model.validation_errors,
            model.validation_predictions,
            model.validation_targets,
        ):
            self.assertAlmostEqual(error, target - prediction)
        self.assertGreaterEqual(model.validation_date_blocks, 20)
        diagnostics = calibration_diagnostics(model)
        self.assertEqual(
            diagnostics["validation_date_blocks"], model.validation_date_blocks
        )
        self.assertEqual(len(diagnostics["edge_ratio_buckets"]), 6)

    def test_empirical_probability_does_not_double_count_bias(self) -> None:
        # A raw prediction of zero with uniformly positive residuals is already
        # calibrated by those residuals; adding their median to the prediction
        # a second time would overstate the probability.
        errors = [-0.02, -0.005, 0.02]
        once = _empirical_positive_probability(0.0, errors)
        double_counted = _empirical_positive_probability(0.01, errors)
        self.assertLess(once, double_counted)

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
        self.assertEqual(investment_candidate_forecasts([forecast], settings), [])
        approved = replace(settings, execution_calibration_approved=True)
        self.assertEqual(investment_candidate_forecasts([forecast], approved), [])

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
        self.assertEqual(investment_candidate_forecasts([forecast], settings), [])

    def test_broad_shadow_name_cannot_enter_shared_optimizer(self) -> None:
        settings = Settings.load(Path("config/settings.toml")).forecast
        forecast = AssetForecast(
            symbol="BROAD", data_as_of="2026-09-10",
            expected_excess_return_5d=.02,
            expected_excess_return_20d=.04,
            probability_positive_excess_5d=.60,
            probability_positive_excess_20d=.65,
            uncertainty_5d=.05, uncertainty_20d=.10, signals={},
            calibration_date_blocks_20d=10,
        )
        self.assertEqual(
            investment_candidate_forecasts(
                [forecast], settings, allowed_symbols={"CORE"}
            ),
            [],
        )
        self.assertEqual(
            investment_candidate_forecasts(
                [forecast], settings, allowed_symbols={"BROAD"}
            ),
            [forecast],
        )

    def test_execution_gate_reports_the_exact_failed_threshold(self) -> None:
        settings = Settings.load(Path("config/settings.toml")).forecast
        forecast = AssetForecast(
            symbol="PBF", data_as_of="2026-09-11",
            expected_excess_return_5d=.01,
            expected_excess_return_20d=.0246,
            probability_positive_excess_5d=.55,
            probability_positive_excess_20d=.598,
            uncertainty_5d=.10, uncertainty_20d=.2134, signals={},
            calibration_date_blocks_20d=7,
        )
        failures = investment_eligibility_failures(forecast, settings)
        self.assertEqual(len(failures), 1)
        self.assertIn("edge_ratio", failures[0])


if __name__ == "__main__":
    unittest.main()
