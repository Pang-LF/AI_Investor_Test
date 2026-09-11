import unittest

from ai_investor.forecasting import (
    _beta_adjusted_return,
    _empirical_positive_probability,
    _rolling_beta,
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


if __name__ == "__main__":
    unittest.main()
