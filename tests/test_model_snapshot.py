import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_investor.config import Settings
from ai_investor.forecasting import RidgeModel
from ai_investor.model_snapshot import get_or_create_daily_models


def _model(horizon: int) -> RidgeModel:
    return RidgeModel(
        horizon=horizon,
        intercept=0.0,
        coefficients=(0.0,) * 5,
        feature_means=(0.0,) * 5,
        feature_scales=(1.0,) * 5,
        residual_std=0.1,
        validation_errors=(0.01, -0.01),
        training_samples=100,
        validation_samples=2,
        validation_date_blocks=2,
        validation_bias=0.0,
        validation_bias_interval=(-0.01, 0.01),
        applied_validation_bias=0.0,
        validation_dates=("2026-01-01", "2026-02-01"),
        validation_predictions=(0.0, 0.0),
        validation_targets=(0.01, -0.01),
    )


class ModelSnapshotTests(unittest.TestCase):
    def test_second_run_reuses_same_day_models(self) -> None:
        settings = Settings.load(Path("config/settings.toml")).forecast
        fitted = {"5d": _model(5), "20d": _model(20)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "ai_investor.model_snapshot.forecast_assets",
                return_value=([], fitted),
            ) as fit:
                first, first_meta = get_or_create_daily_models(
                    root, "2026-09-14", {}, settings
                )
                second, second_meta = get_or_create_daily_models(
                    root, "2026-09-14", {}, settings
                )
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(first, second)
            self.assertEqual(first_meta["status"], "created")
            self.assertEqual(second_meta["status"], "reused")


if __name__ == "__main__":
    unittest.main()
