import unittest
from pathlib import Path

from ai_investor.config import Settings
from ai_investor.decision_funnel import (
    build_candidate_funnel,
    finalize_candidate_funnel,
)
from ai_investor.forecasting import AssetForecast


class DecisionFunnelTests(unittest.TestCase):
    def test_every_observation_keeps_an_explicit_disposition(self) -> None:
        settings = Settings.load(Path("config/settings.toml")).forecast
        forecast = AssetForecast(
            symbol="AAA",
            data_as_of="2026-09-14",
            expected_excess_return_5d=0.01,
            expected_excess_return_20d=0.02,
            probability_positive_excess_5d=0.55,
            probability_positive_excess_20d=0.56,
            uncertainty_5d=0.1,
            uncertainty_20d=0.1,
            signals={},
            raw_expected_excess_return_20d=0.02,
            raw_probability_positive_excess_20d=0.60,
        )
        rows = build_candidate_funnel(
            [
                {"symbol": "SPY", "bucket": "context", "investable": False},
                {"symbol": "AAA", "bucket": "large", "investable": True},
                {"symbol": "BBB", "bucket": "small", "investable": True},
            ],
            [forecast],
            settings,
            event_symbols=(),
            persistent_symbols=(),
            research_symbols={"AAA"},
            holding_symbols=(),
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["research_selection_reason"], "context_only_not_investable")
        self.assertEqual(rows[2]["research_selection_reason"], "missing_or_short_history")
        finalized = finalize_candidate_funnel(
            rows,
            assessments={"AAA": {"verdict": "allow"}},
            investment_failures={"AAA": ["edge_ratio"]},
            holding_decisions={},
            target_weights={},
            orders=[],
        )
        self.assertEqual(
            finalized[1]["final_disposition"], "investment_eligibility_failed"
        )


if __name__ == "__main__":
    unittest.main()
