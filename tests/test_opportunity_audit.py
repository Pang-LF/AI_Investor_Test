import unittest

from ai_investor.opportunity_audit import (
    _event_proxy,
    _spearman,
    _universe_symbols,
)


class OpportunityAuditTests(unittest.TestCase):
    def test_universe_accepts_object_candidates_and_string_event_bucket(self) -> None:
        broad, active = _universe_symbols({
            "large_candidates": [{"symbol": "AAA"}],
            "event_bucket": ["BBB"],
            "entries": [
                {"symbol": "CCC"},
                {"symbol": "QQQ", "investable": False},
            ],
        })
        self.assertEqual(broad, ["AAA", "CCC", "SPY"])
        self.assertEqual(active, {"CCC", "SPY"})

    def test_event_proxy_requires_price_move_or_move_with_relative_volume(self) -> None:
        quiet_closes = [100.0] * 21
        volumes = [100.0] * 20 + [250.0]
        self.assertFalse(_event_proxy(quiet_closes, volumes, 20))

        moderate_move = [100.0] * 20 + [102.5]
        self.assertTrue(_event_proxy(moderate_move, volumes, 20))

        large_move = [100.0] * 20 + [104.1]
        ordinary_volume = [100.0] * 21
        self.assertTrue(_event_proxy(large_move, ordinary_volume, 20))

    def test_spearman_measures_rank_direction(self) -> None:
        self.assertAlmostEqual(_spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertAlmostEqual(_spearman([1, 2, 3], [2, 4, 8]), 1.0)


if __name__ == "__main__":
    unittest.main()
