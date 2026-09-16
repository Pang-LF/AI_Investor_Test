import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ai_investor.config import FactorChallengerSettings
from ai_investor.factor_challenger import (
    ENGINE_VERSION,
    FAMILY_NAMES,
    build_factor_rank_snapshot,
    evaluate_factor_rank_rows,
)
from ai_investor.forecasting import AssetForecast
from ai_investor.ledger import Ledger
from ai_investor.market_data import DailyBar


def bars(symbol: str, slope: float, volume_scale: float = 1.0, count: int = 110):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = []
    price = 100.0
    for index in range(count):
        price *= 1.0 + slope + ((index % 7) - 3) * 0.0001
        result.append(DailyBar(
            symbol=symbol,
            begins_at=(start + timedelta(days=index)).isoformat(),
            open=price * .998,
            high=price * 1.01,
            low=price * .99,
            close=price,
            volume=(1_000_000 + index * 1000) * volume_scale,
        ))
    return result


class FactorChallengerTests(unittest.TestCase):
    def test_live_permission_is_rejected_by_configuration(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "shadow-only"):
            FactorChallengerSettings.from_dict({
                "enabled": True,
                "live_entry_enabled": True,
                "shortlist_size": 25,
                "winsor_lower": .01,
                "winsor_upper": .99,
                "top_bucket_fraction": .10,
                "minimum_cross_section_size": 100,
            })

    def test_snapshot_is_rank_based_equal_family_weight_and_shadow_only(self) -> None:
        histories = {"SPY": bars("SPY", .0005)}
        forecasts = []
        for index in range(60):
            symbol = f"S{index:03d}"
            histories[symbol] = bars(symbol, -.001 + index * .00004, 1 + index / 100)
            forecasts.append(AssetForecast(
                symbol, "2026-04-20", 0, index / 10000, .5, .5, .1, .2, {}
            ))
        snapshot = build_factor_rank_snapshot(histories, forecasts)
        self.assertEqual(len(snapshot.signals), 60)
        self.assertEqual(len(snapshot.top_25), 25)
        self.assertFalse(snapshot.live_eligible)
        self.assertEqual(snapshot.engine_version, ENGINE_VERSION)
        for signal in snapshot.signals:
            self.assertEqual(set(signal.family_ranks), set(FAMILY_NAMES))
            self.assertAlmostEqual(
                signal.composite_score,
                sum(signal.family_ranks.values()) / len(FAMILY_NAMES),
            )
            self.assertTrue(all(
                0 <= value <= 1
                for value in signal.factor_percentile_ranks.values()
            ))
        self.assertEqual(snapshot.ridge_top_25[0], "S059")

    def test_evaluation_compares_ridge_and_composite_rank_direction(self) -> None:
        rows = []
        for rank in range(1, 21):
            rows.append({
                "formation_date": "2026-01-01",
                "symbol": f"S{rank:02d}",
                "composite_rank": rank,
                "ridge_rank": 21 - rank,
                "realized_excess_5d": (21 - rank) / 100,
                "realized_excess_10d": None,
                "realized_excess_20d": None,
                "factors_json": json.dumps({
                    "family_ranks": {
                        family: (21 - rank) / 20 for family in FAMILY_NAMES
                    }
                }),
            })
        result = evaluate_factor_rank_rows(rows)["horizons"]["5"]
        self.assertAlmostEqual(result["composite_mean_rank_ic"], 1.0)
        self.assertAlmostEqual(result["ridge_mean_rank_ic"], -1.0)
        self.assertGreater(
            result["mean_excess_return_by_rank_bucket"]["composite"]["1_5"],
            result["mean_excess_return_by_rank_bucket"]["composite"]["16_25"],
        )

    def test_ledger_signal_write_is_idempotent_and_resolves_horizons(self) -> None:
        histories = {"SPY": bars("SPY", .0005, count=110)}
        histories["AAA"] = bars("AAA", .0010, count=110)
        snapshot = build_factor_rank_snapshot(
            histories,
            [AssetForecast("AAA", "2026-04-20", 0, .01, .5, .5, .1, .2, {})],
            shortlist_size=1,
        )
        payload = snapshot.to_dict()
        # Move formation back twenty sessions so the current histories can resolve it.
        payload["signals"][0]["data_as_of"] = histories["AAA"][-21].begins_at[:10]
        payload["signals"][0]["formation_price"] = histories["AAA"][-21].close
        payload["signals"][0]["benchmark_price"] = histories["SPY"][-21].close
        with tempfile.TemporaryDirectory() as directory:
            with Ledger(Path(directory) / "ledger.sqlite") as ledger:
                self.assertEqual(ledger.record_factor_rank_signals("r1", payload), 1)
                self.assertEqual(ledger.record_factor_rank_signals("r2", payload), 0)
                self.assertEqual(
                    ledger.resolve_factor_rank_signals(
                        histories, engine_version=ENGINE_VERSION
                    ),
                    1,
                )
                row = ledger.factor_rank_signals(ENGINE_VERSION)[0]
                self.assertIsNotNone(row["realized_excess_5d"])
                self.assertIsNotNone(row["realized_excess_10d"])
                self.assertIsNotNone(row["realized_excess_20d"])


if __name__ == "__main__":
    unittest.main()
