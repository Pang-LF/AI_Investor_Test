import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ai_investor.config import EventEngineSettings
from ai_investor.event_engine import (
    build_event_shadow_forecasts,
    classify_price_event,
)
from ai_investor.market_data import DailyBar
from ai_investor.ledger import Ledger


class EventEngineTests(unittest.TestCase):
    def test_live_entry_permission_is_rejected_by_configuration(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "must remain shadow-only"):
            EventEngineSettings.from_dict({
                "enabled": True,
                "live_entry_enabled": True,
                "maximum_candidates": 5,
                "minimum_absolute_move": .02,
                "strong_move_threshold": .08,
                "minimum_analog_samples": 30,
                "shrinkage_prior_samples": 50,
            })

    def test_event_type_separates_direction_and_strength(self) -> None:
        self.assertEqual(classify_price_event(.04, .08), "standard_up")
        self.assertEqual(classify_price_event(-.09, .08), "strong_down")

    def test_event_forecast_is_always_shadow_and_uses_past_analogs(self) -> None:
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        spy = []
        asset = []
        price = 100.0
        for index in range(220):
            date = (base + timedelta(days=index)).isoformat()
            spy.append(DailyBar("SPY", date, 100, 101, 99, 100, 1_000_000))
            if index >= 60 and index % 6 == 0:
                price *= 1.04
            else:
                price *= 1.001
            asset.append(DailyBar("AAA", date, price, price, price, price, 1_000_000))
        settings = EventEngineSettings(
            enabled=True,
            live_entry_enabled=False,
            maximum_candidates=5,
            minimum_absolute_move=.02,
            strong_move_threshold=.08,
            minimum_analog_samples=20,
            shrinkage_prior_samples=20,
        )
        result = build_event_shadow_forecasts(
            histories={"SPY": spy, "AAA": asset},
            quotes={"AAA": {
                "last_trade_price": 104,
                "adjusted_previous_close": 100,
            }},
            entries=[{"symbol": "AAA", "bucket": "event", "investable": True}],
            triggers=[{"symbol": "AAA", "type": "price_move"}],
            settings=settings,
            observed_at="2026-01-01T15:00:00Z",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].event_type, "standard_up")
        self.assertGreaterEqual(result[0].analog_samples, 20)
        self.assertFalse(result[0].live_entry_enabled)
        self.assertEqual(result[0].confidence, "SHADOW_UNVALIDATED")
        self.assertLessEqual(abs(result[0].five_day.expected_excess_return), .50)
        self.assertIn("intraday_price_trigger", result[0].discovery_sources)

    def test_shadow_signal_realizations_are_durable_and_idempotent(self) -> None:
        forecast = {
            "symbol": "AAA",
            "observed_at": "2026-01-02T15:00:00Z",
            "event_type": "standard_up",
            "observed_price": 100.0,
            "benchmark_price": 100.0,
            "beta_to_spy": 1.0,
            "engine_version": "test_event_v1",
        }
        bars = {
            "AAA": [
                DailyBar("AAA", f"2026-01-{day:02d}T00:00:00Z", 100, 100, 100,
                         100 + day, 1_000)
                for day in range(3, 9)
            ],
            "SPY": [
                DailyBar("SPY", f"2026-01-{day:02d}T00:00:00Z", 100, 100, 100,
                         100 + day / 10, 1_000)
                for day in range(3, 9)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            with Ledger(Path(directory) / "ledger.sqlite") as ledger:
                self.assertEqual(
                    ledger.record_event_shadow_signals(
                        "run", "2026-01-02", [forecast]
                    ),
                    1,
                )
                self.assertEqual(
                    ledger.record_event_shadow_signals(
                        "run-duplicate", "2026-01-02", [forecast]
                    ),
                    0,
                )
                self.assertEqual(ledger.resolve_event_shadow_signals(bars), 1)
                row = ledger.event_shadow_signals()[0]
                self.assertIsNotNone(row["realized_excess_1d"])
                self.assertIsNotNone(row["realized_excess_5d"])


if __name__ == "__main__":
    unittest.main()
