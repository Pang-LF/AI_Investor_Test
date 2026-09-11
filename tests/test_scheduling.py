import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ai_investor.agent import _decision_reason
from ai_investor.config import Settings
from ai_investor.ledger import Ledger


class SchedulingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings.load(Path("config/settings.toml"))

    def test_oldest_due_fixed_slot_has_priority_and_catches_up(self) -> None:
        at_1320 = datetime(2026, 9, 11, 17, 20, tzinfo=timezone.utc)
        self.assertEqual(
            _decision_reason(at_1320, self.settings, True, set()),
            "scheduled_0945",
        )
        self.assertEqual(
            _decision_reason(
                at_1320, self.settings, True, {"scheduled_0945"}
            ),
            "scheduled_1300",
        )

    def test_event_trigger_is_separate_after_due_slots_complete(self) -> None:
        at_1320 = datetime(2026, 9, 11, 17, 20, tzinfo=timezone.utc)
        completed = {"scheduled_0945", "scheduled_1300"}
        self.assertEqual(
            _decision_reason(at_1320, self.settings, True, completed),
            "market_trigger",
        )
        self.assertIsNone(
            _decision_reason(at_1320, self.settings, False, completed)
        )

    def test_ledger_counts_only_event_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with Ledger(Path(directory) / "ledger.sqlite") as ledger:
                for index, reason in enumerate(
                    ("market_trigger", "market_trigger_1115", "scheduled_0945")
                ):
                    ledger.record_run(
                        run_id=str(index),
                        decision_key=f"2026-09-11|{reason}|strategy_v1",
                        trading_date="2026-09-11",
                        mode="SHADOW",
                        status="completed",
                        strategy_version="strategy_v1",
                        risk_policy_version="risk_v1",
                        portfolio_value=1000,
                        cash=1000,
                        payload={},
                    )
                self.assertEqual(
                    ledger.event_decision_runs_today("2026-09-11"), 2
                )
                self.assertEqual(
                    ledger.scheduled_decision_reasons("2026-09-11"),
                    {"scheduled_0945"},
                )
                self.assertIsNotNone(ledger.last_decision_at("2026-09-11"))


if __name__ == "__main__":
    unittest.main()
