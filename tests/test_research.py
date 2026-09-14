import unittest
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ai_investor.ledger import Ledger
from ai_investor.research import (
    collect_research,
    estimate_model_cost,
    persist_security_facts,
    plan_deep_research,
    plan_sec_research,
)


class FakeResearchClient:
    def __init__(self):
        self.calls = []

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {"data": {}}


class ResearchTests(unittest.TestCase):
    def test_ten_names_and_five_deep_names_use_twelve_calls(self) -> None:
        client = FakeResearchClient()
        symbols = [f"S{index}" for index in range(10)]
        payload, tools = collect_research(client, symbols, deep_candidate_count=5)
        self.assertEqual(len(client.calls), 12)
        self.assertEqual(len(tools), 12)
        self.assertEqual(set(payload["earnings"]), set(symbols[:5]))
        self.assertEqual(set(payload["news"]), set(symbols[:5]))

    def test_account_tradability_adds_one_bounded_call(self) -> None:
        client = FakeResearchClient()
        symbols = [f"S{index}" for index in range(5)]
        payload, tools = collect_research(
            client, symbols, deep_candidate_count=5, account_number="masked-test"
        )
        self.assertEqual(len(client.calls), 13)
        self.assertEqual(tools[0], "get_equity_tradability")
        self.assertIn("tradability", payload)

    def test_deep_research_prioritizes_state_trigger_and_event(self) -> None:
        symbols = ["HOLD", "EVENT", "STATE", "PLAIN"]
        entries = [
            {"symbol": "EVENT", "bucket": "event"},
            {"symbol": "STATE", "bucket": "large"},
        ]
        deep = plan_deep_research(
            symbols,
            entries=entries,
            triggers=[{"symbol": "EVENT"}],
            holding_symbols={"HOLD"},
            persistent_state={"STATE": [{"event_type": "financing"}]},
            limit=3,
        )
        self.assertEqual(deep, ["EVENT", "STATE", "HOLD"])
        self.assertEqual(
            plan_sec_research(
                deep,
                entries=entries,
                triggers=[{"symbol": "EVENT"}],
                persistent_state={"STATE": [{"event_type": "financing"}]},
                limit=2,
            ),
            ["EVENT", "STATE"],
        )

    def test_durable_fact_survives_between_runs_and_low_confidence_does_not(self) -> None:
        assessment = {
            "candidates": [
                {
                    "symbol": "AAA",
                    "durable_facts": [
                        {
                            "event_type": "merger_acquisition",
                            "status": "active",
                            "summary": "Pending acquisition",
                            "confidence": "high",
                            "source_basis": ["sec_filing"],
                            "valid_until": "2027-12-31",
                            "invalidation_condition": "Deal closes or terminates",
                        },
                        {
                            "event_type": "other_material_event",
                            "status": "active",
                            "summary": "Unconfirmed rumor",
                            "confidence": "low",
                            "source_basis": ["robinhood_news"],
                            "valid_until": "2026-10-01",
                            "invalidation_condition": "No confirmation",
                        },
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            with Ledger(Path(directory) / "ledger.sqlite") as ledger:
                count = persist_security_facts(
                    ledger,
                    assessment,
                    datetime(2026, 9, 14, tzinfo=timezone.utc),
                )
                state = ledger.active_security_events({"AAA"}, "2026-09-15")
        self.assertEqual(count, 1)
        self.assertEqual(len(state["AAA"]), 1)
        self.assertEqual(state["AAA"][0]["source_basis"], ["sec_filing"])
        self.assertEqual(state["AAA"][0]["valid_until"], "2027-03-13")

    def test_prior_state_alone_cannot_extend_its_own_expiry(self) -> None:
        assessment = {
            "candidates": [{
                "symbol": "AAA",
                "durable_facts": [{
                    "event_type": "financing",
                    "status": "active",
                    "summary": "Old financing state",
                    "confidence": "high",
                    "source_basis": ["prior_persistent_state"],
                    "valid_until": "2026-12-01",
                    "invalidation_condition": "New filing",
                }],
            }]
        }
        with tempfile.TemporaryDirectory() as directory:
            with Ledger(Path(directory) / "ledger.sqlite") as ledger:
                count = persist_security_facts(
                    ledger,
                    assessment,
                    datetime(2026, 9, 14, tzinfo=timezone.utc),
                )
        self.assertEqual(count, 0)

    def test_official_model_price_ratio(self) -> None:
        terra = estimate_model_cost("gpt-5.6-terra", 40_000, 2_400)
        sol = estimate_model_cost("gpt-5.6-sol", 40_000, 2_400)
        self.assertAlmostEqual(terra, 0.1088)
        self.assertAlmostEqual(sol, 0.2080)
        self.assertGreater(sol, terra)


if __name__ == "__main__":
    unittest.main()
