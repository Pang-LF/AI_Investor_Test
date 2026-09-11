import unittest

from ai_investor.research import collect_research, estimate_model_cost


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

    def test_official_model_price_ratio(self) -> None:
        terra = estimate_model_cost("gpt-5.6-terra", 40_000, 2_400)
        sol = estimate_model_cost("gpt-5.6-sol", 40_000, 2_400)
        self.assertAlmostEqual(terra, 0.1088)
        self.assertAlmostEqual(sol, 0.2080)
        self.assertGreater(sol, terra)


if __name__ == "__main__":
    unittest.main()
