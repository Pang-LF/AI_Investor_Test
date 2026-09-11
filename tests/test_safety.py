import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ai_investor.config import Settings
from ai_investor.monitor import (
    UniverseCache,
    is_regular_market_window,
    market_summary,
    quote_batches,
    quotes_are_fresh,
)
from ai_investor.robinhood_mcp import (
    KNOWN_MUTATING_TOOLS,
    MONITOR_READ_ONLY_TOOLS,
)
from ai_investor.robinhood_readonly import READ_ONLY_TOOLS, WRITE_TOOLS
from ai_investor.universe import UniverseEntry, _rank_core, assemble_universe


class SafetyTests(unittest.TestCase):
    def test_core_ranking_does_not_use_recent_return(self) -> None:
        def row(change: float) -> dict:
            return {
                "ticker": "TEST",
                "columns": {
                    "Last": 100,
                    "Average volume": 2_000_000,
                    "Market cap": 10_000_000_000,
                    "Sector": "Technology",
                    "% Change": change,
                },
            }

        rising = _rank_core([row(0.30)], min_dollar_volume=1)
        falling = _rank_core([row(-0.30)], min_dollar_volume=1)
        self.assertEqual(rising, falling)

    def test_read_only_allowlist_has_no_write_tools(self) -> None:
        self.assertFalse(set(READ_ONLY_TOOLS) & WRITE_TOOLS)
        self.assertTrue(all(name.startswith("get_") for name in READ_ONLY_TOOLS))
        self.assertFalse(MONITOR_READ_ONLY_TOOLS & KNOWN_MUTATING_TOOLS)
        self.assertEqual(
            MONITOR_READ_ONLY_TOOLS,
            {
                "get_accounts",
                "get_equity_positions",
                "get_equity_quotes",
                "preview_scan",
            },
        )

    def test_sixty_symbols_make_three_quote_batches(self) -> None:
        symbols = [f"S{number}" for number in range(60)]
        batches = quote_batches(symbols, 20)
        self.assertEqual([len(batch) for batch in batches], [20, 20, 20])
        self.assertEqual([item for batch in batches for item in batch], symbols)

    def test_regular_market_window(self) -> None:
        self.assertTrue(
            is_regular_market_window(
                datetime(2026, 9, 10, 14, 0, tzinfo=timezone.utc),
                "America/New_York",
            )
        )

    def test_quote_freshness_guard(self) -> None:
        now = datetime(2026, 9, 10, 15, 0, tzinfo=timezone.utc)
        self.assertTrue(
            quotes_are_fresh(
                [{"last_trade_time": "2026-09-10T14:45:00Z"}], now, 30
            )
        )
        self.assertFalse(
            quotes_are_fresh(
                [{"last_trade_time": "2026-09-09T20:00:00Z"}], now, 30
            )
        )
        self.assertFalse(
            quotes_are_fresh(
                [
                    {"last_trade_time": "2026-09-10T14:59:00Z"},
                    {"last_trade_time": "2026-09-09T20:00:00Z"},
                ],
                now,
                30,
            )
        )
        self.assertFalse(
            quotes_are_fresh(
                [
                    {"last_trade_time": "2026-09-10T14:59:00Z"},
                    {"last_trade_time": None},
                ],
                now,
                30,
            )
        )
        self.assertFalse(
            is_regular_market_window(
                datetime(2026, 9, 12, 14, 0, tzinfo=timezone.utc),
                "America/New_York",
            )
        )

    def test_market_summary_uses_all_quotes_and_previous_snapshot(self) -> None:
        records = [
            {
                "symbol": "SPY", "last_trade_price": "102",
                "adjusted_previous_close": "100", "bid_price": "101.9",
                "ask_price": "102.1",
            },
            {
                "symbol": "AAA", "last_trade_price": "98",
                "adjusted_previous_close": "100", "bid_price": "97.9",
                "ask_price": "98.1",
            },
        ]
        previous = {
            "SPY": {"last_trade_price": "101"},
            "AAA": {"last_trade_price": "100"},
        }
        summary = market_summary(records, previous, ["SPY"])
        self.assertEqual(summary["advancers"], 1)
        self.assertEqual(summary["decliners"], 1)
        self.assertEqual(summary["positive_breadth"], 0.5)
        self.assertEqual(summary["fixed_etf_changes"]["SPY"], 0.02)
        self.assertEqual(summary["largest_interval_moves"][0]["symbol"], "AAA")

    def test_universe_cache_accepts_a_quiet_empty_event_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = UniverseCache(Path(directory) / "universe.json")
            large = [UniverseEntry("L", "large")]
            mid = [UniverseEntry("M", "mid")]
            small = [UniverseEntry("S", "small")]
            cache.save("2026-09-10", "afternoon", large, mid, small, [], [])
            loaded = cache.load("2026-09-10")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["event_candidates"], [])

    def test_size_buckets_keep_global_sector_cap_and_dedupe_issuer(self) -> None:
        settings = Settings.load(Path("config/settings.toml")).monitor

        def entries(prefix: str, bucket: str) -> list[UniverseEntry]:
            return [
                UniverseEntry(
                    symbol=f"{prefix}{index}", bucket=bucket,
                    score=1.0 - index / 100.0, sector=str(index % 10),
                )
                for index in range(40)
            ]

        large = [
            UniverseEntry("GOOG", "large", 2.0, "0"),
            UniverseEntry("GOOGL", "large", 1.9, "0"),
            *entries("L", "large"),
        ]
        result = assemble_universe(
            settings, [], large, entries("M", "mid"),
            entries("S", "small"), entries("E", "event"),
        )
        self.assertEqual(len(result), 60)
        symbols = {item.symbol for item in result}
        self.assertFalse({"GOOG", "GOOGL"} <= symbols)
        self.assertTrue(all(not item.investable for item in result if item.bucket == "fixed_etf"))
        sector_counts = {}
        for item in result:
            if item.bucket in {"position", "fixed_etf"}:
                continue
            sector_counts[item.sector] = sector_counts.get(item.sector, 0) + 1
        self.assertTrue(all(count <= 5 for count in sector_counts.values()))

    def test_live_requires_two_matching_switches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.toml"
            path.write_text(
                'mode="SHADOW"\nlive_trading=false\n'
                'strategy_version="x"\nprompt_version="x"\nopenai_model="x"\n'
                "max_input_tokens=100\n"
                "max_output_tokens=1\nmax_llm_calls_per_run=1\n"
                "max_tool_calls_per_run=1\n"
                "max_mcp_calls_per_decision_run=50\n"
                "max_estimated_cost_per_run_usd=0.01\n"
                "max_monthly_llm_budget_usd=1.0\n"
                "notification_required=true\n"
                "[monitor]\n"
                "interval_minutes=15\nuniverse_size=60\nquote_batch_size=20\n"
                "max_mcp_calls_per_cycle=9\nquote_batch_delay_seconds=1.0\n"
                "max_quote_age_minutes=30\n"
                'market_timezone="America/New_York"\n'
                "large_target=13\nmid_target=10\nsmall_target=7\n"
                "event_target=8\nposition_reserve=10\n"
                "large_min_price=10.0\nlarge_min_market_cap=10000000000\n"
                "large_min_average_volume=1000000\n"
                "large_min_average_dollar_volume=100000000\n"
                "mid_min_price=5.0\nmid_min_market_cap=2000000000\n"
                "mid_max_market_cap=10000000000\nmid_min_average_volume=500000\n"
                "mid_min_average_dollar_volume=30000000\n"
                "small_min_price=5.0\nsmall_min_market_cap=500000000\n"
                "small_max_market_cap=2000000000\nsmall_min_average_volume=500000\n"
                "small_min_average_dollar_volume=20000000\n"
                "event_min_market_cap=500000000\n"
                "event_min_average_volume=500000\n"
                "event_min_relative_volume=1.25\n"
                "event_min_absolute_change=0.02\n"
                "general_move_trigger=0.02\nposition_move_trigger=0.01\n"
                'fixed_etfs=["SPY","QQQ","IWM","RSP","XLK","XLF",'
                '"XLV","XLY","XLP","XLI","XLE","XLU"]\n'
                "[forecast]\nhistory_calendar_days=240\nmin_history_bars=100\n"
                "ridge_penalty=8\nmin_training_samples=500\nshrinkage=0.35\n"
                "candidate_count=3\nmin_probability_positive=0.55\n"
                "min_expected_excess_return_20d=0.005\n"
                "max_abs_forecast_20d=0.08\n"
                "[research]\ndeep_candidate_count=1\n"
                "[portfolio]\nmax_invested_fraction=1.0\n"
                "soft_max_position_fraction=0.35\n"
                "soft_max_sector_fraction=0.50\nrisk_aversion=8\n"
                "uncertainty_penalty=2\noptimizer_iterations=250\n"
                "optimizer_step_size=0.5\nmin_trade_usd=10\n"
                "[risk]\npolicy_version=\"hard_risk_v1.0\"\n"
                "allow_margin=false\nallow_options=false\n"
                "allow_short_selling=false\nallow_crypto=false\n"
                "allow_leveraged_etfs=false\nmax_position_fraction=0.2\n"
                "max_positions=10\nmax_trade_usd=100000\n"
                "max_trade_fraction=1.0\n"
                "max_daily_turnover_fraction=1.0\n"
                "max_portfolio_drawdown_fraction=0.2\n"
                "max_daily_loss_fraction=0.08\nmax_live_orders_per_day=10\n"
                "max_spread_fraction=0.003\nmax_execution_quote_age_seconds=60\n"
                "max_decision_age_seconds=180\n"
                "max_decision_price_drift_fraction=0.01\n"
                "[execution]\nmax_decision_runs_per_day=3\n"
                'decision_windows=["09:45","13:00","15:30"]\n'
                'order_type="market"\ntime_in_force="gfd"\n'
                'market_hours="regular_hours"\n'
                "[capital_plan]\ntarget_value_usd=100000\n"
                "target_horizon_years=5\n"
                "planned_contributions_usd=[1000,1000]\n"
                "planned_contribution_months=[1,2]\n",
                encoding="utf-8",
            )
            old = os.environ.get("AI_INVESTOR_LIVE_TRADING")
            os.environ["AI_INVESTOR_LIVE_TRADING"] = "true"
            try:
                with self.assertRaises(RuntimeError):
                    Settings.load(path)
            finally:
                if old is None:
                    os.environ.pop("AI_INVESTOR_LIVE_TRADING", None)
                else:
                    os.environ["AI_INVESTOR_LIVE_TRADING"] = old


if __name__ == "__main__":
    unittest.main()
