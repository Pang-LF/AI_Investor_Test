import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ai_investor.config import Settings
from ai_investor.market_data import DailyBar, HistoricalCache
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
from ai_investor.universe import (
    UniverseEntry,
    _rank_core,
    assemble_research_universe,
    assemble_universe,
    filter_small_candidates_by_median_liquidity,
)


class SafetyTests(unittest.TestCase):
    def test_broad_research_universe_has_two_hundred_unique_stocks(self) -> None:
        def candidates(prefix: str, bucket: str, count: int) -> list[UniverseEntry]:
            return [
                UniverseEntry(
                    symbol=f"{prefix}{index}",
                    bucket=bucket,
                    sector=str(index % 9),
                    issuer=f"{prefix} issuer {index}",
                )
                for index in range(count)
            ]

        result = assemble_research_universe(
            200,
            ["HELD"],
            candidates("L", "large", 200),
            candidates("M", "mid", 200),
            candidates("S", "small", 30),
            candidates("E", "event", 30),
        )
        self.assertEqual(len(result), 200)
        self.assertEqual(len({item.symbol for item in result}), 200)
        self.assertEqual(result[0].symbol, "HELD")

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
                "get_equity_historicals",
                "get_scans",
                "run_scan",
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
        self.assertTrue(
            quotes_are_fresh(
                [{"last_trade_time": "2026-09-10T14:59:59.123456789Z"}],
                now,
                30,
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
            snapshots = list(
                (Path(directory) / "universe_snapshots").glob(
                    "2026-09-10_afternoon_*.json"
                )
            )
            self.assertEqual(len(snapshots), 1)
            snapshot = json.loads(snapshots[0].read_text(encoding="utf-8"))
            self.assertEqual(snapshot["trading_date"], "2026-09-10")
            self.assertIn("captured_at", snapshot)

            # An unchanged composition is deduplicated rather than creating a
            # new file on every 15-minute monitor cycle.
            cache.save("2026-09-10", "afternoon", large, mid, small, [], [])
            self.assertEqual(
                len(list((Path(directory) / "universe_snapshots").glob("*.json"))),
                1,
            )

    def test_historical_cache_is_bound_to_requested_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = HistoricalCache(Path(directory))
            bars = {
                "AAA": [
                    DailyBar("AAA", "2026-09-09T00:00:00Z", 1, 1, 1, 1, 1)
                ]
            }
            cache.save("2026-09-10", ["AAA"], bars, 240)
            self.assertTrue(cache.load("2026-09-10", ["AAA"], 1, 240))
            self.assertEqual(cache.load("2026-09-10", ["AAA"], 1, 1095), {})

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

    def test_small_cap_recent_reverse_split_is_excluded(self) -> None:
        class FakeClient:
            def call_tool(self, _name, arguments):
                adjustment = arguments["adjustment_type"]
                bars = []
                for day in range(1, 26):
                    adjusted = 10.0
                    raw = 1.0 if day <= 12 else 10.0
                    bars.append(
                        {
                            "begins_at": f"2026-08-{day:02d}T00:00:00Z",
                            "close_price": str(adjusted if adjustment == "split" else raw),
                            "volume": "3000000",
                            "interpolated": False,
                        }
                    )
                return {"data": {"results": [{"symbol": "REV", "bars": bars}]}}

        settings = Settings.load(Path("config/settings.toml")).monitor
        result = filter_small_candidates_by_median_liquidity(
            FakeClient(), [UniverseEntry("REV", "small")], settings, "2026-09-11"
        )
        self.assertEqual(result, [])

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
                "max_mcp_calls_per_cycle=14\nquote_batch_delay_seconds=1.0\n"
                "max_quote_age_minutes=30\n"
                'market_timezone="America/New_York"\n'
                "large_target=15\nmid_target=12\nsmall_target=8\n"
                "event_target=8\nposition_reserve=5\n"
                "large_min_price=10.0\nlarge_min_market_cap=10000000000\n"
                "large_min_average_volume=1000000\n"
                "large_min_average_dollar_volume=100000000\n"
                "mid_min_price=5.0\nmid_min_market_cap=2000000000\n"
                "mid_max_market_cap=10000000000\nmid_min_average_volume=500000\n"
                "mid_min_average_dollar_volume=30000000\n"
                "small_min_price=5.0\nsmall_min_market_cap=500000000\n"
                "small_max_market_cap=2000000000\nsmall_min_average_volume=500000\n"
                "small_min_average_dollar_volume=20000000\n"
                "small_min_float_ratio=0.10\n"
                "small_median_liquidity_candidate_limit=20\n"
                "stable_min_ipo_age_calendar_days=252\n"
                "event_min_market_cap=500000000\n"
                "event_min_average_volume=500000\n"
                "event_min_relative_volume=1.25\n"
                "event_min_absolute_change=0.02\n"
                "general_move_trigger=0.02\nposition_move_trigger=0.01\n"
                'fixed_etfs=["SPY","QQQ","IWM","RSP","XLK","XLF",'
                '"XLV","XLY","XLP","XLI","XLE","XLU"]\n'
                "[forecast]\nhistory_calendar_days=1095\nmin_history_bars=100\n"
                "ridge_penalty=8\nmin_training_samples=500\nshrinkage=0.35\n"
                "calibration_prior_date_blocks=20\n"
                "minimum_bias_correction_date_blocks=20\n"
                "bias_bootstrap_samples=1000\n"
                "research_candidate_count=3\n"
                "research_min_raw_probability_positive=0.48\n"
                "research_min_raw_expected_excess_return_20d=0.005\n"
                "execution_calibration_approved=false\n"
                "twenty_day_new_entry_live_enabled=false\n"
                "twenty_day_existing_position_management_enabled=true\n"
                "twenty_day_shadow_decisions_enabled=true\n"
                "execution_min_calibrated_probability_positive=0.50\n"
                "execution_min_bias_adjusted_excess_return_20d=0.0\n"
                "execution_min_edge_ratio_20d=0.15\n"
                "execution_min_calibration_date_blocks=5\n"
                "holding_min_calibrated_probability_positive=0.45\n"
                "holding_min_bias_adjusted_excess_return_20d=-0.005\n"
                "holding_min_edge_ratio_20d=-0.05\n"
                "holding_exit_confirmation_runs=2\n"
                "max_abs_forecast_20d=0.08\n"
                "[research]\ndeep_candidate_count=1\nmax_sec_symbols_per_run=1\n"
                "quantitative_universe_size=200\nquantitative_shortlist_size=25\n"
                "assessment_ttl_minutes=390\nmax_fresh_llm_symbols_per_run=5\n"
                "[event_engine]\nenabled=true\nlive_entry_enabled=false\n"
                "maximum_candidates=5\nminimum_absolute_move=0.02\n"
                "strong_move_threshold=0.08\nminimum_analog_samples=30\n"
                "shrinkage_prior_samples=50\n"
                "[portfolio]\nmax_invested_fraction=1.0\n"
                "soft_max_position_fraction=0.35\n"
                "soft_max_sector_fraction=0.50\nrisk_aversion=8\n"
                "uncertainty_penalty=2\noptimizer_iterations=250\n"
                "optimizer_step_size=0.5\nmin_trade_usd=10\n"
                "reallocation_cost_fraction=0.004\n"
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
                "[execution]\nmax_event_decision_runs_per_day=3\n"
                "minimum_minutes_between_decisions=60\n"
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
