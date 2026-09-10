import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ai_investor.config import Settings
from ai_investor.monitor import (
    is_regular_market_window,
    quote_batches,
    quotes_are_fresh,
)
from ai_investor.robinhood_mcp import (
    KNOWN_MUTATING_TOOLS,
    MONITOR_READ_ONLY_TOOLS,
)
from ai_investor.robinhood_readonly import READ_ONLY_TOOLS, WRITE_TOOLS
from ai_investor.universe import _rank_core


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
                "[monitor]\n"
                "interval_minutes=15\nuniverse_size=60\nquote_batch_size=20\n"
                "max_mcp_calls_per_cycle=8\nquote_batch_delay_seconds=1.0\n"
                "max_quote_age_minutes=30\n"
                'market_timezone="America/New_York"\n'
                "core_target=33\nevent_target=10\nposition_reserve=5\n"
                "core_min_price=10.0\ncore_min_market_cap=5000000000\n"
                "core_min_average_volume=1000000\n"
                "core_min_average_dollar_volume=100000000\n"
                "event_min_market_cap=1000000000\n"
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
                "[portfolio]\nmax_invested_fraction=1.0\nrisk_aversion=8\n"
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
