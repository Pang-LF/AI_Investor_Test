import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ai_investor.config import PortfolioSettings, RiskSettings
from ai_investor.execution import (
    BrokerState,
    Position,
    assert_live_armed,
    deterministic_ref_id,
    plan_orders,
    execute_orders,
)
from ai_investor.config import Settings
from ai_investor.forecasting import AssetForecast
from ai_investor.ledger import Ledger
from ai_investor.market_data import DailyBar
from ai_investor.portfolio import optimize_portfolio
from ai_investor.risk_engine import OrderIntent, approve_order


def risk_settings() -> RiskSettings:
    return RiskSettings(
        policy_version="hard_risk_v1.1",
        allow_margin=False,
        allow_options=False,
        allow_short_selling=False,
        allow_crypto=False,
        allow_leveraged_etfs=False,
        max_position_fraction=1.0,
        max_positions=10,
        max_trade_usd=100_000,
        max_trade_fraction=1.0,
        max_daily_turnover_fraction=1.0,
        max_portfolio_drawdown_fraction=0.20,
        max_daily_loss_fraction=0.08,
        max_live_orders_per_day=10,
        max_spread_fraction=0.003,
        max_execution_quote_age_seconds=60,
        max_decision_age_seconds=180,
        max_decision_price_drift_fraction=0.01,
    )


class FakeSettings:
    mode = "SHADOW"
    live_trading = False


class ExecutionSafetyTests(unittest.TestCase):
    def test_ref_id_is_deterministic_and_order_plan_stable(self) -> None:
        self.assertEqual(
            deterministic_ref_id("d", "SPY", "buy", 100),
            deterministic_ref_id("d", "SPY", "buy", 100),
        )
        state = BrokerState("x", 1000, 1000, 0, 0, 0, ())
        first = plan_orders(
            decision_key="d", target_weights={"SPY": 1.0}, state=state,
            prices={"SPY": 500}, minimum_trade_usd=10,
        )
        second = plan_orders(
            decision_key="d", target_weights={"SPY": 1.0}, state=state,
            prices={"SPY": 500}, minimum_trade_usd=10,
        )
        self.assertEqual(first, second)

    def test_risk_allows_full_nav_but_never_more(self) -> None:
        now = datetime.now(timezone.utc)
        quote = {
            "last_trade_price": 100,
            "bid_price": 99.9,
            "ask_price": 100.1,
            "last_trade_time": now.isoformat(),
        }
        allowed = approve_order(
            OrderIntent("AAPL", "buy", 1000), policy=risk_settings(),
            portfolio_value=1000, cash=1000, positions_value={},
            daily_order_notional=0, daily_order_count=0, daily_open_value=1000,
            high_watermark=1000, quote=quote, now=now, tradable=True,
        )
        blocked = approve_order(
            OrderIntent("AAPL", "buy", 1000.01), policy=risk_settings(),
            portfolio_value=1000, cash=2000, positions_value={},
            daily_order_notional=0, daily_order_count=0, daily_open_value=1000,
            high_watermark=1000, quote=quote, now=now, tradable=True,
        )
        self.assertTrue(allowed.approved)
        self.assertIn("trade_size_limit", blocked.reasons)

    def test_stale_quote_and_daily_loss_block_new_buy(self) -> None:
        now = datetime.now(timezone.utc)
        decision = approve_order(
            OrderIntent("AAPL", "buy", 100), policy=risk_settings(),
            portfolio_value=910, cash=910, positions_value={},
            daily_order_notional=0, daily_order_count=0, daily_open_value=1000,
            high_watermark=1000,
            quote={
                "last_trade_price": 100, "bid_price": 99.9, "ask_price": 100.1,
                "last_trade_time": (now - timedelta(minutes=5)).isoformat(),
            },
            now=now, tradable=True,
        )
        self.assertIn("stale_quote", decision.reasons)
        self.assertIn("daily_loss_circuit_breaker", decision.reasons)

    def test_old_decision_or_price_drift_blocks_order(self) -> None:
        now = datetime.now(timezone.utc)
        decision = approve_order(
            OrderIntent("AAPL", "buy", 100), policy=risk_settings(),
            portfolio_value=1000, cash=1000, positions_value={},
            daily_order_notional=0, daily_order_count=0, daily_open_value=1000,
            high_watermark=1000,
            quote={
                "last_trade_price": 102, "bid_price": 101.9, "ask_price": 102.1,
                "last_trade_time": now.isoformat(),
            },
            now=now, tradable=True,
            decision_started_at=now - timedelta(seconds=181),
            reference_price=100,
        )
        self.assertIn("decision_age_limit", decision.reasons)
        self.assertIn("decision_price_drift_limit", decision.reasons)

    def test_prohibited_exposure_and_leveraged_etf_are_blocked(self) -> None:
        now = datetime.now(timezone.utc)
        decision = approve_order(
            OrderIntent("TQQQ", "buy", 100), policy=risk_settings(),
            portfolio_value=1000, cash=1000, positions_value={},
            daily_order_notional=0, daily_order_count=0, daily_open_value=1000,
            high_watermark=1000,
            quote={
                "last_trade_price": 100, "bid_price": 99.9, "ask_price": 100.1,
                "last_trade_time": now.isoformat(),
            },
            now=now, tradable=True, option_value=1,
        )
        self.assertIn("leveraged_or_inverse_etf", decision.reasons)
        self.assertIn("prohibited_account_exposure", decision.reasons)

    def test_live_needs_both_config_and_daily_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "kill switch"):
                assert_live_armed(FakeSettings(), Path(directory), "x", "2026-01-01")

    def test_ledger_order_upsert_preserves_one_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with Ledger(Path(directory) / "ledger.sqlite") as ledger:
                for status in ("planned", "submitted"):
                    ledger.upsert_order(
                        ref_id="same", decision_key="d", trading_date="2026-01-01",
                        mode="SHADOW", symbol="SPY", side="buy", order_type="market",
                        quantity=None, dollar_amount="100.00", planned_notional=100,
                        status=status,
                    )
                row = ledger.get_order("same")
                self.assertEqual(row["status"], "submitted")
                self.assertEqual(ledger.daily_order_count("2026-01-01"), 1)

    def test_optimizer_respects_full_nav_and_position_caps(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        histories = {}
        forecasts = []
        for offset, symbol in enumerate(("AAA", "BBB")):
            histories[symbol] = [
                DailyBar(symbol, (base + timedelta(days=index)).isoformat(), 100, 101, 99,
                         100 + index * (1 + offset * 0.1), 1_000_000)
                for index in range(80)
            ]
            forecasts.append(
                AssetForecast(symbol, "2026-03-20", 0.04, 0.20, 0.8, 0.8, 0.01, 0.02, {})
            )
        portfolio = PortfolioSettings(1.0, 1.0, 0.0, 100, 0.5, 10)
        result = optimize_portfolio(forecasts, histories, portfolio, risk_settings())
        self.assertLessEqual(sum(result.weights.values()), 1.0 + 1e-9)
        self.assertTrue(all(value <= 1.0 for value in result.weights.values()))

    def test_shadow_execution_reviews_but_never_places(self) -> None:
        class FakeClient:
            def __init__(self):
                self.calls = []

            def call_tool(self, name, arguments):
                self.calls.append((name, arguments))
                if name != "review_equity_order":
                    raise AssertionError("Shadow execution attempted a write")
                return {"data": {"order_checks": {}}}

        settings = Settings.load(Path(__file__).parents[1] / "config" / "settings.toml")
        settings = replace(settings, mode="SHADOW", live_trading=False)
        now = datetime.now(timezone.utc)
        order = plan_orders(
            decision_key="shadow", target_weights={"SPY": 1.0},
            state=BrokerState("account", 1000, 1000, 0, 0, 0, ()),
            prices={"SPY": 100}, minimum_trade_usd=10,
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            with Ledger(Path(directory) / "ledger.sqlite") as ledger:
                client = FakeClient()
                result = execute_orders(
                    settings=settings, root=Path(directory), ledger=ledger,
                    client=client, state=BrokerState("account", 1000, 1000, 0, 0, 0, ()),
                    orders=[order],
                    quotes={"SPY": {
                        "last_trade_price": 100, "bid_price": 99.9,
                        "ask_price": 100.1, "venue_last_trade_time": now.isoformat(),
                    }},
                    tradability={"SPY": True}, trading_date="2026-01-01", now=now,
                    allowed_buy_symbols={"SPY"},
                )
                self.assertEqual(result[0]["status"], "hypothetical_reviewed")
                self.assertEqual([name for name, _ in client.calls], ["review_equity_order"])
                self.assertNotIn("ref_id", client.calls[0][1])


if __name__ == "__main__":
    unittest.main()
