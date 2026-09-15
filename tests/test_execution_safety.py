import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from ai_investor.config import PortfolioSettings, RiskSettings
from ai_investor.execution import (
    BrokerState,
    Position,
    PlannedOrder,
    arm_live,
    assert_live_armed,
    deterministic_ref_id,
    plan_orders,
    execute_orders,
    fetch_broker_state,
    reconcile_orders,
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

    def test_live_needs_both_config_and_persistent_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "kill switch"):
                assert_live_armed(FakeSettings(), Path(directory), "x", "2026-01-01")

    def test_persistent_authorization_allows_strategy_change(self) -> None:
        settings = SimpleNamespace(
            mode="LIVE",
            live_trading=True,
            strategy_version="strategy_v1",
            risk=risk_settings(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm_live(root, "account", settings)
            assert_live_armed(settings, root, "account")
            changed = SimpleNamespace(
                mode="LIVE",
                live_trading=True,
                strategy_version="strategy_v2",
                risk=risk_settings(),
            )
            assert_live_armed(changed, root, "account")

    def test_persistent_authorization_is_bound_to_account_and_all_risk_fields(self) -> None:
        settings = SimpleNamespace(
            mode="LIVE",
            live_trading=True,
            strategy_version="strategy_v1",
            risk=risk_settings(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm_live(root, "account", settings)
            with self.assertRaisesRegex(RuntimeError, "selected account"):
                assert_live_armed(settings, root, "different-account")
            changed = SimpleNamespace(
                mode="LIVE",
                live_trading=True,
                strategy_version="strategy_v1",
                risk=replace(risk_settings(), max_daily_loss_fraction=0.07),
            )
            with self.assertRaisesRegex(RuntimeError, "hard-risk settings"):
                assert_live_armed(changed, root, "account")

    def test_legacy_daily_arm_cannot_authorize_persistent_live(self) -> None:
        settings = SimpleNamespace(
            mode="LIVE",
            live_trading=True,
            strategy_version="strategy_v1",
            risk=risk_settings(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / ".local/state/live_arm.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"schema_version":2}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "current schema"):
                assert_live_armed(settings, root, "account")

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

    def test_failed_pre_submission_orders_do_not_consume_daily_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with Ledger(Path(directory) / "ledger.sqlite") as ledger:
                for index, status in enumerate(
                    ("review_rejected", "review_failed", "placement_failed")
                ):
                    ledger.upsert_order(
                        ref_id=f"failed-{index}", decision_key=f"d-{index}",
                        trading_date="2026-01-01", mode="LIVE", symbol="SPY",
                        side="buy", order_type="market", quantity=None,
                        dollar_amount="100.00", planned_notional=100,
                        status=status,
                    )
                self.assertEqual(ledger.daily_order_count("2026-01-01"), 0)
                self.assertEqual(ledger.daily_order_notional("2026-01-01"), 0)

    def test_holding_exit_confirmation_is_idempotent_per_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with Ledger(Path(directory) / "ledger.sqlite") as ledger:
                first = ledger.update_holding_exit_signal(
                    symbol="AAA", decision_key="d1", failing=True, reason="weak"
                )
                duplicate = ledger.update_holding_exit_signal(
                    symbol="AAA", decision_key="d1", failing=True, reason="weak"
                )
                second = ledger.update_holding_exit_signal(
                    symbol="AAA", decision_key="d2", failing=True, reason="weak"
                )
                reset = ledger.update_holding_exit_signal(
                    symbol="AAA", decision_key="d3", failing=False, reason="pass"
                )
                self.assertEqual((first, duplicate, second, reset), (1, 1, 2, 0))
                ledger.prune_holding_exit_states(set())
                self.assertIsNone(ledger.get_holding_exit_state("AAA"))

    def test_fetch_broker_state_values_positions_with_quotes(self) -> None:
        class FakeClient:
            def call_tool(self, name, arguments):
                if name == "get_accounts":
                    return {"data": {"accounts": [{
                        "agentic_allowed": True, "state": "active",
                        "account_number": "account",
                    }]}}
                if name == "get_portfolio":
                    return {"data": {"total_value": "1000", "cash": "950"}}
                if name == "get_equity_positions":
                    return {"data": {"positions": [{
                        "symbol": "AAA", "quantity": "0.5",
                        "shares_available_for_sells": "0.4",
                    }]}}
                if name == "get_equity_quotes":
                    return {"data": {"results": [{"quote": {
                        "symbol": "AAA", "last_trade_price": "100",
                    }}]}}
                raise AssertionError(name)

        state = fetch_broker_state(FakeClient())
        self.assertEqual(state.positions[0].market_value, 50.0)
        self.assertEqual(state.positions[0].sellable_quantity, 0.4)

    def test_nested_placement_id_is_stored_and_reconciled(self) -> None:
        class FakeClient:
            def __init__(self):
                self.placed = False

            def call_tool(self, name, arguments):
                if name == "review_equity_order":
                    return {"data": {"order_checks": {}}}
                if name == "place_equity_order":
                    self.placed = True
                    return {"data": {"order": {
                        "id": "broker-1", "state": "unconfirmed",
                    }}}
                if name == "get_equity_orders":
                    return {"data": {"orders": [{
                        "id": "broker-1", "symbol": "AAA", "side": "buy",
                        "state": "filled", "cumulative_quantity": "0.5",
                        "average_price": "100",
                    }]}}
                raise AssertionError(name)

        settings = Settings.load(Path(__file__).parents[1] / "config" / "settings.toml")
        settings = replace(
            settings,
            mode="LIVE",
            live_trading=True,
            forecast=replace(
                settings.forecast, twenty_day_new_entry_live_enabled=True
            ),
        )
        now = datetime.now(timezone.utc)
        state = BrokerState("account", 1000, 1000, 0, 0, 0, ())
        order = plan_orders(
            decision_key="live", target_weights={"AAA": 0.05}, state=state,
            prices={"AAA": 100}, minimum_trade_usd=10,
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm_live(root, "account", settings)
            with Ledger(root / "ledger.sqlite") as ledger:
                client = FakeClient()
                result = execute_orders(
                    settings=settings, root=root, ledger=ledger, client=client,
                    state=state, orders=[order], quotes={"AAA": {
                        "last_trade_price": 100, "bid_price": 99.9,
                        "ask_price": 100.1, "venue_last_trade_time": now.isoformat(),
                    }}, tradability={"AAA": True}, trading_date="2026-01-01",
                    now=now, allowed_buy_symbols={"AAA"},
                )
                self.assertEqual(result[0]["broker_order_id"], "broker-1")
                filled_state = BrokerState(
                    "account", 1000, 950, 0, 0, 0,
                    (Position("AAA", 0.5, 50),),
                )
                self.assertEqual(
                    reconcile_orders(client, ledger, filled_state, "2026-01-01"), 1
                )
                row = ledger.get_order(order.ref_id)
                self.assertEqual(row["status"], "filled")
                self.assertEqual(row["average_fill_price"], 100.0)

    def test_live_twenty_day_buy_is_shadow_only_while_sell_can_execute(self) -> None:
        class FakeClient:
            def __init__(self):
                self.calls = []

            def call_tool(self, name, arguments):
                self.calls.append((name, dict(arguments)))
                if name == "review_equity_order":
                    return {"data": {"order_checks": {}}}
                if name == "place_equity_order":
                    return {"data": {"order": {"id": "broker-sell"}}}
                raise AssertionError(name)

        settings = Settings.load(Path(__file__).parents[1] / "config" / "settings.toml")
        settings = replace(settings, mode="LIVE", live_trading=True)
        self.assertFalse(settings.forecast.twenty_day_new_entry_live_enabled)
        self.assertTrue(
            settings.forecast.twenty_day_existing_position_management_enabled
        )
        now = datetime.now(timezone.utc)
        state = BrokerState(
            "account", 1000, 500, 0, 0, 0,
            (Position("HELD", 5, 500, sellable_quantity=5),),
        )
        orders = plan_orders(
            decision_key="permission-split",
            target_weights={"HELD": 0.4, "NEW": 0.1},
            state=state,
            prices={"HELD": 100, "NEW": 100},
            minimum_trade_usd=10,
        )
        quotes = {
            symbol: {
                "last_trade_price": 100,
                "bid_price": 99.9,
                "ask_price": 100.1,
                "venue_last_trade_time": now.isoformat(),
            }
            for symbol in ("HELD", "NEW")
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm_live(root, "account", settings)
            with Ledger(root / "ledger.sqlite") as ledger:
                client = FakeClient()
                results = execute_orders(
                    settings=settings,
                    root=root,
                    ledger=ledger,
                    client=client,
                    state=state,
                    orders=orders,
                    quotes=quotes,
                    tradability={"HELD": True, "NEW": True},
                    trading_date="2026-01-01",
                    now=now,
                    allowed_buy_symbols={"NEW"},
                )
                by_symbol = {item["symbol"]: item for item in results}
                self.assertEqual(by_symbol["HELD"]["status"], "submitted")
                self.assertEqual(
                    by_symbol["NEW"]["status"], "shadow_20d_buy_reviewed"
                )
                placed_symbols = [
                    arguments["symbol"]
                    for name, arguments in client.calls
                    if name == "place_equity_order"
                ]
                self.assertEqual(placed_symbols, ["HELD"])
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
        portfolio = PortfolioSettings(
            1.0, 0.35, 0.50, 1.0, 0.0, 100, 0.5, 10, 0.004
        )
        result = optimize_portfolio(
            forecasts, histories, portfolio, risk_settings(),
            {"AAA": "technology", "BBB": "technology"},
        )
        self.assertLessEqual(sum(result.weights.values()), 1.0 + 1e-9)
        self.assertTrue(all(value <= 0.35 for value in result.weights.values()))
        self.assertLessEqual(sum(result.weights.values()), 0.50 + 1e-9)

    def test_optimizer_requires_improvement_after_reallocation_cost(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = [
            DailyBar("AAA", (base + timedelta(days=index)).isoformat(), 100, 101, 99,
                     100 + index * 0.1, 1_000_000)
            for index in range(80)
        ]
        forecast = AssetForecast(
            "AAA", "2026-03-20", 0.0, 0.001, 0.5, 0.5, 0.05, 0.10, {}
        )
        portfolio = PortfolioSettings(
            1.0, 0.35, 0.50, 1.0, 0.0, 100, 0.5, 10, 0.004
        )
        result = optimize_portfolio(
            [forecast], {"AAA": bars}, portfolio, risk_settings()
        )
        self.assertEqual(result.weights, {})

    def test_optimizer_respects_pending_exit_weight_floor(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = [
            DailyBar("AAA", (base + timedelta(days=index)).isoformat(), 100, 101, 99,
                     100 + index * 0.1, 1_000_000)
            for index in range(80)
        ]
        forecast = AssetForecast(
            "AAA", "2026-03-20", -0.01, -0.01, 0.4, 0.4, 0.05, 0.10, {}
        )
        portfolio = PortfolioSettings(
            1.0, 0.35, 0.50, 1.0, 0.0, 100, 0.5, 10, 0.004
        )
        result = optimize_portfolio(
            [forecast], {"AAA": bars}, portfolio, risk_settings(),
            current_weights={"AAA": 0.05}, minimum_weights={"AAA": 0.05},
        )
        self.assertGreaterEqual(result.weights["AAA"], 0.05 - 1e-8)

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

    def test_fractional_sell_is_floored_to_six_decimal_places(self) -> None:
        class FakeClient:
            def __init__(self):
                self.arguments = None

            def call_tool(self, name, arguments):
                self.assert_review(name)
                self.arguments = arguments
                return {"data": {"order_checks": {}}}

            @staticmethod
            def assert_review(name):
                if name != "review_equity_order":
                    raise AssertionError("Shadow execution attempted a write")

        settings = Settings.load(Path(__file__).parents[1] / "config" / "settings.toml")
        settings = replace(settings, mode="SHADOW", live_trading=False)
        now = datetime.now(timezone.utc)
        order = PlannedOrder(
            ref_id="fractional-sell", decision_key="fractional-sell",
            symbol="AVAV", side="sell", planned_notional=16.28,
            quantity=0.11141349,
        )
        state = BrokerState(
            "account", 1000, 950, 0, 0, 0,
            (Position("AVAV", 0.331622, 48.46, 0.331622),),
        )
        with tempfile.TemporaryDirectory() as directory:
            with Ledger(Path(directory) / "ledger.sqlite") as ledger:
                client = FakeClient()
                execute_orders(
                    settings=settings, root=Path(directory), ledger=ledger,
                    client=client, state=state, orders=[order],
                    quotes={"AVAV": {
                        "last_trade_price": 146.13, "bid_price": 146.10,
                        "ask_price": 146.15,
                        "venue_last_trade_time": now.isoformat(),
                    }},
                    tradability={"AVAV": True}, trading_date="2026-01-01",
                    now=now,
                )
                self.assertEqual(client.arguments["quantity"], "0.111413")
                self.assertEqual(
                    ledger.get_order("fractional-sell")["quantity"], "0.111413"
                )

    def test_placement_exception_is_recorded_before_reraising(self) -> None:
        class FakeClient:
            def call_tool(self, name, arguments):
                if name == "review_equity_order":
                    return {"data": {"order_checks": {}}}
                if name == "place_equity_order":
                    raise RuntimeError("broker rejected test order")
                raise AssertionError(name)

        settings = Settings.load(Path(__file__).parents[1] / "config" / "settings.toml")
        settings = replace(settings, mode="LIVE", live_trading=True)
        now = datetime.now(timezone.utc)
        order = PlannedOrder(
            ref_id="failed-placement", decision_key="failed-placement",
            symbol="AVAV", side="sell", planned_notional=16.28,
            quantity=0.11141349,
        )
        state = BrokerState(
            "account", 1000, 950, 0, 0, 0,
            (Position("AVAV", 0.331622, 48.46, 0.331622),),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm_live(root, "account", settings)
            with Ledger(root / "ledger.sqlite") as ledger:
                with self.assertRaisesRegex(RuntimeError, "broker rejected"):
                    execute_orders(
                        settings=settings, root=root, ledger=ledger,
                        client=FakeClient(), state=state, orders=[order],
                        quotes={"AVAV": {
                            "last_trade_price": 146.13, "bid_price": 146.10,
                            "ask_price": 146.15,
                            "venue_last_trade_time": now.isoformat(),
                        }},
                        tradability={"AVAV": True}, trading_date="2026-01-01",
                        now=now,
                    )
                row = ledger.get_order("failed-placement")
                self.assertEqual(row["status"], "placement_failed")
                self.assertEqual(row["quantity"], "0.111413")
                self.assertIn("broker rejected test order", row["response_json"])


if __name__ == "__main__":
    unittest.main()
