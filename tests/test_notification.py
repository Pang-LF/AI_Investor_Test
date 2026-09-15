import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from ai_investor.credential_store import SMTPConfig
from ai_investor.forecasting import AssetForecast, MarketRegime
from ai_investor.health import clear_operational_failures, report_operational_failure
from ai_investor.notification import (
    build_decision_email,
    classify_intraday_tone,
    send_or_queue,
)
from ai_investor.research import ResearchResult


class NotificationTests(unittest.TestCase):
    def test_email_explains_selected_and_rejected_names(self) -> None:
        forecasts = [
            AssetForecast("AAA", "2026-01-01", .01, .03, .6, .7, .01, .02, {}),
            AssetForecast("BBB", "2026-01-01", .00, .02, .5, .6, .01, .02, {}),
        ]
        research = ResearchResult(
            model="gpt-5.6-terra",
            assessment={
                "market_summary": "mixed",
                "candidates": [
                    {"symbol": "AAA", "verdict": "allow", "bull_case": "b",
                     "bear_case": "r", "falsification": "f", "concise_rationale": "ok"},
                    {"symbol": "BBB", "verdict": "veto", "bull_case": "b",
                     "bear_case": "r", "falsification": "f", "concise_rationale": "no"},
                ],
            },
            source_tools=("fundamentals",), input_tokens=100, output_tokens=50,
            estimated_cost_usd=.001, latency_seconds=2.5,
        )
        subject, body = build_decision_email(
            run_id="r", timestamp="2026-01-01T15:00:00Z", mode="SHADOW",
            portfolio_value=1000, cash=1000, quote_count=60, triggers=[],
            intraday_market_summary={},
            regime=MarketRegime("mixed", .01, .02, .2, .5), forecasts=forecasts,
            research=research, target_weights={"AAA": .5}, orders=[], timings={},
        )
        self.assertIn("AAA", subject)
        self.assertIn("SELECTED: target weight=50.00%", body)
        self.assertIn("NOT SELECTED: LLM verdict=veto", body)

    def test_email_distinguishes_execution_gate_from_optimizer(self) -> None:
        forecast = AssetForecast(
            "PBF", "2026-01-01", .01, .0246, .55, .598, .10, .2134, {},
            calibration_observations_20d=336,
            calibration_date_blocks_20d=7,
        )
        research = ResearchResult(
            model="gpt-5.6-terra",
            assessment={"market_summary": "mixed", "candidates": [{
                "symbol": "PBF", "verdict": "allow", "bull_case": "b",
                "bear_case": "r", "falsification": "f",
                "concise_rationale": "ok", "data_quality_severity": "none",
                "data_quality_issues": [],
            }]},
            source_tools=(), input_tokens=1, output_tokens=1,
            estimated_cost_usd=0.0, latency_seconds=1.0,
        )
        _, body = build_decision_email(
            run_id="r", timestamp="2026-01-01T15:00:00Z", mode="LIVE",
            portfolio_value=1000, cash=1000, quote_count=60, triggers=[],
            intraday_market_summary={"positive_breadth": .8,
                "fixed_etf_changes": {"SPY": .01, "QQQ": .01}},
            regime=MarketRegime("risk_off", -.01, .01, .2, .4),
            forecasts=[forecast], research=research, target_weights={}, orders=[],
            timings={}, investment_eligibility_failures={"PBF": ["edge_ratio=0.115<0.150"]},
        )
        self.assertIn("structural_regime=risk_off; intraday_tone=risk_on", body)
        self.assertIn("investment eligibility failed", body)
        self.assertIn("336/7, confidence=LOW", body)

    def test_email_marks_live_buy_as_shadow_only(self) -> None:
        forecast = AssetForecast(
            "AAA", "2026-01-01", .01, .02, .55, .60, .10, .20, {}
        )
        research = ResearchResult(
            model="gpt-5.6-terra",
            assessment={"market_summary": "mixed", "candidates": [{
                "symbol": "AAA", "verdict": "allow", "concise_rationale": "ok",
            }]},
            source_tools=(), input_tokens=1, output_tokens=1,
            estimated_cost_usd=0.0, latency_seconds=1.0,
        )
        _, body = build_decision_email(
            run_id="r", timestamp="2026-01-01T15:00:00Z", mode="LIVE",
            portfolio_value=1000, cash=1000, quote_count=60, triggers=[],
            intraday_market_summary={},
            regime=MarketRegime("mixed", 0, 0, .2, .5),
            forecasts=[forecast], research=research,
            target_weights={"AAA": .1},
            orders=[{"symbol": "AAA", "status": "shadow_20d_buy_reviewed"}],
            timings={}, twenty_day_new_entry_live_enabled=False,
        )
        self.assertIn("SHADOW 20D BUY", body)
        self.assertIn("20D new-entry LIVE permission: DISABLED", body)

    def test_intraday_tone_has_a_mixed_middle_state(self) -> None:
        self.assertEqual(
            classify_intraday_tone({"positive_breadth": .5,
                "fixed_etf_changes": {"SPY": .001, "QQQ": -.001}}),
            "mixed",
        )

    def test_failed_delivery_is_persisted_to_outbox(self) -> None:
        config = SMTPConfig("smtp.example.com", 465, "u", "p", "a@b.com", "c@d.com")
        with tempfile.TemporaryDirectory() as directory:
            with patch("ai_investor.notification._deliver", side_effect=OSError("offline")):
                result = send_or_queue(
                    Path(directory), run_id="r", subject="s", body="b", config=config
                )
            self.assertEqual(result.status, "queued")
            self.assertTrue((Path(directory) / ".local/notification_outbox/r.json").exists())

    def test_operational_failure_alert_is_rate_limited(self) -> None:
        config = SMTPConfig("smtp.example.com", 465, "u", "p", "a@b.com", "c@d.com")
        start = datetime(2026, 9, 11, 14, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("ai_investor.notification.get_smtp_config", return_value=config):
                with patch("ai_investor.notification._deliver") as deliver:
                    first = report_operational_failure(
                        root, component="live_authorization", error="missing",
                        repeat_minutes=360, now=start,
                    )
                    second = report_operational_failure(
                        root, component="live_authorization", error="missing",
                        repeat_minutes=360, now=start + timedelta(minutes=15),
                    )
                    third = report_operational_failure(
                        root, component="live_authorization", error="missing",
                        repeat_minutes=360, now=start + timedelta(minutes=361),
                    )
            self.assertEqual(first.status, "sent")
            self.assertEqual(second.status, "suppressed")
            self.assertEqual(third.status, "sent")
            self.assertEqual(deliver.call_count, 2)

    def test_degraded_alert_waits_for_threshold_and_resets_silently(self) -> None:
        config = SMTPConfig("smtp.example.com", 465, "u", "p", "a@b.com", "c@d.com")
        start = datetime(2026, 9, 11, 14, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("ai_investor.notification.get_smtp_config", return_value=config):
                with patch("ai_investor.notification._deliver") as deliver:
                    first = report_operational_failure(
                        root, component="market_data", error="stale",
                        repeat_minutes=360, occurrence_threshold=2, now=start,
                    )
                    second = report_operational_failure(
                        root, component="market_data", error="stale",
                        repeat_minutes=360, occurrence_threshold=2,
                        now=start + timedelta(minutes=15),
                    )
                    clear_operational_failures(root, ("market_data",))
                    after_reset = report_operational_failure(
                        root, component="market_data", error="stale",
                        repeat_minutes=360, occurrence_threshold=2,
                        now=start + timedelta(minutes=30),
                    )
            self.assertEqual(first.status, "suppressed")
            self.assertEqual(second.status, "sent")
            self.assertEqual(after_reset.status, "suppressed")
            self.assertEqual(deliver.call_count, 1)


if __name__ == "__main__":
    unittest.main()
