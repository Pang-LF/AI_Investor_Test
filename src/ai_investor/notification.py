from __future__ import annotations

import json
import smtplib
import time
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .credential_store import SMTPConfig, get_smtp_config
from .forecasting import AssetForecast, MarketRegime
from .research import ResearchResult


@dataclass(frozen=True)
class DeliveryResult:
    status: str
    detail: str = ""


def classify_intraday_tone(summary: Mapping[str, Any]) -> str:
    breadth = float(summary.get("positive_breadth") or 0.0)
    etfs = summary.get("fixed_etf_changes") or {}
    spy = float(etfs.get("SPY") or 0.0)
    qqq = float(etfs.get("QQQ") or 0.0)
    if breadth >= 0.65 and spy >= 0.005 and qqq >= 0.005:
        return "risk_on"
    if breadth <= 0.35 and spy <= -0.005 and qqq <= -0.005:
        return "risk_off"
    return "mixed"


def _calibration_confidence(blocks: int) -> str:
    if blocks < 10:
        return "LOW"
    if blocks < 20:
        return "MEDIUM"
    return "HIGH"


def require_email_configuration() -> SMTPConfig:
    config = get_smtp_config()
    if config is None:
        raise RuntimeError(
            "Email notification is required but SMTP is not configured; "
            "run ai-investor set-email"
        )
    return config


def build_decision_email(
    *,
    run_id: str,
    timestamp: str,
    mode: str,
    portfolio_value: float,
    cash: float,
    quote_count: int,
    triggers: Sequence[Mapping[str, Any]],
    intraday_market_summary: Mapping[str, Any],
    regime: MarketRegime,
    forecasts: Sequence[AssetForecast],
    research: ResearchResult,
    target_weights: Mapping[str, float],
    orders: Sequence[Mapping[str, Any]],
    timings: Mapping[str, float],
    investment_eligibility_failures: Optional[Mapping[str, Sequence[str]]] = None,
    holding_decisions: Optional[Mapping[str, Mapping[str, Any]]] = None,
    portfolio_diagnostics: Optional[Mapping[str, Any]] = None,
    execution_calibration_approved: bool = True,
    pipeline_error: Optional[str] = None,
) -> tuple[str, str]:
    verdicts = {
        str(item.get("symbol", "")).upper(): item
        for item in research.assessment.get("candidates", [])
    }
    gate_failures = investment_eligibility_failures or {}
    holding_statuses = holding_decisions or {}
    order_symbols = {
        str(item.get("symbol") or item.get("ticker") or "").upper()
        for item in orders
    }
    lines = [
        f"Run: {run_id}",
        f"Time: {timestamp}",
        f"Mode: {mode}",
        f"Portfolio: ${portfolio_value:,.2f}; cash: ${cash:,.2f}",
        (
            f"Market data: {quote_count} fresh quotes; "
            f"structural_regime={regime.label}; "
            f"intraday_tone={classify_intraday_tone(intraday_market_summary)}"
        ),
        (
            "Regime metrics: SPY 20d={:.2%}, SPY 60d={:.2%}, "
            "vol={:.2%}, breadth={:.2%}"
        ).format(
            regime.spy_return_20d,
            regime.spy_return_60d,
            regime.annualized_volatility_20d,
            regime.positive_breadth_20d,
        ),
        f"Triggers: {json.dumps(list(triggers)[:10], ensure_ascii=False, default=str)}",
        "Intraday market summary: "
        + json.dumps(dict(intraday_market_summary), ensure_ascii=False, default=str),
        "",
        "Candidate analysis:",
    ]
    for forecast in forecasts:
        item = verdicts.get(forecast.symbol, {})
        verdict = str(item.get("verdict", "missing_assessment"))
        rationale = str(item.get("concise_rationale", "No LLM rationale returned"))
        target = target_weights.get(forecast.symbol, 0.0)
        holding_state = str(
            holding_statuses.get(forecast.symbol, {}).get("status", "")
        )
        if not execution_calibration_approved:
            action_reason = "NOT TRADED: execution calibration is not approved"
        elif target > 0 and holding_state == "pending_exit_retained":
            action_reason = (
                f"RETAINED_PENDING_EXIT_CONFIRMATION: "
                f"target weight={target:.2%}"
            )
        elif target > 0 and holding_state == "holding_gate_passed":
            action_reason = (
                f"REBALANCED: target weight={target:.2%}"
                if forecast.symbol in order_symbols
                else f"RETAINED_NO_REBALANCE: target weight={target:.2%}"
            )
        elif verdict != "allow":
            action_reason = f"NOT SELECTED: LLM verdict={verdict}"
        elif gate_failures.get(forecast.symbol):
            action_reason = (
                "NOT SELECTED: investment eligibility failed: "
                + "; ".join(gate_failures[forecast.symbol])
            )
        elif target <= 0:
            action_reason = "NOT SELECTED: optimizer assigned zero weight"
        else:
            action_reason = f"SELECTED: target weight={target:.2%}"
        lines.extend(
            [
                (
                    f"- {forecast.symbol}: model forecast 20d="
                    f"{forecast.model_prediction_excess_return_20d:.2%} × "
                    f"shrinkage {forecast.forecast_shrinkage:.2f} = "
                    f"shrunk forecast "
                    f"{forecast.raw_expected_excess_return_20d:.2%}, "
                    f"raw validation bias={forecast.validation_bias_20d:+.2%}, "
                    f"bias interval=[{forecast.validation_bias_interval_20d[0]:+.2%}, "
                    f"{forecast.validation_bias_interval_20d[1]:+.2%}], "
                    f"applied bias={forecast.applied_validation_bias_20d:+.2%}, "
                    f"bias-adjusted expected excess 5d="
                    f"{forecast.expected_excess_return_5d:.2%}, 20d="
                    f"{forecast.expected_excess_return_20d:.2%}, "
                    f"raw P={forecast.raw_probability_positive_excess_20d:.1%}, "
                    f"calibrated P(20d>SPY)="
                    f"{forecast.probability_positive_excess_20d:.1%}, "
                    f"date-clustered 95% interval="
                    f"[{forecast.probability_positive_excess_20d_interval[0]:.1%}, "
                    f"{forecast.probability_positive_excess_20d_interval[1]:.1%}], "
                    f"calibration observations/date blocks="
                    f"{forecast.calibration_observations_20d}/"
                    f"{forecast.calibration_date_blocks_20d}, "
                    f"confidence={_calibration_confidence(forecast.calibration_date_blocks_20d)}"
                ),
                f"  {action_reason}",
                (
                    "  Holding state: "
                    + json.dumps(
                        holding_statuses.get(forecast.symbol, {}),
                        ensure_ascii=False,
                        default=str,
                    )
                    if forecast.symbol in holding_statuses
                    else "  Holding state: not_currently_held"
                ),
                f"  Rationale: {rationale}",
                (
                    f"  Data quality: {item.get('data_quality_severity', 'none')}; "
                    f"issues={item.get('data_quality_issues', [])}"
                ),
                f"  Bull: {item.get('bull_case', '')}",
                f"  Bear: {item.get('bear_case', '')}",
                f"  Falsification: {item.get('falsification', '')}",
            ]
        )
    lines.extend(
        [
            "",
            "Order outcomes:",
            json.dumps(list(orders), ensure_ascii=False, indent=2, default=str),
            "",
            (
                f"LLM: model={research.model}, input={research.input_tokens}, "
                f"output={research.output_tokens}, "
                f"cost=${research.estimated_cost_usd:.6f}, "
                f"latency={research.latency_seconds:.2f}s"
            ),
            "Pipeline timings: " + json.dumps(dict(timings), sort_keys=True),
            "Portfolio sizing diagnostics: "
            + json.dumps(
                dict(portfolio_diagnostics or {}),
                ensure_ascii=False,
                default=str,
            ),
            (
                "Forecast calibration gate: PROVISIONALLY_ENABLED"
                if execution_calibration_approved
                else "Forecast calibration gate: RESEARCH_ONLY; orders blocked"
            ),
        ]
    )
    if pipeline_error:
        lines.extend(["", "Post-LLM pipeline error: " + pipeline_error])
    selected = ", ".join(
        sorted(symbol for symbol, weight in target_weights.items() if weight > 0)
    ) or "CASH"
    subject = f"[AI Investor] {mode} decision {selected} | {timestamp[:16]}"
    return subject, "\n".join(lines)


def _deliver(config: SMTPConfig, subject: str, body: str) -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config.sender
    message["To"] = config.recipient
    message.set_content(body)
    with smtplib.SMTP_SSL(config.host, config.port, timeout=15) as smtp:
        smtp.login(config.username, config.password)
        smtp.send_message(message)


def send_or_queue(
    root: Path,
    *,
    run_id: str,
    subject: str,
    body: str,
    config: Optional[SMTPConfig] = None,
) -> DeliveryResult:
    selected = config or get_smtp_config()
    outbox = root / ".local" / "notification_outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    queued = outbox / f"{run_id}.json"
    if selected is None:
        queued.write_text(
            json.dumps({"run_id": run_id, "subject": subject, "body": body}),
            encoding="utf-8",
        )
        return DeliveryResult("queued", "SMTP is not configured")
    last_error = ""
    for attempt in range(2):
        try:
            _deliver(selected, subject, body)
            queued.unlink(missing_ok=True)
            return DeliveryResult("sent")
        except (OSError, smtplib.SMTPException) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == 0:
                time.sleep(1)
    queued.write_text(
        json.dumps({"run_id": run_id, "subject": subject, "body": body}),
        encoding="utf-8",
    )
    return DeliveryResult("queued", last_error)


def flush_outbox(
    root: Path, config: Optional[SMTPConfig] = None, limit: int = 5
) -> int:
    selected = config or get_smtp_config()
    if selected is None:
        return 0
    outbox = root / ".local" / "notification_outbox"
    if not outbox.exists():
        return 0
    sent = 0
    for path in sorted(outbox.glob("*.json"))[:limit]:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        result = send_or_queue(
            root,
            run_id=str(item.get("run_id") or path.stem),
            subject=str(item.get("subject") or "AI Investor notification"),
            body=str(item.get("body") or ""),
            config=selected,
        )
        if result.status != "sent":
            break
        sent += 1
    return sent
