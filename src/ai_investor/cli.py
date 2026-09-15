from __future__ import annotations

import argparse
import getpass
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .credential_store import (
    SMTPConfig,
    get_smtp_config,
    set_openai_api_key,
    set_smtp_config,
)
from .agent import run_agent_cycle
from .execution import (
    arm_live,
    assert_live_armed,
    execution_toolset,
    fetch_broker_state,
)
from .health import (
    classify_operational_failure,
    clear_operational_failures,
    report_operational_failure,
    sanitize_error,
)
from .ledger import Ledger
from .monitor import run_monitor_cycle
from .notification import send_or_queue
from .robinhood_mcp import RobinhoodMCPClient
from .robinhood_oauth import RobinhoodOAuth
from .robinhood_readonly import get_account_snapshot
from .scanners import provision_scanners


ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "config" / "settings.toml"


def main() -> None:
    parser = argparse.ArgumentParser(prog="ai-investor")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-config")
    subparsers.add_parser("oauth-login")
    subparsers.add_parser("oauth-status")
    subparsers.add_parser("set-openai-key")
    subparsers.add_parser("account-snapshot")
    scanner_parser = subparsers.add_parser("setup-scanners")
    scanner_parser.add_argument(
        "--ack",
        required=True,
        help='Must equal "CREATE ROBINHOOD SAVED SCANNERS".',
    )
    email_parser = subparsers.add_parser("set-email")
    email_parser.add_argument("--sender", required=True)
    email_parser.add_argument("--recipient", required=True)
    email_parser.add_argument("--username")
    email_parser.add_argument("--host", default="smtp.gmail.com")
    email_parser.add_argument("--port", type=int, default=465)
    subparsers.add_parser("email-status")
    subparsers.add_parser("test-email")
    monitor_parser = subparsers.add_parser("monitor-cycle")
    monitor_parser.add_argument(
        "--force",
        action="store_true",
        help="Run outside regular market hours for a read-only connectivity test.",
    )
    monitor_parser.add_argument(
        "--no-delay",
        action="store_true",
        help="Skip the one-second delay between quote batches (test use only).",
    )
    agent_parser = subparsers.add_parser("agent-cycle")
    agent_parser.add_argument("--force-monitor", action="store_true")
    agent_parser.add_argument("--no-delay", action="store_true")
    arm_parser = subparsers.add_parser("arm-live")
    arm_parser.add_argument(
        "--persistent",
        action="store_true",
        required=True,
        help="Authorize LIVE until disarmed or account/hard-risk settings change.",
    )
    arm_parser.add_argument(
        "--ack",
        required=True,
        help='Must equal "I ACCEPT LIVE TRADING RISK".',
    )
    subparsers.add_parser("live-status")
    subparsers.add_parser("event-shadow-status")
    subparsers.add_parser("research-cache-status")
    subparsers.add_parser("disarm-live")
    args = parser.parse_args()

    try:
        settings = Settings.load(SETTINGS_PATH)
    except Exception as exc:
        if args.command == "agent-cycle":
            try:
                report_operational_failure(
                    ROOT,
                    component="configuration",
                    error=f"{type(exc).__name__}: {exc}",
                    repeat_minutes=360,
                )
            except Exception:
                pass
        raise
    oauth = RobinhoodOAuth()

    if args.command == "check-config":
        print(json.dumps({"mode": settings.mode, "live_trading": settings.live_trading}))
    elif args.command == "oauth-login":
        oauth.login()
        print("Robinhood OAuth stored in the macOS Keychain.")
    elif args.command == "oauth-status":
        print(oauth.status())
    elif args.command == "set-openai-key":
        set_openai_api_key(getpass.getpass("OpenAI API key: ").strip())
        print("OpenAI API key stored in the macOS Keychain.")
    elif args.command == "account-snapshot":
        result = get_account_snapshot(settings)
        print(result.text)
        print(
            json.dumps(
                {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "estimated_cost_usd": round(result.estimated_cost_usd, 6),
                }
            )
        )
    elif args.command == "setup-scanners":
        if args.ack != "CREATE ROBINHOOD SAVED SCANNERS":
            raise RuntimeError("Exact saved-scanner acknowledgement is required")
        with RobinhoodMCPClient(
            max_calls=10,
            allowed_tools={"get_scans", "create_scan", "run_scan"},
            allow_order_submission=False,
            allow_scanner_configuration=True,
        ) as client:
            registry = provision_scanners(
                client,
                settings.monitor,
                ROOT / ".local" / "state" / "scanners.json",
            )
        print(
            json.dumps(
                {
                    key: {
                        "scan_id": "…" + record.scan_id[-8:],
                        "title": record.title,
                        "fingerprint": record.fingerprint[:12],
                    }
                    for key, record in registry.records.items()
                },
                indent=2,
            )
        )
    elif args.command == "set-email":
        password = getpass.getpass("SMTP app password (hidden): ").strip()
        set_smtp_config(
            SMTPConfig(
                host=args.host,
                port=args.port,
                username=args.username or args.sender,
                password=password,
                sender=args.sender,
                recipient=args.recipient,
            )
        )
        print("SMTP configuration stored in the macOS Keychain.")
    elif args.command == "email-status":
        config = get_smtp_config()
        print(
            json.dumps(
                {
                    "configured": config is not None,
                    "host": config.host if config else None,
                    "port": config.port if config else None,
                    "sender": config.sender if config else None,
                    "recipient": config.recipient if config else None,
                }
            )
        )
    elif args.command == "test-email":
        config = get_smtp_config()
        if config is None:
            raise RuntimeError("Run ai-investor set-email first")
        result = send_or_queue(
            ROOT,
            run_id="email-connectivity-test",
            subject="[AI Investor] Email connectivity test",
            body="Email notification is configured. This message contains no account data.",
            config=config,
        )
        print(json.dumps(asdict(result)))
    elif args.command == "monitor-cycle":
        result = run_monitor_cycle(
            settings,
            ROOT,
            force=args.force,
            no_delay=args.no_delay,
        )
        print(json.dumps(asdict(result), indent=2))
    elif args.command == "agent-cycle":
        try:
            result = run_agent_cycle(
                settings,
                ROOT,
                force_monitor=args.force_monitor,
                no_delay=args.no_delay,
            )
        except Exception as exc:
            component = classify_operational_failure(exc)
            safe_error = sanitize_error(f"{type(exc).__name__}: {exc}")
            alert_status = "failed"
            try:
                delivery = report_operational_failure(
                    ROOT,
                    component=component,
                    error=safe_error,
                    repeat_minutes=settings.health.persistent_failure_repeat_minutes,
                )
                alert_status = delivery.status
            except Exception as alert_exc:
                alert_status = f"alert_failed: {type(alert_exc).__name__}"
            print(
                json.dumps(
                    {
                        "status": "operational_failure",
                        "component": component,
                        "error": safe_error,
                        "alert": alert_status,
                    },
                    indent=2,
                )
            )
            raise SystemExit(1) from exc
        clear_operational_failures(ROOT, ("agent_runtime", "configuration"))
        print(json.dumps(asdict(result), indent=2))
    elif args.command == "arm-live":
        if args.ack != "I ACCEPT LIVE TRADING RISK":
            raise RuntimeError("Exact live-risk acknowledgement is required")
        with RobinhoodMCPClient(
            max_calls=4,
            allowed_tools=execution_toolset(False),
            allow_order_submission=False,
        ) as client:
            state = fetch_broker_state(client)
        path = arm_live(
            ROOT,
            state.account_number,
            settings,
        )
        print(
            f"Persistent LIVE authorization created at {path}; it remains bound "
            "to this Agentic account and the complete hard-risk configuration."
        )
    elif args.command == "live-status":
        try:
            with RobinhoodMCPClient(
                max_calls=4,
                allowed_tools=execution_toolset(False),
                allow_order_submission=False,
            ) as client:
                state = fetch_broker_state(client)
            assert_live_armed(settings, ROOT, state.account_number)
            payload = {
                "authorized": True,
                "authorization_mode": "persistent",
                "account": "••••" + state.account_number[-4:],
                "risk_policy_version": settings.risk.policy_version,
                "execution_calibration_approved": (
                    settings.forecast.execution_calibration_approved
                ),
                "new_order_generation_enabled": (
                    settings.forecast.execution_calibration_approved
                    and settings.forecast.twenty_day_new_entry_live_enabled
                ),
                "twenty_day_new_entry_live_enabled": (
                    settings.forecast.twenty_day_new_entry_live_enabled
                ),
                "twenty_day_existing_position_management_enabled": (
                    settings.forecast.twenty_day_existing_position_management_enabled
                ),
                "twenty_day_shadow_decisions_enabled": (
                    settings.forecast.twenty_day_shadow_decisions_enabled
                ),
                "event_engine_enabled": settings.event_engine.enabled,
                "event_engine_live_entry_enabled": (
                    settings.event_engine.live_entry_enabled
                ),
            }
        except Exception as exc:
            payload = {
                "authorized": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(payload, indent=2))
    elif args.command == "event-shadow-status":
        with Ledger(ROOT / ".local" / "state" / "ledger.sqlite") as ledger:
            rows = ledger.event_shadow_signals()
        resolved_1d = sum(row["realized_excess_1d"] is not None for row in rows)
        resolved_5d = sum(row["realized_excess_5d"] is not None for row in rows)
        latest_fields = (
            "trading_date",
            "observed_at",
            "symbol",
            "event_type",
            "signal_price",
            "realized_excess_1d",
            "realized_excess_5d",
        )
        print(json.dumps({
            "engine_enabled": settings.event_engine.enabled,
            "live_entry_enabled": settings.event_engine.live_entry_enabled,
            "signals": len(rows),
            "resolved_1d": resolved_1d,
            "resolved_5d": resolved_5d,
            "latest": [
                {field: row[field] for field in latest_fields}
                for row in rows[-10:]
            ],
        }, indent=2))
    elif args.command == "research-cache-status":
        with Ledger(ROOT / ".local" / "state" / "ledger.sqlite") as ledger:
            payload = ledger.research_assessment_cache_status(
                datetime.now(timezone.utc).isoformat()
            )
        print(json.dumps({
            "quantitative_universe_size": (
                settings.research.quantitative_universe_size
            ),
            "quantitative_shortlist_size": (
                settings.research.quantitative_shortlist_size
            ),
            "assessment_ttl_minutes": settings.research.assessment_ttl_minutes,
            **payload,
        }, indent=2))
    elif args.command == "disarm-live":
        path = ROOT / ".local" / "state" / "live_arm.json"
        path.unlink(missing_ok=True)
        print("Persistent LIVE authorization removed; order placement is blocked.")


if __name__ == "__main__":
    main()
