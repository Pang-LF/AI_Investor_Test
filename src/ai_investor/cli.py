from __future__ import annotations

import argparse
import getpass
import json
from dataclasses import asdict
from pathlib import Path

from .config import Settings
from .credential_store import set_openai_api_key
from .agent import run_agent_cycle
from .execution import arm_live, execution_toolset, fetch_broker_state
from .monitor import run_monitor_cycle
from .robinhood_mcp import RobinhoodMCPClient
from .robinhood_oauth import RobinhoodOAuth
from .robinhood_readonly import get_account_snapshot


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
    arm_parser.add_argument("--date", required=True, help="Trading date, YYYY-MM-DD")
    arm_parser.add_argument(
        "--ack",
        required=True,
        help='Must equal "I ACCEPT LIVE TRADING RISK".',
    )
    subparsers.add_parser("disarm-live")
    args = parser.parse_args()

    settings = Settings.load(SETTINGS_PATH)
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
    elif args.command == "monitor-cycle":
        result = run_monitor_cycle(
            settings,
            ROOT,
            force=args.force,
            no_delay=args.no_delay,
        )
        print(json.dumps(asdict(result), indent=2))
    elif args.command == "agent-cycle":
        result = run_agent_cycle(
            settings, ROOT, force_monitor=args.force_monitor, no_delay=args.no_delay
        )
        print(json.dumps(asdict(result), indent=2))
    elif args.command == "arm-live":
        if args.ack != "I ACCEPT LIVE TRADING RISK":
            raise RuntimeError("Exact live-risk acknowledgement is required")
        with RobinhoodMCPClient(
            max_calls=3,
            allowed_tools=execution_toolset(False),
            allow_order_submission=False,
        ) as client:
            state = fetch_broker_state(client)
        path = arm_live(ROOT, state.account_number, args.date)
        print(f"Daily live arm created at {path}; config kill switch is unchanged.")
    elif args.command == "disarm-live":
        path = ROOT / ".local" / "state" / "live_arm.json"
        path.unlink(missing_ok=True)
        print("LIVE disarmed. Config kill switch should also remain false.")


if __name__ == "__main__":
    main()
