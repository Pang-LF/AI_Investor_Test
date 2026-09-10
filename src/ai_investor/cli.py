from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path

from .config import Settings
from .credential_store import set_openai_api_key
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


if __name__ == "__main__":
    main()
