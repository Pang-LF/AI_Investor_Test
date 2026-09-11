from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import keyring


SERVICE = "ai-investor-test"


def get_openai_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY") or keyring.get_password(SERVICE, "openai-api-key")


def set_openai_api_key(value: str) -> None:
    if not value.startswith("sk-"):
        raise ValueError("This does not look like an OpenAI API key")
    keyring.set_password(SERVICE, "openai-api-key", value)


@dataclass(frozen=True)
class SMTPConfig:
    host: str
    port: int
    username: str
    password: str
    sender: str
    recipient: str


def get_smtp_config() -> Optional[SMTPConfig]:
    raw = keyring.get_password(SERVICE, "smtp-config")
    if not raw:
        return None
    try:
        value = json.loads(raw)
        return SMTPConfig(
            host=str(value["host"]),
            port=int(value["port"]),
            username=str(value["username"]),
            password=str(value["password"]),
            sender=str(value["sender"]),
            recipient=str(value["recipient"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def set_smtp_config(config: SMTPConfig) -> None:
    if not config.password or "@" not in config.sender or "@" not in config.recipient:
        raise ValueError("Valid sender, recipient, and SMTP credential are required")
    keyring.set_password(SERVICE, "smtp-config", json.dumps(asdict(config)))
