from __future__ import annotations

import os
from typing import Optional

import keyring


SERVICE = "ai-investor-test"


def get_openai_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY") or keyring.get_password(SERVICE, "openai-api-key")


def set_openai_api_key(value: str) -> None:
    if not value.startswith("sk-"):
        raise ValueError("This does not look like an OpenAI API key")
    keyring.set_password(SERVICE, "openai-api-key", value)
