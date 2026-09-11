from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from openai import OpenAI

from .config import Settings
from .robinhood_oauth import MCP_URL, RobinhoodOAuth
from .credential_store import get_openai_api_key
from .research import estimate_model_cost


READ_ONLY_TOOLS = [
    "get_accounts",
    "get_portfolio",
    "get_equity_positions",
    "get_equity_orders",
]

WRITE_TOOLS = {
    "place_equity_order",
    "cancel_equity_order",
    "place_option_order",
    "cancel_option_order",
    "place_crypto_order",
    "cancel_crypto_order",
}


@dataclass(frozen=True)
class SnapshotResult:
    text: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


def get_account_snapshot(settings: Settings) -> SnapshotResult:
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    if set(READ_ONLY_TOOLS) & WRITE_TOOLS:
        raise RuntimeError("Safety invariant violated: write tool in read-only allowlist")

    response = OpenAI(api_key=api_key).responses.create(
        model=settings.openai_model,
        store=False,
        reasoning={"effort": "low"},
        max_output_tokens=settings.max_output_tokens,
        max_tool_calls=settings.max_tool_calls_per_run,
        parallel_tool_calls=False,
        tools=[
            {
                "type": "mcp",
                "server_label": "robinhood",
                "server_description": (
                    "Official Robinhood Agentic Trading MCP. This run is read-only."
                ),
                "server_url": MCP_URL,
                "authorization": RobinhoodOAuth().access_token(),
                "allowed_tools": READ_ONLY_TOOLS,
                "require_approval": "never",
            }
        ],
        input=(
            "Read the available Robinhood accounts, portfolio, equity positions, "
            "and open/recent equity orders. Do not make recommendations and do not "
            "create, review, place, modify, or cancel any order. Return a concise "
            "account-mapping summary. Mask account numbers except the last four digits."
        ),
    )
    usage = response.usage
    input_tokens = int(usage.input_tokens if usage else 0)
    output_tokens = int(usage.output_tokens if usage else 0)
    cost = estimate_model_cost(settings.openai_model, input_tokens, output_tokens)
    if cost > settings.max_estimated_cost_per_run_usd:
        raise RuntimeError("LLM run exceeded configured per-run cost limit")
    return SnapshotResult(
        text=response.output_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
    )
