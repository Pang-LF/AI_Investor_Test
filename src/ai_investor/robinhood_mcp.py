from __future__ import annotations

import json
import time
from typing import Any, Dict, FrozenSet, Iterable, Optional

import httpx

from .robinhood_oauth import MCP_URL, RobinhoodOAuth


MCP_PROTOCOL_VERSION = "2025-03-26"

# The recurring monitor can execute saved scans but cannot create or alter them.
MONITOR_READ_ONLY_TOOLS: FrozenSet[str] = frozenset(
    {
        "get_accounts",
        "get_equity_positions",
        "get_equity_quotes",
        "get_equity_historicals",
        "get_scans",
        "run_scan",
    }
)

DECISION_READ_ONLY_TOOLS: FrozenSet[str] = frozenset(
    {
        "get_accounts",
        "get_portfolio",
        "get_equity_positions",
        "get_equity_orders",
        "get_equity_quotes",
        "get_equity_historicals",
        "get_equity_tradability",
        "get_equity_fundamentals",
        "get_financials",
        "get_earnings_results",
        "get_equity_news",
    }
)

ORDER_REVIEW_TOOLS: FrozenSet[str] = frozenset({"review_equity_order"})
ORDER_WRITE_TOOLS: FrozenSet[str] = frozenset(
    {"place_equity_order", "cancel_equity_order"}
)
SCANNER_CONFIGURATION_TOOLS: FrozenSet[str] = frozenset({"create_scan"})

KNOWN_MUTATING_TOOLS: FrozenSet[str] = frozenset(
    {
        "add_option_to_watchlist",
        "add_to_watchlist",
        "cancel_crypto_order",
        "cancel_equity_order",
        "cancel_option_exercise",
        "cancel_option_order",
        "create_alert",
        "create_scan",
        "create_watchlist",
        "delete_alert",
        "exercise_option",
        "follow_watchlist",
        "mark_alerts_read",
        "place_crypto_order",
        "place_equity_order",
        "place_option_order",
        "preview_crypto_order",
        "remove_from_watchlist",
        "remove_option_from_watchlist",
        "review_equity_order",
        "review_option_order",
        "unfollow_watchlist",
        "update_alert",
        "update_scan_config",
        "update_scan_filters",
        "update_watchlist",
    }
)


class MCPError(RuntimeError):
    pass


def _parse_mcp_response(response: httpx.Response) -> Dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        return dict(response.json())

    payloads = []
    current_data = []
    for line in response.text.splitlines():
        if line.startswith("data:"):
            current_data.append(line[5:].lstrip())
        elif not line and current_data:
            payloads.append(json.loads("\n".join(current_data)))
            current_data = []
    if current_data:
        payloads.append(json.loads("\n".join(current_data)))
    if not payloads:
        raise MCPError("Robinhood returned an empty MCP event stream")
    return dict(payloads[-1])


def _structured_tool_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if result.get("isError"):
        messages = [
            item.get("text", "")
            for item in result.get("content", [])
            if item.get("type") == "text"
        ]
        raise MCPError("Robinhood tool error: " + " ".join(messages))

    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return dict(structured)
    for item in result.get("content", []):
        if item.get("type") != "text":
            continue
        try:
            parsed = json.loads(item.get("text", ""))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return dict(parsed)
    raise MCPError("Robinhood tool response had no structured JSON content")


class RobinhoodMCPClient:
    """Small Streamable HTTP MCP client with an explicit per-use allowlist."""

    def __init__(
        self,
        max_calls: int,
        allowed_tools: Iterable[str],
        allow_order_submission: bool = False,
        allow_scanner_configuration: bool = False,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.max_calls = max_calls
        self.allowed_tools = frozenset(allowed_tools)
        self.allow_order_submission = allow_order_submission
        self.allow_scanner_configuration = allow_scanner_configuration
        self.max_retries = max_retries
        self.call_count = 0
        self._request_id = 0
        self._session_id: Optional[str] = None
        self._http = httpx.Client(timeout=timeout_seconds)
        self._base_headers = {
            "Authorization": "Bearer " + RobinhoodOAuth().access_token(),
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }

    def __enter__(self) -> "RobinhoodMCPClient":
        self._initialize()
        return self

    def __exit__(self, *_args: object) -> None:
        self._http.close()

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _headers(self) -> Dict[str, str]:
        headers = dict(self._base_headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
            headers["MCP-Protocol-Version"] = MCP_PROTOCOL_VERSION
        return headers

    def _post(self, payload: Dict[str, Any]) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            try:
                response = self._http.post(
                    MCP_URL, headers=self._headers(), json=payload
                )
            except httpx.RequestError:
                if attempt == self.max_retries:
                    raise
                time.sleep(min(float(2**attempt), 8.0))
                continue
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return response
            if attempt == self.max_retries:
                response.raise_for_status()
            retry_after = response.headers.get("retry-after")
            delay = float(retry_after) if retry_after else float(2**attempt)
            time.sleep(min(delay, 8.0))
        raise MCPError("unreachable retry state")

    def _initialize(self) -> None:
        response = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "ai-investor-test",
                        "version": "0.6.0",
                    },
                },
            }
        )
        payload = _parse_mcp_response(response)
        if "error" in payload:
            raise MCPError(f"Robinhood MCP initialize failed: {payload['error']}")
        session_id = response.headers.get("mcp-session-id")
        if not session_id:
            raise MCPError("Robinhood MCP did not return a session id")
        self._session_id = session_id
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self.allowed_tools:
            raise MCPError(f"Rejected non-allowlisted MCP tool: {name}")
        if name in ORDER_WRITE_TOOLS and not self.allow_order_submission:
            raise MCPError(f"Order submission is not armed for MCP tool: {name}")
        if name in SCANNER_CONFIGURATION_TOOLS and not self.allow_scanner_configuration:
            raise MCPError(f"Scanner configuration is not armed for MCP tool: {name}")
        if name in KNOWN_MUTATING_TOOLS and name not in (
            ORDER_REVIEW_TOOLS | ORDER_WRITE_TOOLS | SCANNER_CONFIGURATION_TOOLS
        ):
            raise MCPError(f"Unsupported mutating MCP tool: {name}")
        if self.call_count >= self.max_calls:
            raise MCPError("Per-cycle MCP call budget exhausted")
        self.call_count += 1
        response = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        payload = _parse_mcp_response(response)
        if "error" in payload:
            raise MCPError(f"Robinhood MCP error: {payload['error']}")
        return _structured_tool_result(dict(payload.get("result", {})))


class RobinhoodReadOnlyMCPClient(RobinhoodMCPClient):
    """Compatibility wrapper used by the 60-symbol monitor."""

    def __init__(
        self,
        max_calls: int,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(
            max_calls=max_calls,
            allowed_tools=MONITOR_READ_ONLY_TOOLS,
            allow_order_submission=False,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
