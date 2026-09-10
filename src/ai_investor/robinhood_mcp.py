from __future__ import annotations

import json
import time
from typing import Any, Dict, FrozenSet, Optional

import httpx

from .robinhood_oauth import MCP_URL, RobinhoodOAuth


MCP_PROTOCOL_VERSION = "2025-03-26"

# Phase 1 deliberately exposes only the four calls required by the monitor.
# preview_scan is explicitly non-persistent in Robinhood's tool contract.
MONITOR_READ_ONLY_TOOLS: FrozenSet[str] = frozenset(
    {
        "get_accounts",
        "get_equity_positions",
        "get_equity_quotes",
        "preview_scan",
    }
)

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


class RobinhoodReadOnlyMCPClient:
    """Small Streamable HTTP MCP client with a hard read-only tool allowlist."""

    def __init__(
        self,
        max_calls: int,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.max_calls = max_calls
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

    def __enter__(self) -> "RobinhoodReadOnlyMCPClient":
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
            response = self._http.post(MCP_URL, headers=self._headers(), json=payload)
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
                        "version": "0.2.0",
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
        if name not in MONITOR_READ_ONLY_TOOLS:
            raise MCPError(f"Phase 1 rejected non-allowlisted MCP tool: {name}")
        if name in KNOWN_MUTATING_TOOLS:
            raise MCPError(f"Safety invariant violated for MCP tool: {name}")
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
