from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

import httpx
import keyring


MCP_URL = "https://agent.robinhood.com/mcp/trading"
AUTH_METADATA_URL = (
    "https://agent.robinhood.com/.well-known/"
    "oauth-authorization-server/mcp/trading"
)
KEYRING_SERVICE = "ai-investor-test"
CLIENT_KEY = "robinhood-oauth-client"
TOKEN_KEY = "robinhood-oauth-token"
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8765/callback"


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _load_secret(name: str) -> Optional[Dict[str, Any]]:
    value = keyring.get_password(KEYRING_SERVICE, name)
    return json.loads(value) if value else None


def _save_secret(name: str, value: Dict[str, Any]) -> None:
    keyring.set_password(KEYRING_SERVICE, name, json.dumps(value))


@dataclass(frozen=True)
class OAuthMetadata:
    authorization_endpoint: str
    registration_endpoint: str
    token_endpoint: str


class RobinhoodOAuth:
    """OAuth client for Robinhood's official Agentic Trading MCP."""

    def __init__(self, redirect_uri: str = DEFAULT_REDIRECT_URI) -> None:
        self.redirect_uri = redirect_uri

    def _metadata(self) -> OAuthMetadata:
        response = httpx.get(AUTH_METADATA_URL, timeout=20)
        response.raise_for_status()
        raw = response.json()
        return OAuthMetadata(
            authorization_endpoint=raw["authorization_endpoint"],
            registration_endpoint=raw["registration_endpoint"],
            token_endpoint=raw["token_endpoint"],
        )

    def _client(self, metadata: OAuthMetadata) -> Dict[str, Any]:
        existing = _load_secret(CLIENT_KEY)
        if existing and self.redirect_uri in existing.get("redirect_uris", []):
            return existing

        response = httpx.post(
            metadata.registration_endpoint,
            json={
                "client_name": "AI Investor Test local runner",
                "redirect_uris": [self.redirect_uri],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
            timeout=20,
        )
        response.raise_for_status()
        client = response.json()
        _save_secret(CLIENT_KEY, client)
        return client

    def login(self) -> None:
        metadata = self._metadata()
        client = self._client(metadata)
        verifier = _b64url(secrets.token_bytes(64))
        challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
        expected_state = secrets.token_urlsafe(32)

        params = {
            "response_type": "code",
            "client_id": client["client_id"],
            "redirect_uri": self.redirect_uri,
            "scope": "internal",
            "resource": MCP_URL,
            "state": expected_state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        url = metadata.authorization_endpoint + "?" + urllib.parse.urlencode(params)
        callback = self._receive_callback(url)
        if callback.get("state") != expected_state:
            raise RuntimeError("OAuth state mismatch; refusing to exchange the code")
        if "error" in callback:
            raise RuntimeError(f"Robinhood OAuth failed: {callback['error']}")

        token_response = httpx.post(
            metadata.token_endpoint,
            data={
                "grant_type": "authorization_code",
                "client_id": client["client_id"],
                "redirect_uri": self.redirect_uri,
                "code": callback["code"],
                "code_verifier": verifier,
            },
            timeout=20,
        )
        token_response.raise_for_status()
        self._store_token(token_response.json())

    def _receive_callback(self, authorization_url: str) -> Dict[str, str]:
        parsed_redirect = urllib.parse.urlparse(self.redirect_uri)
        result: Dict[str, str] = {}

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(handler_self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(handler_self.path)
                query = urllib.parse.parse_qs(parsed.query)
                result.update({key: values[0] for key, values in query.items()})
                body = b"Authorization received. You can close this window."
                handler_self.send_response(200)
                handler_self.send_header("Content-Type", "text/plain; charset=utf-8")
                handler_self.send_header("Content-Length", str(len(body)))
                handler_self.end_headers()
                handler_self.wfile.write(body)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        address = (
            parsed_redirect.hostname or "127.0.0.1",
            parsed_redirect.port or 8765,
        )
        server = HTTPServer(address, CallbackHandler)
        server.timeout = 300
        webbrowser.open(authorization_url)
        server.handle_request()
        server.server_close()
        if not result:
            raise TimeoutError("No OAuth callback received within five minutes")
        return result

    def _store_token(self, token: Dict[str, Any]) -> None:
        token = dict(token)
        token["expires_at"] = time.time() + float(token.get("expires_in", 3600))
        _save_secret(TOKEN_KEY, token)

    def access_token(self) -> str:
        token = _load_secret(TOKEN_KEY)
        if not token:
            raise RuntimeError("Robinhood OAuth is not configured; run oauth-login")
        if time.time() >= float(token.get("expires_at", 0)) - 60:
            token = self._refresh(token)
        return str(token["access_token"])

    def _refresh(self, token: Dict[str, Any]) -> Dict[str, Any]:
        metadata = self._metadata()
        client = self._client(metadata)
        refresh_token = token.get("refresh_token")
        if not refresh_token:
            raise RuntimeError("Robinhood did not provide a refresh token; log in again")
        response = httpx.post(
            metadata.token_endpoint,
            data={
                "grant_type": "refresh_token",
                "client_id": client["client_id"],
                "refresh_token": refresh_token,
                "scope": "internal",
                "resource": MCP_URL,
            },
            timeout=20,
        )
        response.raise_for_status()
        refreshed = response.json()
        if "refresh_token" not in refreshed:
            refreshed["refresh_token"] = refresh_token
        self._store_token(refreshed)
        return refreshed

    def status(self) -> str:
        token = _load_secret(TOKEN_KEY)
        if not token:
            return "not_authenticated"
        return "authenticated" if token.get("refresh_token") else "access_token_only"
