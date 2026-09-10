# AI Investor Test

Current phase: **DRY_RUN foundation only**. There is no order execution code in this repository.

## What is implemented

- Robinhood OAuth with PKCE, dynamic client registration, refresh tokens, and macOS Keychain storage.
- A Responses API read-only MCP call restricted to four `get_*` tools.
- A hard configuration lock that rejects `LIVE_TRADING=true`.
- Token and estimated GPT-5.6 Luna cost reporting for the read-only snapshot.

## Local setup

Use Python 3.9 or newer:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/ai-investor check-config
.venv/bin/ai-investor oauth-login
.venv/bin/ai-investor set-openai-key
.venv/bin/ai-investor account-snapshot
```

Do not put keys, OAuth tokens, account snapshots, or trading logs in this public repository.

## Safety status

- `mode = "DRY_RUN"`
- `live_trading = false`
- No options, crypto, margin, short-selling, or order-placement implementation exists.
- The next step is account mapping and a read-only connectivity test.
