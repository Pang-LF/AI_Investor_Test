# AI Investor Test

Current phase: **DRY_RUN read-only monitoring**. There is no order execution code in this repository.

## What is implemented

- Robinhood OAuth with PKCE, dynamic client registration, refresh tokens, and macOS Keychain storage.
- A Responses API read-only MCP call restricted to four `get_*` tools.
- A direct Robinhood MCP monitor that makes zero LLM calls.
- A 60-symbol universe: 12 fixed market/sector ETFs, 33 core slots,
  10 event slots, and up to 5 position-reserve slots. Empty reserve slots are
  filled from the core ranking so every quote cycle contains exactly 60 symbols.
- Three sequential quote batches of 20 symbols every 15 minutes during regular
  U.S. market hours.
- Read-only full-market scan previews for the core and event buckets. Preview
  calls do not create or modify Robinhood scans or watchlists.
- A morning/afternoon universe cache, deterministic price-move triggers, local
  JSONL logs, retry/backoff, and a hard eight-tool-call cycle budget.
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
.venv/bin/ai-investor monitor-cycle
```

`monitor-cycle` exits before connecting to Robinhood outside 9:30 AM-4:00 PM
New York time on weekdays. For a manual read-only connectivity test outside
those hours:

```bash
.venv/bin/ai-investor monitor-cycle --force
```

## macOS 15-minute scheduler

Install the local LaunchAgent after the manual monitor test passes:

```bash
chmod +x scripts/install_launchd.sh scripts/uninstall_launchd.sh
scripts/install_launchd.sh
```

The Mac must be awake for a scheduled cycle to run. Remove the scheduler with:

```bash
scripts/uninstall_launchd.sh
```

Runtime state and market logs stay under `.local/` and `logs/`. Both directories
are ignored by Git because credentials, account mappings, and high-frequency
runtime data must not be published from this public repository.

Do not put keys, OAuth tokens, account snapshots, or trading logs in this public repository.

## Safety status

- `mode = "DRY_RUN"`
- `live_trading = false`
- No options, crypto, margin, short-selling, or order-placement implementation exists.
- The direct monitor allowlist contains only `get_accounts`,
  `get_equity_positions`, `get_equity_quotes`, and non-persistent
  `preview_scan`.
- The monitor never sends account numbers to logs and never calls OpenAI.
