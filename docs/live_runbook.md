# Live runbook

The repository is configured as LIVE-capable and LIVE-enabled. Robinhood states
that the account owner remains responsible for agent orders.

## One-time persistent authorization

1. Keep the Mac awake, connected to power, logged in, and online during market
   hours. Confirm `margin investing` is disabled in Robinhood. The program still
   sizes buys from `cash`, never `buying_power`.
2. Run the test suite and inspect authorization status:

   ```bash
   .venv/bin/python -m unittest discover -s tests -v
   .venv/bin/ai-investor live-status
   ```

   If saved scanners have not yet been provisioned, create the four versioned
   Robinhood scanners once. This changes only the account's scanner list and
   grants no order permission:

   ```bash
   .venv/bin/ai-investor setup-scanners \
     --ack "CREATE ROBINHOOD SAVED SCANNERS"
   ```

3. Confirm both config gates in `config/settings.toml`:

   ```toml
   mode = "LIVE"
   live_trading = true
   ```

4. Create the persistent authorization with the exact acknowledgement:

   ```bash
   .venv/bin/ai-investor arm-live --persistent \
     --ack "I ACCEPT LIVE TRADING RISK"
   ```

5. Verify it without placing an order:

   ```bash
   .venv/bin/ai-investor live-status
   ```

No daily terminal command is required after this. Strategy v0.7.0 keeps monitoring,
forecasting, research, and hypothetical optimization active, but sets
`twenty_day_new_entry_live_enabled = false`. Every 20D buy stops after Robinhood
review and is recorded as `shadow_20d_buy_reviewed`; the placement tool is never
called. Existing holdings use the lower hold gate and remain LIVE-manageable:
non-critical exit signals require two distinct decision runs, while a material
LLM veto or critical data conflict can exit immediately. Increasing an existing
position is also a buy and therefore remains shadow-only.

The Event Engine is independently fixed to `live_entry_enabled = false`. It may
change which abnormal movers receive the bounded research call, but it does not
enter portfolio optimization. Its first signal per symbol/day is stored with
the contemporaneous price, SPY price, beta, forecast distribution, and later
realized 1-/5-day excess returns.

The 15-minute monitor schedules three formal decisions at 09:45, 13:00, and
15:30 ET and permits up to three additional event-triggered decisions. All six
possible decisions share a 60-minute minimum interval. A due scheduled slot
blocked by that interval is retried on a later cycle; the oldest missed slot has
priority over an event trigger. Execution still requires the Mac to be awake and
online during regular market hours, working market/broker/API access, remaining
monthly LLM budget, and every safety gate.

The authorization permits investment-strategy version changes but is invalidated
by an Agentic-account change, a hard-risk policy-version change, a change to any
hard-risk value, a missing/corrupt local authorization, or either LIVE config
switch being disabled. The old date-bound authorization format is rejected.

## Failure notification

- Missing/invalid authorization, OAuth/MCP failures, configuration failures, and
  unhandled background errors send an email when first detected.
- Stale or incomplete market data sends an email after two consecutive failed
  cycles, normally about 30 minutes.
- An unresolved failure may send another reminder after six hours. Successful
  checks reset alert counters silently.
- Alerts are also written to `logs/health/YYYY-MM-DD.jsonl`. No recovery email or
  daily heartbeat email is sent.
- A local process cannot send email while the Mac is off, fully offline, the
  LaunchAgent itself is not running, or SMTP is unavailable. Covering those cases
  requires a separate external watchdog.

## Emergency stop

Run:

```bash
.venv/bin/ai-investor disarm-live
```

Then set `live_trading = false` and `mode = "SHADOW"`. Either an absent persistent
authorization or the false config kill switch independently prevents order placement.
Open or already-routed orders are not automatically cancelled; inspect and
cancel them in Robinhood if necessary.

## Current hard-risk policy

- Cash-funded, long-only equities and unleveraged ETFs.
- No options, crypto, futures, short sales, leveraged/inverse ETFs, or borrowing.
- Up to 100% of NAV in one position or one order, 100% daily turnover, and 10
  positions. These hard concentration limits are deliberately aggressive; the
  current investment strategy applies separate 35% per-name and 50% per-sector
  soft caps.
- New buys stop after an 8% same-day loss or a 20% drawdown from the recorded
  high-water mark. Gap losses can exceed those values.
- Each order requires a current quote, no more than 0.30% bid/ask spread,
  account-specific tradability, cash availability, Robinhood order review, and
  a deterministic idempotency UUID.
- The OpenAI request may wait up to 480 seconds, but the full decision is rejected
  after 600 seconds or if the post-LLM execution price moved over 1%.
- SMTP must be configured when `notification_required = true`; otherwise the
  system monitors but does not call the LLM or trade.
