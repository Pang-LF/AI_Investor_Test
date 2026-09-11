# Live runbook

The repository is currently configured as LIVE-capable and LIVE-enabled, but a
daily version-bound arm is still mandatory. Do not bypass the sequence below.
Robinhood states that the account owner is responsible for agent orders.

## Every trading day

1. Keep the Mac awake, connected to power, and online.
2. Confirm `margin investing` is disabled in Robinhood. The program nevertheless
   sizes buys from `cash`, never `buying_power`.
3. Run the test suite:

   ```bash
   .venv/bin/python -m unittest discover -s tests -v
   ```

4. Run and inspect one market-hours shadow decision:

   ```bash
   .venv/bin/ai-investor agent-cycle
   tail -n 1 logs/decisions/$(TZ=America/New_York date +%F).jsonl
   ```

5. Only after the shadow output, account mapping, proposed orders, and risk
   reasons are correct, set both values in `config/settings.toml`:

   ```toml
   mode = "LIVE"
   live_trading = true
   ```

6. Arm only the current or next calendar date. The arm stores a one-way account
   hash—not the account number—and the active strategy/risk-policy versions:

   ```bash
   .venv/bin/ai-investor arm-live --date YYYY-MM-DD \
     --ack "I ACCEPT LIVE TRADING RISK"
   ```

The scheduler still exits without an order unless there is a scheduled decision
window or a deterministic market trigger, a quantitative candidate, an LLM
`allow`, a valid optimizer target, and a passing order review plus hard-risk
approval.

Any strategy or hard-risk version change invalidates an existing arm and requires
the explicit command again.

## Emergency stop

Run:

```bash
.venv/bin/ai-investor disarm-live
```

Then set `live_trading = false` and `mode = "SHADOW"`. Either the absent daily
arm or the false config kill switch independently prevents order placement.
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
