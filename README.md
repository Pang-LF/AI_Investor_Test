# AI Investor Test

One autonomous Robinhood Agentic-account system for a $1,000 initial account,
with planned $1,000 contributions in months 1 and 2. The aspirational objective
is $100,000 within five years; it is not a promise and is not fed to the model as
a reason to force trades.

Purely as target math, $1,000 now plus $1,000 in months 1 and 2 would require
about 6.12% compounded every month (about 104% annualized) to reach $100,000 at
month 60. That hurdle is extraordinarily high and implies a substantial chance
of permanent capital loss.

Current deployment mode: **LIVE-capable and config-enabled**. It reads real
market/account data, trains the forecast, runs bounded LLM research, optimizes
the target portfolio, applies hard risk, and calls Robinhood's order-review tool.
It still cannot place an order unless the local daily account-bound arm is valid.

## Decision architecture

Every 15 minutes during regular US market hours:

1. A no-LLM monitor scans a 60-symbol universe and validates every quote.
2. Most cycles end immediately. A decision run occurs only at 09:45, 13:00,
   15:30 ET, or on a deterministic move trigger, with at most three per day.
3. A pooled, regularized model estimates 5- and 20-day SPY-relative returns from
   prior completed daily bars. Current-day prices never enter model training.
4. Up to three quantitative candidates plus current holdings receive one
   structured GPT-5.6 Luna research call using fundamentals, financials,
   earnings, and news. The LLM can allow or veto; it cannot size an order.
5. A long-only covariance-shrinkage optimizer creates target weights including
   cash. No-trade and 100% cash are valid outputs.
6. The independent hard-risk engine checks the target and fresh execution quote.
7. A deterministic UUID suppresses duplicate submissions. Robinhood review runs
   before placement; later cycles reconcile broker order states into SQLite.

The versioned strategy is
[`strategies/agentic/strategy_v0.2.0.yaml`](strategies/agentic/strategy_v0.2.0.yaml).
The hard-risk policy is separate in [`config/settings.toml`](config/settings.toml).

## Cost controls

- One LLM call per decision; no LLM call on ordinary 15-minute monitor cycles.
- Maximum 12,000 input and 1,200 output tokens.
- Maximum $0.05 per call and $2.50 total LLM spend per month.
- Four public-research calls and 50 total Robinhood MCP calls per decision run.
- The first real integration research call cost about $0.0032.

## Commands

```bash
.venv/bin/pip install -e .
.venv/bin/ai-investor check-config
.venv/bin/ai-investor oauth-status
.venv/bin/ai-investor account-snapshot
.venv/bin/ai-investor monitor-cycle
.venv/bin/ai-investor agent-cycle
.venv/bin/python -m unittest discover -s tests -v
```

Install or refresh the 15-minute macOS LaunchAgent:

```bash
chmod +x scripts/install_launchd.sh scripts/uninstall_launchd.sh
scripts/install_launchd.sh
```

The Mac must stay awake and online. See
[`docs/live_runbook.md`](docs/live_runbook.md) before changing to LIVE.

## Data and secret boundary

Code, versioned strategies, configuration, and documentation belong in GitHub.
OAuth tokens and the OpenAI key stay in macOS Keychain. Account mappings,
high-frequency quotes, decision logs, historical caches, arms, and the SQLite
ledger stay under ignored `.local/` and `logs/`; publishing them in this public
repository would expose financial activity. No full chain-of-thought is stored—
only concise structured rationale.

Robinhood says Agentic Trading can place trades without per-order confirmation
when instructed, but the account owner remains responsible for those trades and
the product can lose the full investment. This project adds controls; it cannot
eliminate market, model, API, execution, or operational risk.
