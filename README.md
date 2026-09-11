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
It still cannot place an order unless the local persistent account- and
hard-risk-bound authorization is valid.

## Decision architecture

Every 15 minutes during regular US market hours:

1. A no-LLM monitor scans a 60-symbol universe and validates every quote.
2. Most cycles end immediately. A decision run occurs only at 09:45, 13:00,
   15:30 ET, or on a deterministic move trigger, with at most three per day.
3. A pooled, regularized model estimates 5- and 20-day rolling-beta-adjusted
   returns from prior completed daily bars using purged temporal validation.
4. Up to five new research candidates plus current holdings receive one structured
   GPT-5.6 Terra research call. All receive fundamentals and financials; the top
   five receive earnings and news. The LLM can allow or veto; it cannot size an order.
5. A long-only covariance-shrinkage optimizer uses 35% per-name and 50%
   per-sector strategy soft caps and creates target weights including cash.
   No-trade and 100% cash are valid outputs.
6. The independent hard-risk engine checks the target and fresh execution quote.
7. A deterministic UUID suppresses duplicate submissions. Robinhood review runs
   before placement; later cycles reconcile broker order states into SQLite.

The versioned strategy is
[`strategies/agentic/strategy_v0.5.1.yaml`](strategies/agentic/strategy_v0.5.1.yaml).
The hard-risk policy is separate in [`config/settings.toml`](config/settings.toml).
The full implemented data and decision flow is documented in
[`docs/current_architecture.md`](docs/current_architecture.md).

## Cost controls

- One LLM call per decision; no LLM call on ordinary 15-minute monitor cycles.
- Maximum 40,000 input and 2,400 output tokens.
- Maximum $0.15 per call and $8 total LLM spend per month.
- Thirteen public-research calls and 50 total Robinhood MCP calls per decision run.
- The earlier three-name Luna integration call cost about $0.0032; Terra/five-name
  runs will cost more and are measured individually in the ledger and email.

## Commands

```bash
.venv/bin/pip install -e .
.venv/bin/ai-investor check-config
.venv/bin/ai-investor oauth-status
.venv/bin/ai-investor account-snapshot
.venv/bin/ai-investor email-status
.venv/bin/ai-investor monitor-cycle
.venv/bin/ai-investor agent-cycle
.venv/bin/ai-investor live-status
.venv/bin/python -m unittest discover -s tests -v
```

Install or refresh the 15-minute macOS LaunchAgent:

```bash
chmod +x scripts/install_launchd.sh scripts/uninstall_launchd.sh
scripts/install_launchd.sh
```

The Mac must stay awake and online. See
[`docs/live_runbook.md`](docs/live_runbook.md) before changing to LIVE.

Persistent LIVE is authorized once with an exact risk acknowledgement. It does
not expire daily and permits strategy evolution, but any Agentic-account change
or change to any hard-risk parameter invalidates it. Fatal background failures
send a rate-limited email; two consecutive stale/incomplete market cycles also
trigger an alert. No recovery or daily heartbeat emails are sent.

Email summaries are mandatory before an LLM decision can trade. Configure SMTP
credentials interactively so the password goes only to macOS Keychain:

```bash
.venv/bin/ai-investor set-email --sender you@gmail.com --recipient you@gmail.com
.venv/bin/ai-investor test-email
```

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
