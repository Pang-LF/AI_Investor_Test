# AI Investor Test

One autonomous Robinhood Agentic-account system for a $1,000 initial account,
with planned $1,000 contributions in months 1 and 2. The aspirational objective
is $100,000 within five years; it is not a promise and is not fed to the model as
a reason to force trades.

Purely as target math, $1,000 now plus $1,000 in months 1 and 2 would require
about 6.12% compounded every month (about 104% annualized) to reach $100,000 at
month 60. That hurdle is extraordinarily high and implies a substantial chance
of permanent capital loss.

Current deployment mode: **LIVE monitoring with 20D entries shadow-only**. It reads real
market/account data, trains the forecast, runs bounded LLM research, optimizes
the target portfolio, applies hard risk, and calls Robinhood's order-review tool.
The current 20D engine cannot place buy orders. Existing-position reductions or
exits still require the local persistent account- and hard-risk-bound authorization.

## Decision architecture

Every 15 minutes during regular US market hours:

1. A no-LLM monitor runs four versioned Robinhood saved scanners, assembles a
   60-symbol universe, and validates every quote. Scanner IDs and definition
   fingerprints stay in the ignored local runtime directory.
2. Three scheduled decisions are due at 09:45, 13:00, and 15:30 ET. A missed
   slot is caught up on the next eligible cycle. Deterministic move triggers may
   start up to three additional decisions per day. Every formal decision,
   scheduled or triggered, must be at least 60 minutes after the previous one.
3. The first formal decision of each day fits and freezes a pooled, regularized
   5-/20-day model snapshot; later decisions that day reuse it. Calibration is
   date-blocked, probabilities shrink toward 50%, and unsupported bias is not applied.
4. Up to five new research candidates plus current holdings receive one structured
   GPT-5.6 Terra research call. All receive fundamentals and financials. Persistent
   events, live triggers, event-bucket names, and holdings determine which five get
   earnings/news and which two get bounded SEC filing research. The LLM can allow
   or veto; it cannot size an order.
   Critical data conflicts are separated from quarantined non-critical fields.
   A separate shadow-only Event Engine prioritizes abnormal movers and supplies
   empirical 1-/5-day analog distributions to the same research call. Its output
   cannot create a target weight or authorize an order.
   Separately, formal decisions build a 200-stock daily-bar research tier from
   the cached scanner candidates. The 60-name intraday monitor and its Robinhood
   quote load remain unchanged. The frozen 60-name model is applied to the broad
   tier for shadow discovery; the broad tier does not retrain the existing core
   model used for live holding management.
   LLM assessments are cached for 390 minutes and reused when the prompt/model,
   trading date, durable event state, and material price-event band are unchanged.
   Unused LLM capacity rotates through new names from a 25-stock shortlist.
5. A long-only covariance-shrinkage optimizer lets each eligible name and sector
   use up to 100% of NAV and creates target weights including cash.
   No-trade and 100% cash are valid outputs. Existing holdings use lower hold
   thresholds and two-run confirmation for non-critical exits.
6. The independent hard-risk engine checks the target and fresh execution quote.
7. A deterministic UUID suppresses duplicate submissions. Robinhood review runs
   before placement; later cycles reconcile broker order states into SQLite.

Reviewed, quantitatively eligible 20D candidates from the 200-stock research
universe may enter the LIVE optimizer and reach Robinhood placement after fresh
quotes, tradability, hard-risk checks, and Robinhood's nonplacing order review.

Every decision log contains a 60-slot candidate funnel showing why each symbol
did or did not progress through research, LLM review, eligibility, optimization,
and order generation.

The versioned strategy is
[`strategies/agentic/strategy_v0.10.0.yaml`](strategies/agentic/strategy_v0.10.0.yaml).
The hard-risk policy is separate in [`config/settings.toml`](config/settings.toml).
The full implemented data and decision flow is documented in
[`docs/current_architecture.md`](docs/current_architecture.md).
The reviewed QuantSkills methods and the reasons external factor code was not
copied directly into LIVE are recorded in
[`docs/quantskills_review.md`](docs/quantskills_review.md).

Strategy v0.10.0 also records a strictly shadow-only
`cross_sectional_factor_rank_v0.1` challenger. It winsorizes and percentile-ranks
seven point-in-time components, combines them inside four independent families,
then gives trend, reversal, volume, and risk one equal vote each. Ridge and
challenger ranks are persisted side by side with later 5/10/20-day outcomes.
The challenger has no optimizer or order-generation path.

## Cost controls

- One LLM call per decision; no LLM call on ordinary 15-minute monitor cycles.
- Maximum 40,000 input and 4,000 output tokens.
- Maximum $0.15 per call and $10 total LLM spend per month.
- At most 18 research calls, including bounded SEC requests, and 50 total
  Robinhood MCP calls per decision run.
- The earlier three-name Luna integration call cost about $0.0032; Terra/five-name
  runs will cost more and are measured individually in the ledger and email.

## Commands

```bash
.venv/bin/pip install -e .
.venv/bin/ai-investor check-config
.venv/bin/ai-investor oauth-status
.venv/bin/ai-investor account-snapshot
.venv/bin/ai-investor setup-scanners --ack "CREATE ROBINHOOD SAVED SCANNERS"
.venv/bin/ai-investor email-status
.venv/bin/ai-investor monitor-cycle
.venv/bin/ai-investor agent-cycle
.venv/bin/ai-investor live-status
.venv/bin/ai-investor event-shadow-status
.venv/bin/ai-investor research-cache-status
.venv/bin/python scripts/opportunity_capture_audit.py
.venv/bin/python -m unittest discover -s tests -v
```

The opportunity-capture audit is read-only: it fetches daily price history, does
not call the LLM, and cannot submit orders. Monitor cycles also retain deduplicated
point-in-time universe compositions under ignored local state so later audits do
not have to project today's 60-name selection backward onto history.

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

Robinhood saved scanners are provisioned only by the explicit `setup-scanners`
command. Recurring monitor cycles can run them but cannot create or modify
account-side scanners. If their definitions drift from local configuration, the
system fails closed and requests explicit reprovisioning.

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
