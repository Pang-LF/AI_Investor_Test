# Current autonomous investment architecture

This document describes the code deployed in this repository. It distinguishes
observation, prediction, LLM research, portfolio construction, hard risk, and
execution so that no single model response can directly become an order.

## Daily lifecycle

The macOS LaunchAgent starts `ai-investor agent-cycle` every 900 seconds. The Mac
must be awake and online; the scheduler does not make a sleeping laptop monitor
the market. LIVE requires one persistent local authorization rather than a new
authorization every trading date.

The top-level sequence is:

1. Market-hours and market-data gate.
2. Sixty-symbol observation cycle with zero LLM calls.
3. Local deterministic trigger decision.
4. Account-state read and previous-order reconciliation.
5. Daily-bar forecast, OOS calibration, and research-candidate selection.
6. Public research collection and one structured LLM call.
7. Portfolio optimization with cash.
8. Fresh-quote, price-drift, account, and hard-risk validation.
9. Robinhood order review and, only when all LIVE gates hold, placement.
10. Order ledger update, concise decision log, and email summary.

## Fifteen-minute observation cycle

The universe contains 60 unique symbols:

- 12 context-only ETFs: SPY, QQQ, IWM, RSP and eight sector ETFs. They inform
  the market snapshot and LLM context but do not become new buy candidates.
- 15 target large-cap slots above $10 billion.
- 12 target mid-cap slots between $2 billion and $10 billion.
- 8 target smaller-company slots between $500 million and $2 billion.
- 8 event slots above $500 million.
- Five current-position reserve slots. The independent hard limit remains ten;
  if more than five positions already exist, every holding is still observed and
  excess holdings displace large-cap slots. Unused reserve slots are filled
  round-robin across the size/event buckets.

Large and mid pools have minimum average dollar volume of $100 million and
$30 million. Small caps use a 30-day average-volume prefilter, then must pass a
20-completed-day median dollar-volume floor of $20 million and a free-float
ratio of at least 10%. Stable size pools require an IPO age of at least 252
calendar days (approximately 180 trading days), exclude identified OTC venues,
blank-check/SPAC issuers, and explicit non-regular trading status. Ranking
within each size bucket is 75% liquidity and 25% data completeness; larger
market cap does not improve rank inside a bucket.

For the top 20 small-cap prefilter results, the monitor fetches both raw and
split-adjusted daily histories. A step-up in the raw/adjusted price ratio flags a
recent reverse split; missing comparison history or a detected reverse split
excludes the name from the stable small-cap pool. This check runs only on a
daily full universe refresh and is cached, not every 15 minutes.

The event scan requests stocks with market cap at least $500 million, average
volume at least 500,000, relative volume at least 1.25, and absolute daily move
at least 2%. Event ranking uses absolute move, relative volume, log dollar
volume, and data completeness. Breakouts and breakdowns are interleaved, and no
more than two event names from one sector are selected. Recent IPOs may enter
this event pool but not a stable size pool. These names are observations, not
automatic buys.

Every selection and fill path shares a maximum of five stocks per sector and
deduplicates multiple share classes of the same issuer. Context ETFs do not
consume these stock-sector limits. If the constrained pools cannot provide 60
unique names, the cycle fails instead of silently relaxing a control.

Robinhood quotes are fetched in exactly three batches of 20. For every symbol the
monitor stores the last regular-hours trade and time, bid, ask, adjusted previous
close, official prior close, state, and `has_traded`. A missing symbol, malformed
timestamp, or any quote older than 30 minutes rejects the entire observation
cycle. From the same snapshots it derives advancing/declining breadth, median
daily move, cross-sectional dispersion, median/maximum spread, fixed-ETF moves,
and the ten largest moves since the prior 15-minute snapshot. These features add
market context without another API call. Normal cycles never call the LLM.

Deterministic price triggers are 2% from adjusted previous close for general
names and 1% for current positions. Three scheduled decisions are due at 09:45,
13:00, and 15:30 ET. If the computer, data source, or cooldown is unavailable at
the exact time, the oldest missed slot is caught up on the next eligible regular-
market cycle. Scheduled completion is tracked across strategy versions so a
version change cannot duplicate a slot. Price triggers have a separate quota of
up to three decisions per trading day. Every scheduled or triggered formal
decision shares a global 60-minute minimum interval. SQLite stores distinct
time-specific keys and suppresses duplicate decisions.

## Quantitative prediction

The prediction layer requests 1,095 calendar days of Robinhood daily bars but excludes
the current incomplete trading day. A symbol needs at least 100 aligned daily
bars; SPY is the benchmark. Short-history names are excluded without blocking
the other symbols.

The pooled ridge model uses five transparent features. Each price-return feature
first removes market exposure using beta estimated from the preceding 60 daily
returns:

1. 20-day return relative to SPY.
2. 60-day return relative to SPY.
3. Negative 5-day residual return, representing short-term reversal.
4. 1-day residual return, representing a price event.
5. Current volume relative to its trailing 20-day mean.

The estimated 60-day beta is also logged and supplied to the LLM as context,
although it is not an additional ridge coefficient.

Separate models predict 5- and 20-day beta-adjusted return. Samples are split by
date, not randomly, with a horizon-length purge between training and validation
so overlapping forward labels do not cross the boundary. Ridge regularization
reduces unstable coefficients. Forecasts are shrunk 65% by multiplying raw
values by 0.35, then capped at +/-4% for 5 days and +/-8% for 20 days.
Validation residuals are defined as `realized - shrunk prediction`. Consecutive
20-day labels overlap, so calibration retains only one cross-section per
horizon-length date block. The median non-overlapping OOS residual corrects
forecast bias. Calibrated positive-return probability comes from that empirical
OOS residual distribution applied to the shrunk forecast before bias adjustment,
which is equivalent to applying centered residuals to the bias-adjusted forecast
and avoids counting the bias twice. A separate normal-error raw probability is retained
only for inexpensive research qualification.

The decision funnel has three levels. Level 1 sends at most five new names to
research when raw shrunk 20-day alpha is at least 0.5% and raw probability is at
least 48%; Dynamic/Event names can enter research without passing that gate.
This is permission to spend research cost, never permission to buy. Current
holdings are added first so they receive explicit retain/exit review.

Entry and holding rules are asymmetric. A new position must pass the full Level
2 entry gate. An existing position can remain eligible down to -0.5% adjusted
20-day alpha, 45% calibrated probability, and -0.05 edge ratio. A non-critical
quantitative or `insufficient_evidence` exit signal must occur in two distinct
decision runs before it can force liquidation; the first signal places a floor
at the current weight. An LLM `veto` or critical identity/tradability/corporate-
action conflict can exit immediately. This state is durable and idempotent in
SQLite.

Level 2 requires at least 0.5% bias-adjusted 20-day alpha, 55% calibrated
probability, a 0.15 expected-alpha/uncertainty ratio, at least five
non-overlapping calibration blocks, and no LLM veto. These provisional thresholds
were enabled on 2026-09-11 after a 22-block expanding walk-forward diagnostic.
That diagnostic uses a static current universe and is therefore affected by
survivorship and selection bias; the live ledger must replace it with prospective
evidence. The thresholds are protective floors, not a proven optimal policy.
Each email labels a seven-block calibration as LOW confidence and reports a
date-clustered 95% interval rather than presenting the empirical point estimate
as precise certainty.

Level 3, once enabled, makes eligible names compete with current holdings, other
candidates, and cash in the optimizer. No research or LLM story alone can create
a quantitative edge.

## LLM research gate

The program, not the LLM, makes up to thirteen bounded research calls:

- account-specific tradability for the research set;
- fundamentals for up to ten candidates/current holdings;
- financial statements for those same names;
- earnings results for the top five research names;
- news for the top five research names.

The prompt contains the market regime, numeric forecasts, signals, uncertainty,
and those research payloads. External text is explicitly treated as untrusted
data. GPT-5.6 Terra runs with low reasoning effort and Structured Outputs. It must
return `allow`, `veto`, or `insufficient_evidence`, plus a concise rationale,
bull case, bear case, and falsification condition for each name. `allow` means
only "no material thesis-breaking information found." The prompt explicitly
calls out halts, delisting warnings, recent reverse splits, unresolved
restructurings, and critical corporate-action data. Stale optional fields such
as an old dividend record are quarantined as non-critical and cannot alone veto
an otherwise supported ticker. The structured result records data severity and
the affected fields.

The OpenAI client uses a 480-second request timeout and no SDK retry. The
separate 600-second decision-age gate rejects a result that arrives too late.

The LLM cannot set a forecast, weight, quantity, order type, risk limit, or place
an order. A name that is missing from the response, vetoed, or marked
insufficient cannot enter the calibrated execution layer. The system allows one LLM call per
decision, 40,000 input tokens, 2,400 output tokens, $0.15 per call, and $10 per
month. Ordinary monitor cycles use no LLM tokens.

## Portfolio construction

Allowed names enter a long-only projected mean-variance optimizer expressed on
one consistent 20-day horizon. The covariance matrix scales recent aligned daily
returns to 20 days and uses 50% off-diagonal shrinkage. The objective rewards
expected return and penalizes covariance risk, forecast uncertainty, and 0.40%
estimated round-trip reallocation cost. Optimization starts from current weights;
if the new portfolio's net objective does not exceed the current portfolio's,
the current allocation is retained.

The decision log and email expose the actual sizing terms: shrunk and bias-
adjusted forecast, forecast uncertainty, 20-day variance, current/minimum/final
weights, expected-return contribution, covariance penalty, estimation penalty,
reallocation cost, and old/new objective values. There is no hidden Kelly or
regime multiplier in v0.5.3. Structural regime and deterministic intraday tone
are reported separately and are research context only.

Every optimization step projects weights to nonnegative values, a 35% strategy
soft cap per name, a 50% strategy soft cap per known sector, and at most 100%
total investment. The independent hard single-name limit remains 100%; the
strategy therefore normally diversifies without changing the user-selected
absolute boundary. Cash is the residual and can remain 100%.
No-trade is valid. Target weights are compared with current market values;
changes below $10 are ignored, and sells are planned before buys.

## Hard risk and execution

The independent risk engine rechecks every proposed order. Current user-selected
concentration limits are 100% of NAV per name/order, 100% daily turnover, and 10
positions. The non-overridable product boundary remains cash-funded long-only
stocks and unleveraged ETFs: no borrowing, options, crypto, futures, short sales,
or leveraged/inverse ETFs. Buys are sized against `cash`, never buying power.

New buys are blocked after an 8% same-day loss or 20% drawdown from the recorded
high-water mark. Every order also requires account-specific tradability, a valid
quote no older than 60 seconds, bid/ask spread no wider than 0.30%, membership in
the approved buy universe, and sufficient cash.

To prevent latency from turning a valid analysis into a stale trade, execution
refetches the candidate quote after the LLM finishes. An order is rejected if the
whole decision is older than 600 seconds or the fresh price moved more than 1%
from the monitor snapshot. These checks are in code, not the prompt.

Robinhood's nonplacing review runs next. A deterministic UUID derived from the
decision, symbol, side, and notional is sent only with placement. Retrying uses
the same UUID. SQLite records planned, reviewed, submitted, filled, rejected, and
reconciled states; later cycles compare broker-reported fills with positions.

Actual placement additionally requires all of:

- `mode = LIVE`;
- `live_trading = true`;
- a persistent authorization bound to the selected Agentic account hash, the
  hard-risk policy version, and a hash of every hard-risk parameter;
- all preceding risk and Robinhood review checks.

The persistent authorization does not expire at midnight and a strategy-version
change does not invalidate it, allowing the agent to evolve investment rules.
It fails closed if the authorization file is missing/invalid, the selected
Agentic account changes, any hard-risk value changes, the hard-risk version
changes, or either LIVE config switch is disabled. Legacy daily-arm files cannot
authorize the new schema. Authorization is checked on every successful in-hours
monitor connection and again immediately before each order placement.

## Email and logging

After order outcomes are known, one plain-text email is produced from structured
data without another LLM call. It includes market/regime data, triggers,
quantitative forecasts, every LLM verdict and concise rationale, selected target
weights, reasons for nonselection, order/rejection states, tokens, cost, and
timings. It excludes the full account number, credentials, and chain-of-thought.

Email is deliberately after order processing, so a slow SMTP server cannot make
the execution quote stale. Delivery retries once; failures are stored in ignored
`.local/notification_outbox` and retried on later cycles. When
`notification_required = true`, missing SMTP configuration blocks the LLM and
therefore blocks trading.

Operational failures are separate from decision summaries. Unhandled scheduler,
configuration, Robinhood OAuth/MCP, authorization, and runtime failures are
logged under `logs/health` and trigger an email on first detection. A stale or
incomplete market snapshot must occur in two consecutive cycles before alerting
to avoid noise from one transient quote issue. An unresolved failure may remind
again after six hours; successful checks reset its counter silently. Recovery
and daily heartbeat emails are intentionally disabled.

The failure notifier runs on this Mac. It cannot send while the Mac is off,
fully offline, the LaunchAgent itself is not executing, or SMTP is unavailable.
An external dead-man watchdog would be required to actively detect those cases.

## Latency expectations and remaining limitations

The first real integration test for four research tools plus a Luna call took
about 17 seconds end to end before the Terra upgrade. This is not a
Terra latency guarantee. Network, Robinhood, OpenAI, or SMTP latency can vary.
The 600-second age,
60-second execution-quote freshness, and 1% price-drift checks fail closed when
the situation has materially changed. Email happens after execution.

The current forecast is a daily-bar cross-sectional model, not a proven intraday
trading edge. Intraday prices select when to investigate, while the
numeric alpha model mainly uses completed daily information. The system has not
yet accumulated forward live performance. Point-in-time fundamentals, SEC filing
features, analyst revisions, 15-minute OHLCV/VWAP, sector-neutral residuals,
transaction-cost-aware optimization, and ETF look-through exposure are not yet
implemented. Robinhood's preview scanner returns at most 200 rows and exposes no
pagination, so the current four scans evaluate broad filters server-side but do
not yet constitute a provably complete all-US-stock discovery feed. Recent
reverse splits, delisting risk, and complex restructurings are LLM risk flags
rather than guaranteed deterministic exclusions because the MCP does not expose
complete corporate-action history. It remains exposed to selection bias, regime change, corporate
actions, model error, market-order slippage, halts, API changes, and gap risk.
The $100,000 objective does not make any trade more likely and cannot be guaranteed.
