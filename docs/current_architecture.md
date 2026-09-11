# Current autonomous investment architecture

This document describes the code deployed in this repository. It distinguishes
observation, prediction, LLM research, portfolio construction, hard risk, and
execution so that no single model response can directly become an order.

## Daily lifecycle

The macOS LaunchAgent starts `ai-investor agent-cycle` every 900 seconds. The Mac
must be awake and online; the scheduler does not make a sleeping laptop monitor
the market. Each live trading date also requires an account-bound local arm.

The top-level sequence is:

1. Market-hours and market-data gate.
2. Sixty-symbol observation cycle with zero LLM calls.
3. Local deterministic trigger decision.
4. Account-state read and previous-order reconciliation.
5. Daily-bar forecast and quantitative candidate selection.
6. Public research collection and one structured LLM call.
7. Portfolio optimization with cash.
8. Fresh-quote, price-drift, account, and hard-risk validation.
9. Robinhood order review and, only when all LIVE gates hold, placement.
10. Order ledger update, concise decision log, and email summary.

## Fifteen-minute observation cycle

The universe contains 60 unique symbols:

- 12 fixed ETFs: SPY, QQQ, IWM, RSP and eight sector ETFs.
- 33 stable core slots.
- 10 event slots.
- Up to 5 current-position reserve slots. Unused reserve slots are filled from
  the core list.

The core scan requests US stocks with price at least $10, market cap at least
$5 billion, 30-day average volume at least 1 million, and average dollar volume
at least $100 million. Ranking is 50% liquidity percentile, 30% market-cap
percentile, and 20% data completeness. Recent return is explicitly excluded, so
universe construction is not an alpha signal.

The event scan requests stocks with market cap at least $1 billion, average
volume at least 500,000, relative volume at least 1.25, and absolute daily move
at least 2%. Event ranking uses absolute move, relative volume, and log dollar
volume. These names are observations, not automatic buys.

Robinhood quotes are fetched in exactly three batches of 20. For every symbol the
monitor stores the last regular-hours trade and time, bid, ask, adjusted previous
close, official prior close, state, and `has_traded`. A missing symbol, malformed
timestamp, or any quote older than 30 minutes rejects the entire observation
cycle. Normal cycles never call the LLM.

Deterministic price triggers are 2% from adjusted previous close for general
names and 1% for current positions. A decision run is eligible in the 15-minute
windows beginning 09:45, 13:00, and 15:30 ET, or when a price trigger occurs.
SQLite limits this to three decision runs per day and suppresses duplicate
decision keys.

## Quantitative prediction

The prediction layer reads 240 calendar days of Robinhood daily bars but excludes
the current incomplete trading day. A symbol needs at least 100 aligned daily
bars; SPY is the benchmark. Short-history names are excluded without blocking
the other symbols.

The pooled ridge model uses five transparent features:

1. 20-day return relative to SPY.
2. 60-day return relative to SPY.
3. Negative 5-day residual return, representing short-term reversal.
4. 1-day residual return, representing a price event.
5. Current volume relative to its trailing 20-day mean.

Separate models predict 5- and 20-day excess return over SPY. Samples are split
by date for validation, not randomly. Ridge regularization reduces unstable
coefficients. Forecasts are shrunk 65% by multiplying raw values by 0.35, then
capped at +/-4% for 5 days and +/-8% for 20 days. Validation error is
used to calculate an approximate probability of positive excess return.

A new candidate must have expected 20-day excess return of at least 0.5% and at
least 55% probability of outperforming SPY. At most three new names proceed;
current holdings are also evaluated so the system can decide whether to retain
or exit them.

## LLM research gate

The program, not the LLM, retrieves up to four research payloads:

- fundamentals for all candidates;
- financial statements for all candidates;
- earnings results for the top quantitative candidate;
- news for the top quantitative candidate.

The prompt contains the market regime, numeric forecasts, signals, uncertainty,
and those research payloads. External text is explicitly treated as untrusted
data. GPT-5.6 Luna runs with low reasoning effort and Structured Outputs. It must
return `allow`, `veto`, or `insufficient_evidence`, plus a concise rationale,
bull case, bear case, and falsification condition for each name.

The OpenAI client uses a 90-second request timeout and at most one SDK retry.
Even if a late retry eventually succeeds, the separate 180-second decision-age
gate can still reject the resulting order.

The LLM cannot set a forecast, weight, quantity, order type, risk limit, or place
an order. A name that is missing from the response, vetoed, or marked
insufficient does not enter the optimizer. The system allows one LLM call per
decision, 12,000 input tokens, 1,200 output tokens, $0.05 per call, and $2.50 per
month. Ordinary monitor cycles use no LLM tokens.

## Portfolio construction

Allowed names enter a long-only projected mean-variance optimizer. Expected
20-day excess return is converted to a daily estimate. The covariance matrix
uses recent aligned daily returns with 50% off-diagonal shrinkage. The objective
rewards expected return and penalizes covariance risk and forecast uncertainty.

Every optimization step projects weights to nonnegative values, the configured
single-name cap, and at most 100% total investment. Cash is the residual and can
remain 100%. No-trade is a valid result. Target weights are compared with current
market values; changes below $10 are ignored, and sells are planned before buys.

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
whole decision is older than 180 seconds or the fresh price moved more than 1%
from the monitor snapshot. These checks are in code, not the prompt.

Robinhood's nonplacing review runs next. A deterministic UUID derived from the
decision, symbol, side, and notional is sent only with placement. Retrying uses
the same UUID. SQLite records planned, reviewed, submitted, filled, rejected, and
reconciled states; later cycles compare broker-reported fills with positions.

Actual placement additionally requires all of:

- `mode = LIVE`;
- `live_trading = true`;
- a nonexpired daily arm bound to the selected Agentic account hash;
- all preceding risk and Robinhood review checks.

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

## Latency expectations and remaining limitations

The first real integration test for four research tools plus the Luna call took
about 17 seconds end to end. This is an observation, not a guaranteed service
level. Network, Robinhood, OpenAI, or SMTP latency can vary. The 180-second age,
60-second execution-quote freshness, and 1% price-drift checks fail closed when
the situation has materially changed. Email happens after execution.

The current forecast is a daily-bar cross-sectional model, not a proven intraday
trading edge. Intraday prices and volume select when to investigate, while the
numeric alpha model mainly uses completed daily information. The system has not
yet accumulated forward live performance, and it remains exposed to selection
bias, regime change, corporate actions, model error, market-order slippage,
halts, API changes, and gap risk. The $100,000 objective does not make any trade
more likely and cannot be guaranteed.
