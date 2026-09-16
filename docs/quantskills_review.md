# QuantSkills methodology review

Reviewed on 2026-09-16. QuantSkills is a community catalog, not evidence that a
factor or strategy is profitable on this project's US-equity universe. No
external strategy code was copied into the LIVE runtime.

## Incorporated now

- `skill-quant-research`: keep data timing explicit, separate signal discovery
  from portfolio construction, and preserve point-in-time decision artifacts.
- `skill-quant-portfolio-risk`: let eligible forecasts compete in a cost-aware
  optimizer, while keeping hard limits outside the optimizer.
- `skill-quant-execution-microstructure`: distinguish decision and execution
  quotes, refetch immediately before release, preserve deterministic order IDs,
  and reconcile broker results.
- `skill-quant-factor-directional-alpha`: use trend, breakout, reversal, and
  channel-position families as a roadmap for future independent alpha engines,
  not as prevalidated LIVE signals.

## Not imported into LIVE

- The 296 directional factors: their repository explicitly says to re-run
  validation when universe, dates, vendor, or execution assumptions change.
- A-share/PandaData-specific screeners and event models: their data contracts do
  not match Robinhood US-equity inputs.
- TCN, GNN, reinforcement-learning, and other high-flexibility models: this
  account does not yet have enough independent point-in-time outcomes to justify
  their complexity.
- Third-party portfolio or execution code: the current broker state, hard-risk,
  idempotency, and reconciliation contracts remain locally controlled.

## Next research use

The most useful next contribution is a small set of independently logged
short-horizon OHLCV engines—trend/breakout, reversal, and event continuation—
each emitting expected return, uncertainty, horizon, and data timestamp. They
should share the existing optimizer only after their signals have an explicit
point-in-time contract. This is a methodology roadmap, not a prerequisite for
the v0.9.0 LIVE permission requested by the user.

## Concrete methods applicable to the current system

The following ideas fit the data already available in the project and address
observed weaknesses in the pooled ridge model:

1. **Cross-sectional rank transforms.** Winsorize each feature by date and use
   percentile/rank z-scores. This limits the tail extrapolation that previously
   made the model's top-ranked names worse than ranks 6–15.
2. **Small, named OHLCV factor families.** Start with residual trend,
   risk-adjusted trend, channel/breakout position, five-day reversal, relative
   volume/volume z-score, OBV slope, realized/downside volatility, and drawdown
   pressure. Do not import hundreds of highly overlapping variants.
3. **Signal-level redundancy removal.** Measure both rank correlation and
   top-bucket overlap. Keep one representative from a correlated cluster before
   blending, rather than asking the portfolio optimizer to remove redundant
   signals after the fact.
4. **Equal-weight composite first.** With limited independent dates, use equal
   weights after redundancy removal. ICIR- or score-weighted blends should wait
   until enough sealed date blocks exist to avoid fitting noise.
5. **Multi-horizon validation.** Record per-date Rank IC, IC hit rate, 5/10/20-day
   decay, and shortlist turnover. This identifies whether a signal belongs in a
   short-event engine or the slower allocation engine.
6. **Regime conditioning as context, not a new alpha source.** Track trend,
   realized-volatility state, and breadth; report factor performance by regime.
   Do not dynamically change LIVE weights until each regime has adequate
   independent observations.
7. **SEC event timelines.** Preserve filing acceptance time separately from the
   event/report date; deduplicate by accession and link amendments. This would
   improve the current persistent-security state for 8-K, Form 4, 13D/G, and S-1
   events without asking the LLM to rediscover facts.

The recommended implementation order is: rank transforms and factor-family
diagnostics, redundancy removal and equal-weight composite, multi-horizon
tracking, then structured SEC events. HMMs, deep learning, reinforcement
learning, and optimized factor weights remain poor fits for the current amount
of independent data.

## Sources

- https://github.com/quantskills/quantskills
- https://github.com/quantskills/skill-quant-research
- https://github.com/quantskills/skill-quant-portfolio-risk
- https://github.com/quantskills/skill-quant-execution-microstructure
- https://github.com/quantskills/skill-quant-factor-directional-alpha
- https://github.com/quantskills/skill-residual-guided-factor-selection
- https://github.com/quantskills/skill-overseas-equity-factor-miner
- https://github.com/quantskills/skill-factor-blend
- https://github.com/quantskills/skill-quant-factor-volume-stat-alpha
- https://github.com/quantskills/skill-quant-factor-risk-pattern-alpha
- https://github.com/quantskills/skill-market-regime-analysis
- https://github.com/quantskills/skill-us-sec-edgar-harvester
