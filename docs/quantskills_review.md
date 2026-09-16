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

## Sources

- https://github.com/quantskills/quantskills
- https://github.com/quantskills/skill-quant-research
- https://github.com/quantskills/skill-quant-portfolio-risk
- https://github.com/quantskills/skill-quant-execution-microstructure
- https://github.com/quantskills/skill-quant-factor-directional-alpha
- https://github.com/quantskills/skill-residual-guided-factor-selection
