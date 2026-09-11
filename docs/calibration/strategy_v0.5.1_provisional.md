# Strategy v0.5.1 provisional execution calibration

Date: 2026-09-11

## Decision

Enable the calibrated execution layer with provisional floors:

- bias-adjusted 20-day excess return > 0.50%;
- calibrated probability of positive 20-day excess return >= 55%;
- bias-adjusted alpha / forecast uncertainty >= 0.15;
- at least five prior non-overlapping 20-day calibration blocks;
- LLM may veto but may not create alpha;
- optimized portfolio must improve on current holdings/cash after a 0.40%
  estimated round-trip reallocation cost.

## Diagnostic

The reproducible command is:

```bash
.venv/bin/python scripts/walk_forward_calibration.py --date 2026-09-10
```

It used 22 non-overlapping 20-day evaluation dates. Seventeen dates had at least
five prior calibration blocks. For the selected 55% / 0.15 policy, 54 names were
selected across 14 date blocks. Equal-weight block outcomes after the 0.40% cost
assumption had mean 13.45%, median 14.29%, 85.7% positive blocks, and a worst
block of -34.18%.

These large figures must **not** be interpreted as an expected live return. The
test used today's static observation universe across history, causing material
survivorship and selection bias; some dynamic names were known winners by the
time the current universe was formed. Cross-sectional names within a date are
also correlated. The diagnostic establishes that the provisional gate is not
internally empty and is stricter than nearby research qualification. It does not
prove tradable alpha.

## Required prospective review

Every live decision records its forecast, calibration values, LLM verdict,
target, fill, cost, and later realized return. Thresholds must be reviewed after
enough genuinely prospective non-overlapping 20-day blocks accumulate. Any
calibration degradation, unexplained turnover, or execution anomaly should turn
`execution_calibration_approved` off again.
