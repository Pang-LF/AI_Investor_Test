# Opportunity Capture Audit

Signal window: 2026-05-20 through 2026-08-14

This is a diagnostic of opportunity coverage, not a return backtest. The broad and
active universes are frozen from the current scanner snapshot, so results contain
survivorship and selection bias. Repeated overlapping signal dates are opportunity
observations, not independent events.

## Funnel results

| Horizon | Winner observations | Active-investable coverage | Conditional quant recall | Pass precision/lift | Pass mean excess |
|---|---:|---:|---:|---:|---:|
| 5d | 2640 | 14.4% | 0.8% | 60.0% / 4.41x | 2.45% |
| 20d | 2640 | 13.8% | 0.3% | 20.0% / 1.53x | -10.42% |

Active-investable coverage measures discovery, while conditional quant recall measures
what the model/funnel retained after a name was already inside the active pool.
Pass precision alone is not profitability: the realized return distribution also matters.

| Horizon | Event-proxy recall | Quant passes | Positive rate | Active-pool base rate |
|---|---:|---:|---:|---:|
| 5d | 29.8% | 5 | 60.0% | 13.6% |
| 20d | 26.1% | 5 | 40.0% | 13.1% |

## Forecast ranking and economic attribution

Daily cross-sectional 20d forecast buckets:

| Forecast bucket | N | Avg predicted | Avg realized 20d | Median | Win rate | P10 |
|---|---:|---:|---:|---:|---:|---:|
| Top 5% | 120 | 2.88% | 0.36% | 1.66% | 55.0% | -26.26% |
| 5-10% | 180 | 1.97% | -0.68% | -3.63% | 43.3% | -31.20% |
| 10-20% | 240 | 1.31% | 2.98% | 2.59% | 57.1% | -20.60% |
| 20-40% | 572 | 0.85% | 2.88% | 1.90% | 56.3% | -14.05% |
| 40-60% | 568 | 0.54% | 3.10% | 2.26% | 59.7% | -11.40% |
| Bottom 40% | 1112 | -0.04% | 2.44% | 1.95% | 57.3% | -18.12% |

Research-rank cohorts:

| Rank cohort | N | Avg realized 5d | Avg realized 20d | 20d win rate | 20d worst |
|---|---:|---:|---:|---:|---:|
| 1-3 | 180 | 0.15% | -1.42% | 49.4% | -65.18% |
| 4-5 | 120 | 0.76% | 1.46% | 45.8% | -53.92% |
| 6-10 | 300 | 1.27% | 3.30% | 57.3% | -46.96% |
| 11-15 | 300 | 1.64% | 4.05% | 60.0% | -51.76% |
| 16-20 | 294 | 1.34% | 2.18% | 53.4% | -45.76% |
| 21+ | 461 | 0.72% | 1.60% | 55.1% | -41.29% |

Funnel return attribution:

| Stage | N | Avg realized 20d | Median | Win rate | P10 |
|---|---:|---:|---:|---:|---:|
| investment_probability_gate | 295 | -0.09% | -0.97% | 48.1% | -29.83% |
| outside_active_investable_universe | 23340 | 2.26% | 1.93% | 56.7% | -14.27% |
| quant_pipeline_pass | 5 | -10.42% | -2.92% | 40.0% | -38.84% |
| quant_research_gate | 1137 | 2.86% | 2.09% | 59.1% | -14.79% |
| research_capacity | 1355 | 2.64% | 1.91% | 56.3% | -17.02% |

Rank IC (daily Spearman correlation):

| Horizon | Date blocks | Mean IC | Median IC | IC hit rate | Block-bootstrap 95% CI | Non-overlap mean |
|---|---:|---:|---:|---:|---:|---:|
| 5d | 60 | -0.000 | -0.035 | 41.7% | [-0.067, 0.072] | -0.009 (12 blocks) |
| 20d | 60 | -0.016 | -0.005 | 48.3% | [-0.114, 0.091] | 0.027 (3 blocks) |

Matched-universe capture benchmarks:

| Horizon | Benchmark | Mean capture | 5-95% range | Observed capture | Observed lift |
|---|---|---:|---:|---:|---:|
| 5d | Random | 10.8% | 8.5%-13.3% | 14.4% | 1.33x |
| 5d | Market-cap bucket | 12.5% | 10.3%-14.8% | 14.4% | 1.15x |
| 5d | Sector | 9.7% | 7.8%-11.7% | 14.4% | 1.48x |
| 5d | Liquidity | 11.7% | 9.2%-14.4% | 14.4% | 1.23x |
| 5d | Size + sector + liquidity | 15.0% | 13.9%-16.1% | 14.4% | 0.96x |
| 20d | Random | 10.8% | 7.3%-14.7% | 13.8% | 1.28x |
| 20d | Market-cap bucket | 13.1% | 9.7%-16.6% | 13.8% | 1.06x |
| 20d | Sector | 9.0% | 6.2%-12.1% | 13.8% | 1.53x |
| 20d | Liquidity | 11.6% | 8.0%-15.3% | 13.8% | 1.19x |
| 20d | Size + sector + liquidity | 14.6% | 12.8%-16.2% | 13.8% | 0.95x |

Conditional calibrated-probability results within forecast-rank bands (20d):

| Forecast rank | Probability band | N | Avg realized | Median | Win rate |
|---|---|---:|---:|---:|---:|
| top_10pct | pct_50_55 | 295 | -0.09% | -0.97% | 48.1% |
| top_10pct | pct_55_60 | 5 | -10.42% | -2.92% | 40.0% |
| pct_10_20 | pct_50_55 | 240 | 2.98% | 2.59% | 57.1% |
| pct_20_40 | pct_50_55 | 572 | 2.88% | 1.90% | 56.3% |
| bottom_60pct | below_50pct | 446 | 5.12% | 4.84% | 60.8% |
| bottom_60pct | pct_50_55 | 1234 | 1.77% | 1.63% | 57.1% |

Extreme-forecast autopsy overview:

| Metric | Top 5% mean | 10-20% mean |
|---|---:|---:|
| feature_l2_distance | 5.055 | 2.662 |
| maximum_absolute_feature_zscore | 3.929 | 1.983 |
| trailing_volatility_20d | 1.277 | 0.691 |
| maximum_absolute_return_5d | 0.1368 | 0.08068 |
| event_days_20d | 10.88 | 5.6 |
| average_dollar_volume_20d | 2.924e+09 | 1.339e+09 |

The date-clustered 10-20% minus Top-5% realized-return difference was 2.62%; moving-block-bootstrap 95% CI [-4.05%, 9.74%]. The non-overlapping estimate was 10.64% across 3 blocks.

## Diagnostic interpretation

- Within this biased current-snapshot diagnostic, the investable observation slots captured 14.4% of 5d and 13.8% of 20d winner observations. The corresponding random-selection lifts were only 1.33x and 1.28x; discovery value must be judged against the matched baselines rather than against 100% coverage.
- Inside the active pool, the complete quantitative funnel retained only 0.8% of 5d and 0.3% of 20d winners; research capacity is a mechanical rejection stage, not proof that expanding research depth creates alpha.
- Quantitative passes enriched the winner rate by about 4.41x / 1.53x, but their mean realized excess was 2.45% / -10.42%. Only 5 observations passed the complete 20d funnel, so pass precision is statistically uninformative and does not justify loosening gates.
- The 20d daily Rank IC was -0.016 with block-bootstrap interval [-0.114, 0.091]. The corrected investable-universe audit therefore does not establish even weak cross-sectional ranking skill in this window.

## 5d attribution

| Stage | Count |
|---|---:|
| investment_probability_gate | 74 |
| outside_active_investable_universe | 2260 |
| quant_pipeline_pass | 3 |
| quant_research_gate | 112 |
| research_capacity | 191 |

Largest missed winners:

| Date | Symbol | Excess return | Stage | Event proxy |
|---|---|---:|---|---|
| 2026-08-07 | QMCO | 92.96% | outside_active_investable_universe | False |
| 2026-05-21 | UMAC | 85.12% | research_capacity | False |
| 2026-05-22 | APPS | 78.10% | outside_active_investable_universe | False |
| 2026-06-18 | BLZE | 71.12% | outside_active_investable_universe | False |
| 2026-06-12 | WYFI | 64.31% | outside_active_investable_universe | False |
| 2026-05-21 | DELL | 54.57% | outside_active_investable_universe | True |
| 2026-05-22 | SNOW | 54.38% | outside_active_investable_universe | True |
| 2026-06-11 | BFLY | 53.70% | outside_active_investable_universe | True |
| 2026-06-23 | FCEL | 51.69% | investment_probability_gate | True |
| 2026-06-01 | VSXY | 48.30% | outside_active_investable_universe | False |
| 2026-05-28 | MRVL | 47.60% | outside_active_investable_universe | True |
| 2026-07-22 | RNG | 46.55% | outside_active_investable_universe | True |
| 2026-08-04 | HZO | 46.07% | outside_active_investable_universe | False |
| 2026-08-06 | TEAM | 45.97% | outside_active_investable_universe | True |
| 2026-05-22 | OKTA | 45.81% | outside_active_investable_universe | False |

## 20d attribution

| Stage | Count |
|---|---:|
| investment_probability_gate | 62 |
| outside_active_investable_universe | 2275 |
| quant_pipeline_pass | 1 |
| quant_research_gate | 131 |
| research_capacity | 171 |

Largest missed winners:

| Date | Symbol | Excess return | Stage | Event proxy |
|---|---|---:|---|---|
| 2026-07-16 | QMCO | 101.79% | outside_active_investable_universe | True |
| 2026-06-09 | BLZE | 93.04% | outside_active_investable_universe | False |
| 2026-05-28 | BFLY | 90.33% | outside_active_investable_universe | False |
| 2026-05-26 | APPS | 89.61% | outside_active_investable_universe | True |
| 2026-07-23 | TEAM | 84.79% | outside_active_investable_universe | True |
| 2026-05-20 | UMAC | 73.00% | research_capacity | True |
| 2026-06-01 | FCEL | 71.73% | investment_probability_gate | False |
| 2026-07-20 | AEHR | 62.93% | outside_active_investable_universe | True |
| 2026-06-01 | ALHC | 62.88% | research_capacity | True |
| 2026-05-20 | VSXY | 62.62% | outside_active_investable_universe | True |
| 2026-06-01 | UCTT | 61.60% | outside_active_investable_universe | False |
| 2026-07-28 | CAI | 60.83% | quant_research_gate | False |
| 2026-06-01 | ORKA | 60.13% | outside_active_investable_universe | True |
| 2026-07-22 | RNG | 59.83% | outside_active_investable_universe | True |
| 2026-05-20 | MRVL | 58.71% | outside_active_investable_universe | True |

## Limitations

- Broad and active universes are the current scanner snapshot, not historical constituents.
- This has survivorship and selection bias and is not a performance backtest.
- Exact historical market caps are unavailable; scanner size buckets are the market-cap proxy.
- Daily event_proxy is not the production intraday event scanner.
- Overlapping signal dates count repeated opportunities and are not independent events.
- Historical news, LLM verdicts, portfolio state, and optimizer decisions are unavailable before logging began.

## Actual logged production funnel

Decision runs available: 10

| Disposition | Count |
|---|---:|
| investment_eligibility_failed | 12 |
| not_researched | 165 |
| order_blocked | 2 |
| order_submitted | 1 |

Only actual logged runs contain historical LLM and optimizer outcomes.
