# Opportunity Capture Audit

Signal window: 2026-05-20 through 2026-08-14

This is a diagnostic of opportunity coverage, not a return backtest. The broad and
active universes are frozen from the current scanner snapshot, so results contain
survivorship and selection bias. Repeated overlapping signal dates are opportunity
observations, not independent events.

## Funnel results

| Horizon | Winner observations | Active-60 coverage | Conditional quant recall | Pass precision/lift | Pass mean excess |
|---|---:|---:|---:|---:|---:|
| 5d | 2640 | 15.3% | 6.7% | 23.9% / 2.06x | -0.76% |
| 20d | 2640 | 14.4% | 7.1% | 23.9% / 2.19x | -1.66% |

Active-60 coverage measures discovery, while conditional quant recall measures
what the model/funnel retained after a name was already inside the active pool.
Pass precision alone is not profitability: the realized return distribution also matters.

| Horizon | Event-proxy recall | Quant passes | Positive rate | Active-pool base rate |
|---|---:|---:|---:|---:|
| 5d | 29.3% | 113 | 47.8% | 11.6% |
| 20d | 25.7% | 113 | 50.4% | 10.9% |

## Diagnostic interpretation

- Within this biased static-universe diagnostic, the active 60 captured only 15.3% of 5d and 14.4% of 20d winner observations. Discovery is therefore a larger measured bottleneck than final execution gates.
- Inside the active pool, the complete quantitative funnel retained only 6.7% of 5d and 7.1% of 20d winners; research capacity is the largest downstream rejection stage.
- Quantitative passes enriched the winner rate by about 2.06x / 2.19x, but their mean realized excess was -0.76% / -1.66%. This does not establish tradable alpha and does not justify simply loosening gates.

## 5d attribution

| Stage | Count |
|---|---:|
| investment_probability_gate | 60 |
| outside_static_active_60 | 2236 |
| quant_pipeline_pass | 27 |
| quant_research_gate | 103 |
| research_capacity | 214 |

Largest missed winners:

| Date | Symbol | Excess return | Stage | Event proxy |
|---|---|---:|---|---|
| 2026-08-07 | QMCO | 92.96% | outside_static_active_60 | False |
| 2026-05-21 | UMAC | 85.12% | research_capacity | False |
| 2026-05-22 | APPS | 78.10% | outside_static_active_60 | False |
| 2026-06-18 | BLZE | 71.12% | outside_static_active_60 | False |
| 2026-06-12 | WYFI | 64.31% | outside_static_active_60 | False |
| 2026-05-21 | DELL | 54.57% | outside_static_active_60 | True |
| 2026-05-22 | SNOW | 54.38% | outside_static_active_60 | True |
| 2026-06-01 | VSXY | 48.30% | outside_static_active_60 | False |
| 2026-05-28 | MRVL | 47.60% | outside_static_active_60 | True |
| 2026-07-22 | RNG | 46.55% | outside_static_active_60 | True |
| 2026-08-04 | HZO | 46.07% | outside_static_active_60 | False |
| 2026-08-06 | TEAM | 45.97% | outside_static_active_60 | True |
| 2026-05-22 | OKTA | 45.81% | outside_static_active_60 | False |
| 2026-08-10 | NBIS | 43.31% | quant_research_gate | False |
| 2026-07-29 | AAOI | 40.11% | research_capacity | True |

## 20d attribution

| Stage | Count |
|---|---:|
| investment_probability_gate | 45 |
| outside_static_active_60 | 2260 |
| quant_pipeline_pass | 27 |
| quant_research_gate | 92 |
| research_capacity | 216 |

Largest missed winners:

| Date | Symbol | Excess return | Stage | Event proxy |
|---|---|---:|---|---|
| 2026-07-16 | QMCO | 101.79% | outside_static_active_60 | True |
| 2026-06-09 | BLZE | 93.04% | outside_static_active_60 | False |
| 2026-05-26 | APPS | 89.61% | outside_static_active_60 | True |
| 2026-07-23 | TEAM | 84.79% | outside_static_active_60 | True |
| 2026-05-20 | UMAC | 73.00% | research_capacity | True |
| 2026-06-01 | FCEL | 71.73% | investment_probability_gate | False |
| 2026-07-20 | AEHR | 62.93% | outside_static_active_60 | True |
| 2026-05-20 | VSXY | 62.62% | outside_static_active_60 | True |
| 2026-06-01 | UCTT | 61.60% | outside_static_active_60 | False |
| 2026-06-01 | ORKA | 60.13% | outside_static_active_60 | True |
| 2026-07-22 | RNG | 59.83% | outside_static_active_60 | True |
| 2026-05-20 | MRVL | 58.71% | outside_static_active_60 | True |
| 2026-05-20 | DELL | 58.64% | outside_static_active_60 | False |
| 2026-05-20 | ARM | 57.81% | outside_static_active_60 | True |
| 2026-07-23 | WIX | 56.91% | outside_static_active_60 | False |

## Limitations

- Broad and active universes are the current scanner snapshot, not historical constituents.
- This has survivorship and selection bias and is not a performance backtest.
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
