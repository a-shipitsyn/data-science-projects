# Experiment readout

## 1. Sample ratio check

Treatment 99,985 users (49.99%), control 100,015 — planned 50/50. Chi-square p = 0.947: **OK**.

## 2. Metrics

Primary metric: **revenue**. Two-sided tests, α = 0.05.

| metric | control | treatment | effect (95 % CI) | relative (95 % CI) | p | adjustment |
|---|---|---|---|---|---|---|
| revenue | 56.8 | 58 | 0.569 (-1.28 to 2.42) | +1.0% (-2.3% to +4.3%) | 0.546 | CUPED on pre_revenue, variance −17% |
| converted | 0.0949 | 0.0983 | 0.00269 (0.000242 to 0.00514) | +2.8% (+0.2% to +5.4%) | 0.031 | CUPED on pre_revenue, variance −11% |

## 3. Ratio metrics

Delta-method standard errors (users are the randomisation unit, not sessions).

| metric | control | effect (95 % CI) | relative | p |
|---|---|---|---|---|
| revenue / sessions | 8.72 | 0.178 (-0.105 to 0.462) | +2.0% | 0.217 |

## 4. Guardrails

| metric | control | treatment | effect (95 % CI) | relative (95 % CI) | p | adjustment | status |
|---|---|---|---|---|---|---|---|
| sessions | 6.51 | 6.52 | 0.00283 (-0.0646 to 0.0702) | +0.0% (-1.0% to +1.1%) | 0.934 | none | ok |

## 5. Sequential check

Always-valid p-value of revenue (mSPRT, normal mixture with prior sd 5% of the control mean) after each day: it stays valid however often the test was looked at.

| day | users | effect | always-valid p |
|---|---|---|---|
| 1 | 14,236 | 5.45 | 0.823 |
| 4 | 57,370 | 3.07 | 0.492 |
| 8 | 114,706 | 2.49 | 0.404 |
| 11 | 157,442 | 1.43 | 0.404 |
| 14 | 200,000 | 0.569 | 0.404 |

Not significant under continuous monitoring (final always-valid p = 0.404); the fixed-horizon test above is valid only if the decision was made once, at the planned end.

## 6. Segments

Effect on revenue by device × tenure_group, CUPED-adjusted. Holm correction for the number of segments.

| segment | users | effect (95 % CI) | p | p (Holm) |
|---|---|---|---|---|
| desktop / 30d-2y | 45,378 | -2.91 (-7.19 to 1.37) | 0.183 | 0.549 |
| desktop / >2y | 14,836 | -33.2 (-41.3 to -25.1) | < 0.001 | < 0.001 |
| desktop / new | 20,044 | 2.96 (-2 to 7.91) | 0.242 | 0.549 |
| mobile / 30d-2y | 67,744 | 5.89 (2.8 to 8.98) | < 0.001 | < 0.001 |
| mobile / >2y | 22,086 | 0.819 (-5.28 to 6.92) | 0.793 | 0.793 |
| mobile / new | 29,912 | 8.69 (4.92 to 12.5) | < 0.001 | < 0.001 |

Heterogeneity (Cochran's Q, 5 df): p < 0.001 — the effect differs between segments.

## 7. Decision

**Do not ship as is.** revenue not significant overall (-2.3% to +4.3%), but significantly negative in desktop / >2y and positive in mobile / 30d-2y, mobile / new (Holm-corrected). Find why the treatment hurts the first group, fix it and re-test, or roll out only where it helps, confirmed by a follow-up test.
