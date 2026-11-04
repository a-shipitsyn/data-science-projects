# Experiment readout

## 1. Sample ratio check

Treatment 97,252 users (49.30%), control 100,015 — planned 50/50. Chi-square p < 0.001: **sample ratio mismatch (p < 0.001).** Users are missing from one arm, so the arms are no longer comparable and every effect below may be biased. Find the cause (assignment, redirects, logging, bot filtering) before using the results.

Split by segment, to locate the mismatch:

| segment | treatment share | SRM p |
|---|---|---|
| device = desktop | 50.06% | 0.729 |
| device = mobile | 48.78% | < 0.001 |
| tenure_group = 30d-2y | 49.36% | < 0.001 |
| tenure_group = >2y | 49.30% | 0.007 |
| tenure_group = new | 49.16% | < 0.001 |

## 2. Metrics

Primary metric: **revenue**. Two-sided tests, α = 0.05.

| metric | control | treatment | effect (95 % CI) | relative (95 % CI) | p | adjustment |
|---|---|---|---|---|---|---|
| revenue | 56.8 | 59.6 | 1.9 (0.0231 to 3.77) | +3.3% (-0.0% to +6.6%) | 0.047 | CUPED on pre_revenue, variance −17% |
| converted | 0.0949 | 0.101 | 0.00515 (0.00266 to 0.00763) | +5.4% (+2.7% to +8.1%) | < 0.001 | CUPED on pre_revenue, variance −11% |

## 3. Ratio metrics

Delta-method standard errors (users are the randomisation unit, not sessions).

| metric | control | effect (95 % CI) | relative | p |
|---|---|---|---|---|
| revenue / sessions | 8.72 | 0.385 (0.0989 to 0.672) | +4.4% | 0.008 |

## 4. Guardrails

| metric | control | treatment | effect (95 % CI) | relative (95 % CI) | p | adjustment | status |
|---|---|---|---|---|---|---|---|
| sessions | 6.51 | 6.55 | 0.0338 (-0.0343 to 0.102) | +0.5% (-0.5% to +1.6%) | 0.331 | none | ok |

## 5. Sequential check

Always-valid p-value of revenue (mSPRT, normal mixture with prior sd 5% of the control mean) after each day: it stays valid however often the test was looked at.

| day | users | effect | always-valid p |
|---|---|---|---|
| 1 | 14,054 | 6.67 | 0.676 |
| 4 | 56,614 | 4.27 | 0.204 |
| 8 | 113,169 | 3.74 | 0.060 |
| 11 | 155,281 | 2.76 | 0.060 |
| 14 | 197,267 | 1.9 | 0.060 |

Not significant under continuous monitoring (final always-valid p = 0.060); the fixed-horizon test above is valid only if the decision was made once, at the planned end.

## 6. Segments

Effect on revenue by device × tenure_group, CUPED-adjusted. Holm correction for the number of segments.

| segment | users | effect (95 % CI) | p | p (Holm) |
|---|---|---|---|---|
| desktop / 30d-2y | 45,378 | -2.91 (-7.2 to 1.37) | 0.182 | 0.546 |
| desktop / >2y | 14,836 | -33.2 (-41.3 to -25.1) | < 0.001 | < 0.001 |
| desktop / new | 20,044 | 2.95 (-2 to 7.91) | 0.243 | 0.546 |
| mobile / 30d-2y | 66,226 | 8.33 (5.17 to 11.5) | < 0.001 | < 0.001 |
| mobile / >2y | 21,585 | 3.58 (-2.65 to 9.82) | 0.260 | 0.546 |
| mobile / new | 29,198 | 10.5 (6.64 to 14.4) | < 0.001 | < 0.001 |

Heterogeneity (Cochran's Q, 5 df): p < 0.001 — the effect differs between segments.

## 7. Decision

**Investigate — no decision.** sample ratio mismatch: the comparison is not trustworthy.
