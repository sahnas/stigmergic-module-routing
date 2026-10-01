# Preregistration: "gradient-free coordination" thesis

> English translation of the French original kept verbatim in `original-fr/`. The hashes below refer to the French files as executed, which are in `original-fr/`. The translated code in `src/` differs only in comments, strings and identifiers; it reproduces stored results bit for bit (see README).

Written 2026-09-30T23:27Z, before any run on the test seeds.

Code: exp_dynamique.py (sha256 c5f0595ecf9201ca) and exp.py (sha256 9b0d44c40e14df1d).
No hyperparameter changed from the first test. A single development seed (200), used to check that the code runs, without tuning.
Test seeds: 40 to 59 (20 seeds, never used).

2 x 2 factorial design: forgetting rule (ema = recency, aco = evaporation) x per-unit thresholds (no / yes).
Main measure: availability = mean R² over the 17 training tasks, evaluated after each block
throughout the production phase (4 events: death, arrival + death, world drift, double death).

Hypotheses:
- S1: aco_thresholds > ema in availability (the full mechanism against the status quo).
- S2: effect of thresholds > 0, measured per seed as ((ema_thresholds - ema) + (aco_thresholds - aco)) / 2.
- S3: effect of the forgetting rule, ((aco - ema) + (aco_thresholds - ema_thresholds)) / 2, two-sided, no predicted sign.

Test: two-sided paired (or signed-rank on differences) Wilcoxon, threshold 0.05 / 3 = 0.0167.
Criterion: p < 0.0167 and mean difference in the stated direction. Any other result is reported as not confirmed.
Descriptive, outside the test: final R², final zero-shot, availability per event, number of alarms, collapses.
