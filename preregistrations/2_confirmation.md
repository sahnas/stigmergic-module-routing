# Preregistration: confirmation test

> English translation of the French original kept verbatim in `original-fr/`. The hashes below refer to the French files as executed, which are in `original-fr/`. The translated code in `src/` differs only in comments, strings and identifiers; it reproduces stored results bit for bit (see README).

Written 2026-09-30T23:15Z, before any run on the seeds below.

Unchanged code: exp.py, sha256 9b0d44c40e14df1d. Same hyperparameters, no tuning.
New seeds, never used: 20 to 29.
Variants: soft, aco_thresholds (aco run for reference, outside the test).

Hypotheses (from the exploratory analysis on seeds 0 to 9):
- C1: aco_thresholds > soft in zero-shot generalisation (zero_r2).
- C2: aco_thresholds > soft in retention at the end of phase 1 (train_r2).

Test: two-sided paired Wilcoxon, threshold 0.025 per test (Bonferroni over 2).
Confirmation criterion: p < 0.025 AND positive mean difference.
A null or reversed result = not confirmed, reported as is.
