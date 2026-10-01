# Preregistration: strong opponent, change-point bandit

> English translation of the French original kept verbatim in `original-fr/`. The hashes below refer to the French files as executed, which are in `original-fr/`. The translated code in `src/` differs only in comments, strings and identifiers; it reproduces stored results bit for bit (see README).

> **Correction added at publication.** This document states that seeds 60 to 79 were never used. That is false: seeds 60 to 74 had been used by the standard-bandit protocol (experiment 4), found only later. No choice here depended on those results, which were unknown at the time, and the code is deterministic (the tested mechanism gives identical results on shared seeds across runs), but the statement was inaccurate. The opponent tested here belongs to the same family as M-UCB in experiment 4.

Written 2026-10-01T04:43Z, before any run on the test seeds.

Code: exp_rupture.py (sha256 abc0cf698ac27474), exp_dynamique.py (sha256 c5f0595ecf9201ca), exp.py (sha256 9b0d44c40e14df1d).
Production scenario identical to the "gradient-free" test.

Tuning of the opponent: 16 configurations on development seeds 300, 301, 302 (results/5_changepoint/summary.txt).
Kept configuration (best mean development availability, 0.619): UCB, c = 0.1, h = 20, b = 0.2, window 300.
The tested mechanism (aco_thresholds) is not tuned: same hyperparameters since the first test. Deliberate advantage to the opponent.

Test seeds: 60 to 79 (20 seeds, never used). Variants: ema, aco_thresholds, changepoint.

Hypotheses:
- R1: aco_thresholds against changepoint in availability, two-sided. "Gradient-free" thesis maintained if p < 0.025 and difference > 0;
  invalidated if p < 0.025 and difference < 0; undetermined otherwise.
- R2: changepoint > ema in availability (the opponent is much stronger than the previous status quo), threshold 0.025.
Test: two-sided paired Wilcoxon. Descriptive: final R², final zero-shot, availability per event, collapses.
