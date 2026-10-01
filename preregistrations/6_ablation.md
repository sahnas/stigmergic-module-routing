# Preregistration: ablation of the mechanism

> English translation of the French original kept verbatim in `original-fr/`. The hashes below refer to the French files as executed, which are in `original-fr/`. The translated code in `src/` differs only in comments, strings and identifiers; it reproduces stored results bit for bit (see README).

> **Correction added at publication.** This document states that seeds 80 to 99 were never used. That is false: they had been used by the standard-bandit protocol (experiment 4), found only later. No choice here depended on those results, and the tested mechanism gives bit-identical results on these 20 seeds in both runs, but the statement was inaccurate.

Written 2026-10-01T04:57Z, before any run on the test seeds.
Code: exp_mecanisme.py (sha256 02670360a17ee25b), exp_dynamique.py (sha256 c5f0595ecf9201ca), exp.py (sha256 9b0d44c40e14df1d).
One development seed (400) to check that the code runs, without tuning. No hyperparameter changed.
Test seeds: 80 to 99 (20, never used). Variants: aco_thresholds, aco_thresholds_random, aco_global_threshold.

Hypotheses (measure: availability):
- M1: recruiting the least busy unit beats random recruitment (aco_thresholds > aco_thresholds_random).
- M2: thresholds specific to each unit beat a single collective threshold (aco_thresholds > aco_global_threshold).
Test: two-sided paired Wilcoxon, threshold 0.025 per test. Confirmed if p < 0.025 and difference > 0;
contradicted if p < 0.025 and difference < 0; undetermined otherwise.
