# Preregistration: experiment 9, heterogeneous modules

Written 2026-10-01T11:24Z, before any run on the test seeds.
Code: src/exp_heterogeneous.py (sha256 f7f63f18264b82b6), src/exp_dynamic.py (sha256 c3329b81c02b0eb6), src/exp_bandits.py (sha256 aa3d44d1bfc727da),
src/exp_ablation.py (sha256 9d554f028d6cee16), src/exp.py (sha256 6c202cab844bb559).

Production scenario of experiment 3, unchanged, except that module hidden sizes are {2, 4, 8, 16, 32, 64, 2, 4, 8, 16}
in a seed-dependent order. One development seed (401) was run only to check that the code works; no variant is tuned,
and the bandits keep the configurations selected in experiment 4.

Test seeds: 720 to 739 (20 seeds, checked unused). Variants: aco_thresholds, aco_global_threshold, swucb:50:0.05, ducb:0.98:0.3.
Measure: availability.

Hypotheses (paired Wilcoxon, Bonferroni threshold 0.05 / 3):
- HE1: individual thresholds beat a single collective threshold when modules are heterogeneous
  (aco_thresholds > aco_global_threshold). Directional. Prior estimate stated earlier: about 35 %.
- HE2: aco_thresholds vs SW-UCB, two-sided.
- HE3: aco_thresholds vs D-UCB, two-sided.
Confirmed if p below threshold and difference in the stated direction (HE1) or reported with its sign (HE2, HE3).
