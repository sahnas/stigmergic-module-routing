# Preregistration: experiment 8, sparse top-k mixture-of-experts router

Written 2026-10-01T11:24Z, before any run on the test seeds.
Code: src/exp_topk.py (sha256 abfab869d11dba6c), src/exp.py (sha256 6c202cab844bb559).

Tuning of the opponent only, on development seeds 100 to 102: k in {1, 2} x noise in {0.3, 1.0} x alpha in {0, 0.01}
(results/8_topk_moe/tuning_summary.txt). The selection criterion, decided before tuning, was zero-shot R².
Disclosure: the best zero-shot configuration (k = 1, noise 0.3, alpha 0.01) never recovers after the death of a module
(with k = 1 and a dead module, the router receives no gradient). Selecting it alone would make the compensation
comparison trivial, so both the best k = 1 and the best k = 2 configurations are kept as opponents, decided after
seeing the tuning results:
- topk1: k = 1, noise 0.3, alpha 0.01 (dev zero-shot 0.952)
- topk2: k = 2, noise 0.3, alpha 0.0 (dev zero-shot 0.940)
The mechanism (aco_thresholds) is not tuned.

Test seeds: 700 to 719 (20 seeds, checked unused in every result file). Variants: aco_thresholds, topk1, topk2.

Hypotheses, all two-sided with no predicted sign (paired Wilcoxon, Bonferroni threshold 0.05 / 6):
- T1 to T3: aco_thresholds vs topk2 on zero-shot R², retention, and R² of tasks involving k* after recovery.
- T4 to T6: aco_thresholds vs topk1 on the same three measures.
Expectation stated in advance: on development seeds both top-k routers exceed the mechanism's earlier zero-shot and
retention levels, so I expect T1, T2, T4, T5 to favour top-k and T6 to favour the mechanism.
