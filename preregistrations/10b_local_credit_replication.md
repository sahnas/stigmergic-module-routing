# Preregistration: experiment 10b, independent re-implementation of experiment 10, plus a gradient router with local losses

Written 2026-10-02T03:03Z, before any run on the test seeds.
Code: src/exp_local_credit_b.py (sha256 e930daf98bb4019d), src/exp_no_symbols.py (sha256 88c033106677b739), src/exp.py (sha256 6c202cab844bb559).

Provenance. Experiment 10 (preregistrations/10_local_credit.md, results/10_local_credit/) was found complete in the
execution environment, from an execution absent from the conversation history, committed at 2026-10-02T02:50Z.
The code of 10b was written at 02:58Z without knowledge of it, and briefly overwrote the files of experiment 10,
which were restored from the commit. On the three test seeds that 10b had already run (800 to 802), the two
implementations agree: identical values for the reference, and values within a few hundredths for the two local
variants, which differ in how the local quality of slot 2 is defined (10: the module against the true primitive
applied to its actual input; 10b: the module applied to the true intermediate, against y).

Development seed 900 (results/10_local_credit/dev_seed_900.jsonl), declared before the test: level 1 worse than
the reference on known tasks (0.09 vs 0.22); level 2 at 0.99 on known tasks, alignment 0.99, speed 0.95,
interference 0.16; the gradient router with the same local losses at 0.99, speed 0.96, but alignment -1.16 and
interference 0.63.

Test seeds: 810 to 819 (10 seeds, never used; checked against every result file). Seeds 800 to 802 run by 10b
before the discovery are kept apart as results/10_local_credit/10b_seeds_800_802_before_discovery.jsonl.
Variants: aco_thresholds (reference), aco_local_both (level 2), soft_local_both (gradient router, local losses).

Hypotheses (measures as in experiment 7), two-sided paired Wilcoxon, threshold 0.05 / 4:
- P1 (replication of O4): aco_local_both > aco_thresholds on learning speed for new compositions.
- P2: aco_local_both vs soft_local_both on learning speed, no predicted sign.
- P3: aco_local_both has lower interference on known tasks than soft_local_both.
- P4: aco_local_both has higher module/primitive alignment than soft_local_both.
Descriptive: known-task R2, final R2.
