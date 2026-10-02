# Preregistration: experiment 10, oracle local credit

Written 2026-10-02T02:39Z, before any run on the test seeds.
Code: src/exp_local_credit.py (sha256 3f6b3189b357e361), src/exp_no_symbols.py (sha256 88c033106677b739), src/exp.py (sha256 6c202cab844bb559).
No hyperparameter changed. One development seed (500) to check that the code runs.

Question: in the no-symbols world of experiment 7, is the lock the credit assignment or the traces?
Oracle local quality per slot (module output against the true primitive applied to its actual input).
Variants: aco_thresholds (baseline, global credit), local_routing (local quality for traces and alarms only),
local_full (local quality for traces and alarms, and local learning target for each module), soft (reference only).

What the development seed showed, declared before the test: local_routing did not improve the known tasks
(R² 0.14 against 0.42 for the baseline on that seed in experiment 7's development run) and did not align modules
with primitives; local_full reached R² 0.98 on known tasks, alignment 0.99, learning speed 0.97 on new
compositions and low interference. I therefore expect O1 to O3 to fail and O4 to succeed.

Test seeds: 800 to 809 (10 seeds, never used; checked against the seed ledger). Measures as in experiment 7.
Hypotheses, two-sided paired Wilcoxon, threshold 0.05 / 4 = 0.0125, difference in the stated direction:
- O1: local_routing > aco_thresholds on known tasks (phaseA_r2).
- O2: local_routing > aco_thresholds on module/primitive alignment.
- O3: local_routing > aco_thresholds on learning speed for new compositions (fewshot_auc).
- O4: local_full > aco_thresholds on learning speed for new compositions (fewshot_auc).
Reading: if O4 holds while O1 to O3 fail, the lock is the learning signal of the modules (content), not the
routing credit alone; traces indexed by opaque identifiers can then find a correct path once modules are correct.
