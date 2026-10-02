# Preregistration: experiment 15, savings of a consumed specialist

Written 2026-10-02T22:29Z, before any run on the test seeds.
Code: src/exp_savings.py (sha256 1967f6424c6d635a), src/exp_preservation.py (sha256 b112fd314196a825), src/exp_dynamic.py (sha256 c3329b81c02b0eb6), src/exp.py (sha256 6c202cab844bb559).

Question (from a reviewer's remark): after experiment 14 consumed a specialist, does it keep enough structure to
relearn its function faster than a fresh module (Ebbinghaus' savings)? If yes, the cost of losing a module has an
operational definition (relearning time) and preservation can rely on cheap reconstruction; if no, only dormant
reserves protect a function.

Protocol: replay of experiment 14 (learned then rare, no reserve, least-committed recruitment), then isolated
relearning of the single task (r,) for 150 steps with the module's own Adam optimizer (lr 3e-3), same batches
for every candidate: the consumed specialist of r, a live specialist of another primitive b, a fresh module.
Measures: R2 before relearning, area under the R2 curve (every 5 steps), first step reaching 0.8.

Development seeds 951 and 952, declared before the test (results/15_savings/dev_seeds_951_952.jsonl): on 951 the
consumed specialist and the fresh module were close (0.73 vs 0.72), on 952 the fresh module was clearly faster
(0.77 vs 0.53). Prior stated before writing the code: 65 % for net savings of the consumed specialist over the
fresh module; after the development seeds, revised to 35 %.

Test seeds: 1200 to 1209 (10 seeds, never used; checked against every result file).
Hypotheses, two-sided paired Wilcoxon on the area under the curve, threshold 0.05 / 2:
- S1: consumed specialist vs fresh module. Savings would be a positive difference.
- S2: consumed specialist vs other live specialist. A positive difference would indicate structure specific to r
  beyond generic features.
Descriptive: steps to 0.8, R2 before relearning.
Reading fixed in advance: a positive S1 means forgetting is partly superficial here and reconstruction is cheap;
a null or negative S1 means the overwriting is complete and the specialist is worth no more than a fresh module,
so nothing short of a dormant reserve protects a function in this mechanism.
