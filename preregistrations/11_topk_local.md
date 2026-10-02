# Preregistration: experiment 11, sparse top-k routers with local losses

Written 2026-10-02T06:19Z, before any run on the test seeds.
Code: src/exp_local_credit_topk.py (sha256 c907bfb67d603426), src/exp_local_credit_b.py (sha256 e930daf98bb4019d), src/exp_topk.py (sha256 abfab869d11dba6c), src/exp_no_symbols.py (sha256 88c033106677b739), src/exp.py (sha256 6c202cab844bb559).
Protocol of experiment 10b, unchanged. Top-k configurations taken from experiment 8 (top-1: noise 0.3, balancing 0.01;
top-2: noise 0.3), not re-tuned. The tested mechanism is not tuned.

Question: in 10b, trace routing with local losses gave a clean decomposition (alignment 0.99, interference 0.16)
where a dense gradient router did not (-1.65, 0.66). Is that a property of traces, or of hard assignment in general?

Development seed 901 (results/11_topk_local/dev_seed_901.jsonl), declared before the test: top-1 reached known
tasks 0.97, alignment 0.36, speed 0.72, interference 0.25; top-2 reached 0.99, alignment 0.94, speed 0.89,
interference 1.36. Trace routing on its own development seed (900) had alignment 0.99, speed 0.95, interference 0.16.
Prior stated before this experiment: 60 % that a sparse router aligns as well as traces. The development seed
suggests top-2 aligns almost as well but interferes much more, and top-1 aligns less; I now put that at 35 %.

Test seeds: 820 to 829 (10 seeds, never used; checked against every result file).
Variants: aco_local_both (reference), topk1_local_both, topk2_local_both.

Hypotheses, two-sided paired Wilcoxon, threshold 0.05 / 6, no predicted sign:
- K1, K2, K3: aco_local_both vs topk1_local_both on learning speed, interference, alignment.
- K4, K5, K6: aco_local_both vs topk2_local_both on the same three measures.
Reading fixed in advance: if a sparse router matches traces on all three, the 10b result is about hard assignment,
not traces; if traces keep a significant edge on alignment or interference, the edge is specific to the trace rule.
Descriptive: known-task R2, final R2.
