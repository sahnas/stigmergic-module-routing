# Preregistration: experiment 15b, savings of a consumed specialist, corrected protocol

Written 2026-10-04T16:02Z, before any run on the test seeds.
Code: src/exp_savings_b.py (sha256 aa641342b2deca8f), src/exp_preservation.py (sha256 b112fd314196a825), src/exp_dynamic.py (sha256 c3329b81c02b0eb6), src/exp.py (sha256 6c202cab844bb559).

Why: an external review (2026-10-04) found that experiment 15 reset torch.manual_seed before each candidate,
which does not reset the world's own generator; the three candidates therefore relearned on different batches,
contrary to the comment in the code. Here the 150 relearning batches and a 1000-sample evaluation batch are
drawn once per seed from a dedicated generator and reused for every candidate. Everything else is as in 15.

Development seed 953, declared before the test (results/15b_savings/dev_seed_953.jsonl): fresh 0.77, consumed
specialist 0.62, other specialist 0.63 on the area under the curve; the fresh module was fastest.

Test seeds: 1210 to 1219 (10 seeds, never used; checked against every result file).
Hypotheses, two-sided paired Wilcoxon on the area under the curve, threshold 0.05 / 3:
- S1: consumed specialist vs fresh module.
- S2: consumed specialist vs other live specialist.
- S3 (preregistered this time, exploratory in 15): other live specialist vs fresh module.
Prior: after experiment 15 and the development seed, 25 % for S1 positive, 60 % for S2 positive, 65 % for S3
negative (repurposed module worse than fresh).
