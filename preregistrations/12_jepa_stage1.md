# Preregistration: experiment 12, modular JEPA stage 1 (competition and the prior over the winner)

Written 2026-10-02T07:50Z, before any run on the test seeds.
Code: src/jepa/stage1_competition.py (sha256 a061925fbfbebb3a), src/jepa/world_seq.py (sha256 be0713217e1343f9).
Design: design/modular_jepa.md. No hyperparameter tuned; values fixed a priori (8 predictors for 5 primitives,
3000 steps, warm-start 300, horizon 3, regime persistence 0.8, EMA 0.99, trace evaporation 0.05).

Development seed 950 (results/12_jepa_stage1/dev_seed_950.jsonl), declared before the test: competition gives
coverage 0.77 and module purity 0.83 with all 8 modules used; attention gives 0.28 / 0.37; the classifier on the
state fails to recover after a switch (0.98) while persistence and traces recover (0.09 and 0.09); traces and
persistence are indistinguishable, as expected in a world where the previous winner mostly predicts itself.

Test seeds: 1000 to 1004 (5 seeds, never used; 7 variants of about 35 s each). Variants: none (persistence),
traces, classifier, attention, oracle, random, single.

Hypotheses, two-sided paired Wilcoxon, threshold 0.05 / 3 (with 5 seeds the smallest attainable p is 0.0625,
so no test can reach the threshold: results are reported as effect sizes with all five per-seed differences):
- E1 (replication of COMET): competition (none) has higher module purity than attention.
- E2: traces vs persistence (none) on recovery error at steps 2 and 3 after a switch to a held-out pair; a tie
  is expected.
- E3: persistence has lower recovery error than the state classifier on the same measure.
Reading fixed in advance: in this world the "when" needs memory of the previous winner, not the current state;
stage 1 establishes that and the baselines for stage 2 (lifelong recruitment), which is the real test.
