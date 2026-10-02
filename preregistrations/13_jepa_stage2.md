# Preregistration: experiment 13, modular JEPA stage 2 (lifelong discovery, a sixth primitive appears)

Written 2026-10-02T10:28Z, before any run on the test seeds.
Code: src/jepa/stage2_lifelong.py (sha256 c580e20533978d08), src/jepa/stage1_competition.py (sha256 a061925fbfbebb3a), src/jepa/world_seq.py (sha256 be0713217e1343f9).
Design: design/modular_jepa.md. No hyperparameter tuned; values fixed a priori (lifelong phase 1000 steps, surprise
threshold at three deviations of the instantaneous best error, activation when more than 10 % of a batch is surprise).

A first version of the stage-2 code used smoothed errors for surprise detection and flagged about 40 % of all
transitions, because switches to known primitives raise the smoothed error for three steps; it was replaced
before any test seed by detection on instantaneous errors. Both development runs are in results/13_jepa_stage2/
(dev_seed_960.jsonl is the version under test).

Development seed 960, declared before the test: competition5 gave the sixth primitive to a former specialist
(new_module_was_specialist 1, new_module_prior 0.14), error on the new primitive 0.57, interference -0.02;
recruit5 activated one reserve (new_module_prior 0.0), error on the new primitive 0.42, interference -0.08.

Test seeds: 1010 to 1014 (5 seeds, never used; checked against every result file; about 90 s per run).
Variants: competition5, competition8, recruit5.

Hypotheses (with 5 seeds no Wilcoxon test can reach a threshold; results are reported as effect sizes with the
five per-seed values, as in experiment 12):
- F1: competition5 gives the sixth primitive to a former specialist on more seeds than recruit5 (expected 5/5 vs 0/5).
- F2: recruit5 has a lower relative error on the sixth primitive than competition5.
- F3: recruit5 has lower interference (after minus before, on the old primitives) than competition5.
- F4: recruit5 vs competition8 (COMET with spare generalists) on the same two measures, no predicted sign.
Reading fixed in advance: F1 is the capability COMET lacks (instantiating a mechanism); F2 and F3 say whether it
pays; F4 says whether dormant reserves plus recruitment beat merely over-provisioning predictors.
