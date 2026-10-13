# Draft preregistration (not run, for review): experiment 22, cost discrepancy over the plans the planner considers

Status: DRAFT written 2026-10-13 for external review. Nothing has been run. Code not written. To be frozen with code
hashes before any run.

## Motivation
"A Control Theory of Predictability in Latent World Models" (arXiv 2607.10362) bounds the planner's suboptimality
by the largest discrepancy between predicted and true plan cost over the plans the planner considers, not by the
average held-out prediction error. Our experiments 18 to 21 show that neither average latent error nor task-state
error orders the three predictors on both tasks. This experiment measures the quantity the bound points at.

## Question
On each task, does the maximum (and 95th percentile) over considered plans of |true cost - predicted cost| order
the three predictors as their planning success does: bilinear lowest on Two-rooms, highest on PushT?

## Procedure (to be frozen)
- Predictors, encoders, caches and action statistics of experiments 18 and 19, unchanged.
- Starts: 30 per task, drawn exactly as the official evaluator draws them (seed 42, dataset rows with the goal 25
  steps ahead), environment state set from the dataset row as `eval.py` does; the first 30 of the official 100.
- Solver: the authors' CEM reproduced as in experiment 18's exploitation probe (300 samples, 30 iterations, 30
  elites, warm-started from zero), run separately for each predictor from the same start and goal.
- Considered plans: the 300 candidates of the last iteration plus the final mean, clipped to the environment's
  action bounds before both the predicted cost and the execution (the action-bounding control showed unclipped
  comparisons inflate discrepancies).
- Predicted cost: the predictor's own rollout of the clipped plan from the true three-frame context, squared
  latent distance to the goal embedding, as the solver computes it. True cost: the clipped plan executed in 64 cloned
  environment copies at a time from the same state, terminal frame encoded by the frozen encoder, same distance.
- Discrepancy per plan: |true - predicted|; per start and predictor: maximum, 95th percentile, mean over the 301
  plans, and the value at the chosen plan; also the ratio true/predicted for comparability with experiment 18.
- Budget: 30 starts x 301 plans x 3 predictors x 2 tasks = 54,180 executed plans of 25 steps; feasible on a T4.

## Hypotheses and priors (to be finalised at freeze)
- H1: the ordering of the three predictors by the median over starts of the per-start maximum discrepancy equals
  the ordering by planning success on both tasks (Two-rooms: bilinear, then the two networks; PushT: corrected
  network, released, bilinear). Prior 50 %.
- H2: on PushT, the bilinear predictor's maximum discrepancy exceeds both networks' by at least a factor of 2 in
  median over starts. Prior 60 %.
- Descriptive: the same with the 95th percentile and the mean; the discrepancy at the chosen plan; comparison with
  the orderings by average latent error (expected to fail on Two-rooms, as in experiments 18 to 20).
Reading fixed at freeze: if H1 holds on both tasks, the planner-reachable discrepancy is the measure that accounts
for the observed rankings in this setting, and the mechanism question of experiments 18 and 19 is answered at the
level of "where the solver looks, which predictor is wrong by most"; if it fails on either task, the bound's
quantity does not account for the rankings here and the question stays open.
## Known limits
Thirty starts; the solver reproduced, not the authors' object; discrepancy measured on the last iteration's
candidates, not on every candidate of every iteration; one encoder per task shared by the predictors.
