# Draft preregistration (not run, for review): experiment 22, cost discrepancy over the plans the planner considers

Status: DRAFT written 2026-10-13, revised the same day after external review. Nothing has been run. Code not written.
To be frozen with code hashes before any run. Order decided by review: the objective diagnostic (Singh-style shared
decoded-position cost) runs first; this experiment runs only if it still adds decisive information afterwards.

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
- Solver: the authors' own CEM at the pinned revision, instrumented as in experiment 21 (not a reproduction), run
  separately for each predictor from the same start and goal, with the context the planner actually has at an
  episode start (one frame, then its own predictions), not three real frames.
- Considered plans: the 300 candidates of the last iteration plus the final mean, each passed through exactly the
  action transformation its environment applies at execution (Two-rooms clips to its bounds; PushT does not clip at
  the revision used), before both the predicted cost and the execution; the two environments are therefore not
  treated alike, by design.
- Predicted cost: the predictor's own rollout of the environment-transformed plan from the one-frame initial context,
  squared latent distance to the goal embedding, using the official cost definition. True cost: the same transformed plan executed in 64 cloned
  environment copies at a time from the same state, terminal frame encoded by the frozen encoder, same distance.
- Discrepancy per plan: |true - predicted|; per start and predictor: maximum (primary, the bound's quantity), 95th
  percentile (secondary, sensitivity to extremes), mean over the 301 plans, and the value at the chosen plan; also
  the ratio true/predicted for comparability with experiment 18.
- Selection regret, at no extra simulation: true cost of the chosen plan minus the lowest true cost among the 301
  executed plans, per start and predictor. The CEM returns an elite mean, which need not be the lowest-predicted-cost
  candidate. This is a regret on a finite set under the latent objective, not a closed-loop success measure.
- Budget: 30 starts x 301 plans x 3 predictors x 2 tasks = 54,180 executed plans of 25 steps; feasible on a T4.

## Hypotheses and priors (to be finalised at freeze)
- H1: the ordering of the three predictors by the median over starts of the per-start maximum discrepancy equals
  the ordering by planning success on both tasks (Two-rooms: bilinear, then the two networks; PushT: corrected
  network, released, bilinear). Prior 50 %.
- H2: on PushT, the bilinear predictor's maximum discrepancy exceeds both networks' by at least a factor of 2 in
  median over starts. Prior 60 %.
- Descriptive: the same with the 95th percentile and the mean; the discrepancy at the chosen plan; comparison with
  the orderings by average latent error (expected to fail on Two-rooms, as in experiments 18 to 20).
Reading fixed at freeze: a bound does not guarantee a ranking (a model that adds a constant to every cost has a large
discrepancy and a perfect selection). If H1 holds on both tasks, the admissible conclusion is that this measure
reproduces the observed rankings in this protocol; its causal role remains to be established. If it fails on either
task, the bound's quantity does not reproduce the rankings here; this does not refute the bound.
## Related work to integrate before any claim
Nair, Savarese and Finn (2020, Goal-Aware Prediction) show that errors on low-true-cost trajectories matter and
errors on high-cost ones hardly do; "Not All Errors Matter" (2026) proposes decision-relevant prediction error
over 55 models with a common planner and formalises when it ranks models (our experiment 20 found task-state error
insufficient on Two-rooms, which must be confronted with their criterion); AD-WM (2026) measures candidate-selection
quality and regret on shared candidate banks on LeWorldModel and reports the lowest factual error paired with the
lowest selection quality; Singh (2026) notes context length as an explanation one-step error cannot be separated
from with three checkpoints. The review also cited Min et al., "When Low Prediction Error Misleads Planning"; the
title was not located at the time of this revision and is to be confirmed with a link.

## Known limits
Thirty starts; the authors' pinned solver instrumented without changing its search; discrepancy measured on the last iteration's
candidates, not on every candidate of every iteration; one encoder per task shared by the predictors.
