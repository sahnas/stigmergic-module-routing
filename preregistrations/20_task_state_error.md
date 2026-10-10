# Preregistration: experiment 20, does error in task-state coordinates rank the predictors by planning success?

Written 2026-10-10T19:45Z, before the run. Code: kaggle/lewm-taskstate/taskstate.py (sha256 d26305d359e22253).

## Question
On Two-rooms, held-out latent error ranks the three predictors (corrected network 0.005, released 0.067,
bilinear 0.69) in the opposite order of their planning success (85, 84, 98); on PushT it ranks them in the same
order (0.004, 0.008, 0.068 against 92, 87, 38). Full-latent error is therefore not the measure that matters on
Two-rooms. Does the error restricted to the coordinates of the task state rank them correctly on both tasks?

## Procedure (no fitting choices left open)
- Caches, predictors and splits of experiments 18 and 19: released predictors at the pinned revisions, the corrected
  networks (official loss, 8 epochs), the bilinear predictors (Two-rooms refitted exactly as in 18; PushT the fit
  archived in experiment 19), held-out episodes from generator seed 0, dataset action statistics.
- Task state: Two-rooms, agent position (2); PushT, agent position (2), block position (2), block angle as cosine
  and sine (2); from the datasets' recorded state arrays.
- Probe: ridge regression (lambda 1e-3 x N) from the frozen embedding to the task state, fitted on 300,000 training
  frames; its held-out R2 per coordinate is reported and must exceed 0.9 on every coordinate for that coordinate to
  count, otherwise the coordinate is reported but excluded from the mean.
- For each predictor, open-loop rollouts from three real frames over horizons 1 to 5 blocks (5 frames each) on
  20,000 held-out windows; at each horizon, the full-latent normalised error and the task-state error (predicted
  embedding through the probe, against the true state at that frame, normalised per coordinate by the held-out
  variance, averaged over counted coordinates).

## Hypotheses and priors, written before the run
- H1: on each task, the ordering of the three predictors by task-state error at horizon 5 (lower is better)
  equals their ordering by mean planning success over the four evaluation seeds (Two-rooms: bilinear, then the
  two networks, which tie at 85 and 84 and are not ordered; PushT: corrected network, released, bilinear).
  Prior 45 %.
- H2: on PushT, the bilinear predictor's task-state error at horizon 5 is larger on the block coordinates than on
  the agent coordinates by a factor of at least 2. Prior 55 %.
- Descriptive: the same orderings at horizon 1; the full-latent error orderings, expected to fail H1 on Two-rooms.
Reading fixed in advance: if H1 holds on both tasks, task-state error is a candidate offline measure of planning
relevance for this family of predictors, to be tested on a third task before any wider claim; if it fails on either
task, neither full-latent nor task-state error ranks these predictors, and the mechanism question stays open.
Nothing here tests a mechanism; it tests a measure.
