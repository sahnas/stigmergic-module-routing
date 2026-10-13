# Experiment 23: does a shared position objective change the TwoRoom predictor gap?

Written 2026-10-13 before any decoder fit, gate episode or intervention evaluation.
Status: protocol and implementation prepared for first submission; **no research
run yet**. The first submitted source must match `kaggle/lewm-objective/freeze.json`
and a pushed Git commit. Any subsequent repair must be recorded before resubmission;
do not overwrite a failed run or interpret implementation failures scientifically.

## Question and scope

With the existing frozen TwoRoom encoder and three predictor fits, does replacing
the planner's terminal latent distance with a **shared decoded-position distance**
materially reduce the bilinear-minus-corrected-network success gap?

Singh's [objective intervention](https://arxiv.org/html/2608.12959v1) motivates this
test. It does not predict its result: his short protocol reports 92% with decoded
positions against 94% with the original cost. Our intervention concerns predictor
differences on a common encoder and uses our archived checkpoints and official
evaluation harness. We do not reproduce his longer-goal protocol or learned
temporal cost. Experiment 22 remains a separate, conditional draft. No automatic
PushT extension is planned.

## Fixed components and changed component

Use experiment 21's three archived TwoRoom checkpoints and their SHA-256 values,
released encoder revision `77adaae0bc31deab21c93740d1f8bb947cd0bdec`, and unchanged
upstream `eval.py` at `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. The original
experiment-21 helper script is imported at repository commit
`944e7ef3a61952c0ba01b5406921180fc9a3dc3f` and checked against SHA-256
`c68e26ed084d1467c8b448a6fdeebe59190fddb756ac552dffea5b15701997ed`.

The `stable-worldmodel==0.1.1` solver, policy, model rollout, world, pool, native
environment, spaces and HDF5-reader sources are hash-checked. The TwoRoom dataset
must have SHA-256 `129a36aa93ea0de488d2bcc876e396de9e3907bf66c6aae6394e542ef6a6d623`.
Exact source and checkpoint values live in `objective.py` and the pinned helper.
Encoder/projector parameter sets and values must agree exactly across networks.
The bilinear checkpoint is loaded, **not fitted again**.

Fit a single affine decoder `D(z) = z P + b` into native agent `(x,y)` units.
The same saved weights are used by all models; verify their SHA-256 in each worker.

- Baseline: official terminal squared latent distance, no criterion wrapper.
- Intervention: `||D(predicted_terminal) - D(encoded_goal)||²`.
- Implementation: replace only the model's terminal `criterion`; preserve its
  `get_cost`, encoding, action embedding and autoregressive rollout.
- The affine bias cancels. Verify the implementation against independently
  decoded positions, not only a duplicate expression using the projected delta.
- No true coordinate is passed into candidate scoring. Coordinates supervise
  the decoder and support diagnostics only. Dynamics/encoder weights stay frozen.

This Euclidean position cost does not encode walls or true reachability. We test
one intervention, not the best possible objective.

## Decoder recipe and data separation

Reconstruct the official 100-task draws for seeds 42–45, including its existing
`len(valid_indices)-1` convention, and require exact equality to the archived
experiment-21 task files. Verify the draw again inside every evaluation worker.
Exclude every selected episode from fitting and validating the new decoder.

For remaining episodes preserve experiment 20's deterministic 90/10 split:
NumPy generator seed 0 chooses `N//10` held-out episode IDs; exclude planning
episodes from both partitions. Generator seed 1 draws up to 300,000 unique training
frame rows. Validate on all admissible held-out frames. Archive episode IDs and
all selected row indices. This does not retroactively exclude planning episodes
from the historical predictor fits; the evaluation seeds are reused.

Fit ridge with penalty `1e-3 * number_of_training_frames`, including the augmented
bias coefficient. Accumulate the 193×193 Gram matrix and 193×2 cross-product in
float64 over batches of 8,192 rows; solve in float64, save float32 coefficients.
These numerical execution details are fixed before fitting; no hyperparameter
selection by planning results.

Use the existing fp32 embedding cache. Archive cache hashes before fitting;
validate all cached episode offsets/lengths, episode IDs and positions against
the pinned dataset. Re-encode the first 64 selected training frames (sorted for
HDF5 access) with the released `model.encode` and official preprocessing; require
maximum absolute embedding disagreement <= 1e-4. Report this as a 64-frame cache
check, not exhaustive proof of cache parity. Since the private cache is not
available in the authoring environment, its byte hash is recorded at initialization
rather than falsely claimed as already verified here.

Decoder feasibility requirements on held-out observations:

- R² >= 0.90 on each coordinate;
- median Euclidean error <= 4 native units;
- 95th percentile <= 8 units;
- finite inputs, outputs and metrics.

These are prespecified engineering thresholds relative to the native success
radius of 16, not calibrated guarantees. Failure stops this version before any
planning gate. Do not refit, change thresholds or select another split to pass.

## Pre-evaluation checks and first-decision gate

CPU checks use synthetic inputs and the actual hash-checked LeWM, CEM and native
TwoRoom source. Check unchanged baseline actions, distinct goals, one/two/three
frame contexts, an independently calculable one-frame rollout, probe split and
ridge correctness, clone preservation, clipping, episode pairing and gate order.
The kernel reruns synthetic objective-path checks against the installed library.

Then use only the first eight tasks of the official 100-task seed-42 draw for each
model and objective. Stop each gate worker immediately after its first complete
CEM decision and post-search diagnostic. No eight-episode success criterion is
used. These gates produce 48 chosen plans, **not** an extra performance pilot.

For each plan:

1. Require one real initial frame; keep official causal rollout/context handling.
2. Save normalized, submitted and effective controls. Apply TwoRoom's native
   clipping to [-1,1] after denormalization. Search scoring is unchanged apart
   from the terminal objective; effective-plan rescoring is diagnostic only.
3. Clone the native initialized environment by deep copy; compare initial rendered
   pixels and state with the original. Execute all 25 controls on the clone,
   ignoring termination for this diagnostic, with no reset. Verify the original
   rendered pixels and state were not changed.
4. Encode the terminal rendered image with official preprocessing and report the
   three position-error vectors:
   A = decoded actual observation minus actual position;
   B = decoded prediction minus decoded actual observation;
   C = decoded prediction minus actual position; verify A+B=C numerically.
5. Preserve solver RNG state during diagnostic rescoring. Archive plans and all
   vectors/costs, requiring finite values.

A tests observation decoding on these reached states. B and C mix properties of
prediction and decoding and do **not** identify an independent decoder error on
imagined latents. They are descriptive: never exclude a model, retune the decoder,
or condition proceeding on their magnitude. Their size limits later mechanistic
interpretation. A good offline R² does not establish reliable extrapolation.

Input, source, pairing, clone, context or finite-value failures stop the entire
job as implementation/validity failures. No scientific verdict follows.

## Full comparison and baseline gate

TwoRoom only: 3 frozen predictors × 2 objectives × seeds 43,44,45 × 100 paired
tasks = **18 evaluations, 1,800 episode outcomes**.

Keep goal offset 25; native control budget 50; horizon 5 blocks of 5 controls;
official receding horizon 5; CEM 300 candidates, 30 iterations, 30 elites; and
final elite mean. No decoder updates, predictor training, search clipping change
or selection change. Disable video capture only; keep native success metrics.
Log actual controls, decision counts and times since early success changes cost.

Run all nine latent-baseline evaluations first in the same software environment
as the interventions. Proceed to the nine position evaluations only if the
bilinear-minus-corrected success gap is >=5 percentage points on **each** seed.
Otherwise archive and stop because the premise was not reproduced.

Initialize CEM with the same prescribed seed per condition. Objectives can select
different elites immediately, so candidate pools are not asserted equal across
objectives. Episode identities, weights, search budgets and initial conditions
are paired; closed-loop trajectories can diverge.

## Primary criterion and interpretation

For each seed, let `G_latent = S(bilinear,latent) - S(corrected,latent)` and
`G_position = S(bilinear,position) - S(corrected,position)` in percentage points.
Define `I = G_latent - G_position`.

Primary material-effect criterion: after baseline reproduction, **I >=5 points
on all three seeds**. This carries over the preceding diagnostic's useful-effect
threshold; it is not a significance or power claim.

Report every model/objective success count, paired discordant counts and exact
two-sided McNemar tests. For I, report a descriptive paired 20,000-resample
percentile 95% bootstrap interval per seed, generator `23000 + evaluation_seed`.
Keep all four outcomes of a task together. No multiplicity-adjusted inference or
independence across reused evaluation seeds/training runs is claimed. The released
network is a secondary reference, not interchangeable with the corrected one.

- Passing criterion: objective choice contributes materially to the measured gap
  under these conditions; this does not establish an exclusive mechanism.
- Separate network improvement from bilinear deterioration using both absolute
  changes. A shrinking gap alone is not a neural improvement.
- A common near-ceiling score limits observability; do not claim equal capability.
- Valid negative outcome: this intervention does not support the prespecified
  material-effect claim; other objectives or interactions are not eliminated.
- Failed decoder gate, missing baseline or invalid run: no scientific verdict on
  the intended hypothesis.

Any follow-up including experiment 22 requires a new explicit decision about the
remaining alternatives. No automatic extension or post-result retuning.

## Artifacts and status

Retain input/cache/decoder/source manifests, splits, task lists, submitted source,
environment versions, runtime checks, first-plan diagnostics, per-episode outcomes,
executed controls, per-run resolved configs, paired analysis and failure records.
`complete.json` distinguishes decoder stop, baseline stop and all-18 completion.
`failure.json`, missing expected files or mismatched hashes invalidate completion.

Local tests cannot replace the runtime gates: the private checkpoints, cache and
Kaggle GPU environment have not been exercised during authoring. Freeze this file,
code, metadata and local check before submission; keep the submission version and
subsequent retrieval manifest with results.
