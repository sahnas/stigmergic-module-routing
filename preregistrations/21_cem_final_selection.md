# Experiment 21: does CEM's final averaging contribute to the bilinear TwoRoom advantage?

Written 2026-10-11, before any research episode for this intervention.
The implementation and this protocol will be committed and pushed before the
Kaggle kernel is submitted. CPU plumbing checks use synthetic costs only.

## Question and scope

Three frozen predictors share an encoder within each task. On the existing
evaluation the bilinear predictor wins on TwoRoom and loses on PushT. This
diagnostic asks whether the **final plan selection** contributes to the
TwoRoom advantage, rather than claiming a better learned representation.

The pinned CEM implementation generates 300 candidates per iteration, scores
them with the selected model, keeps 30 elites, and performs 30 iterations.
It returns the final elite mean as its actions. The intervention returns the
lowest-cost candidate in the last elite set instead. All candidate sampling,
elite fitting, objectives, iterations and model weights remain unchanged.
The final mean is not generally one of the last scored candidates.

Mean-versus-best CEM output is a classical choice, explicitly described in
Walker et al., *Sample-Efficient and Smooth Cross-Entropy Method Model
Predictive Control Using Deterministic Samples*:
https://isas.iar.kit.edu/pdf/arXiv25_Walker.pdf . The general distinction
between predictive and decision quality is also studied in
https://arxiv.org/abs/2610.05550 . This protocol claims neither an original
optimizer nor that those papers leave this exact diagnostic untouched.
Its possible contribution is a controlled explanation of our fixed result.

## Fixed inputs

- Upstream `lucas-maes/le-wm`: `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
  Unedited `eval.py` SHA-256:
  `6584f009b40550fa0df2f8dee22b9329fbfacdfaf74464aa4fd105f464795245`.
- `stable-worldmodel[train,format]==0.1.1`; the solver, policy and both
  environment source hashes in `selection.py` are asserted before execution.
  Dependencies are pinned where the earlier evaluation explicitly fixed them;
  the actual complete environment is archived. This is a fresh paired
  evaluation, not a claim of a bit-for-bit historical environment replay.
- Existing frozen weights only, attached through `lewm-retrain-v2`,
  `lewm-koopman-confirm`, `lewm-data-tworoom` and `lewm-pusht-fit`.
  Duplicate paths are accepted only if their bytes agree. Encoder and
  projector parameters must be exactly equal between the two networks.
  The bilinear model replaces the released predictor stack and inherits
  that same encoder and projector.
- TwoRoom released model originally from `quentinll/lewm-tworooms`, revision
  `77adaae0bc31deab21c93740d1f8bb947cd0bdec`. Use the archived evaluation
  checkpoint, not another download of `main`.
- PushT weights from experiment 19, model revision
  `22b330c28c27ead4bfd1888615af1340e3fe9052`; dataset archive revision
  `655cd446b992`, `pusht_expert_train.h5.zst`. TwoRoom dataset is the same
  `lewm-data-tworoom/tworoom.h5`. Dataset hashes are logged before evaluations.

| Task | Predictor | Weights SHA-256 |
|---|---|---|
| TwoRoom | released | `566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd` |
| TwoRoom | corrected network | `df94cb98551b9ed0b5ca54fb7477b9fd305faf277261fac04f8ba321f440a132` |
| TwoRoom | bilinear | `611f430358576046c1b6cbdbd52e94b9e6577e28872b9c02812f64c5d857d23f` |
| PushT | released | `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607` |
| PushT | corrected network | `9224779215c068aa4b7c68260fb0380e4e68958e347ec8d0cb08d1a6a55e5f23` |
| PushT | bilinear | `fa06ecffd4f73ccd8ffa64e851dac38e9cf5f2f042b72c5998c4dc9a7ea8283e` |

## Plumbing checks before research data

The CPU check executes the pinned solver core with synthetic costs. The mean
hook must return the original actions bit for bit. Both modes must use
identical candidate tensors, costs, variances and final solver RNG state;
the best output must belong to the last scored elite set. Callback clearing
is exercised across batches and successive decisions. Bilinear features and
causal short contexts are compared with an independent direct expression.
Episode pairing and the interaction analysis are checked on synthetic data.

An additional small check runs against the actual installed library on
Kaggle before any dataset evaluation. An input, source, pairing or plumbing
failure stops the job; it is an implementation failure, not a negative
scientific result.

## Feasibility gate

For each task and each model, use evaluation seed 42 and the first eight
tasks from the official **100-task** sample (do not resample eight tasks).
Run only the mean policy and capture both possible outputs of its first
planning decision, before different choices could change the observations.
The extra `world.num_envs=8` override serves this gate only.

Measure the maximum absolute difference between mean and best over their
25 native controls, in dataset-standardized action units **after the native
execution mapping**. In this pinned release TwoRoom internally clamps
actions to [-1,1]; PushT does not clamp its submitted relative controls.
Do not add a clamp to PushT. Record submitted and effective controls and
re-predict the cost of both effective plans without changing search scoring.

Proceed to the full comparison iff the corrected TwoRoom network differs
by more than `1e-3` normalized units on at least **two of the eight** starts.
This threshold checks that the intervention is non-negligible; it is not
evidence of a success gain. If it fails, archive the gate and stop, without
trying another threshold, seed, iteration count or selector. A failed small
gate does not prove mean and best are universally equivalent.

## Full comparison, conditional on that gate

- Both tasks, three frozen predictors, mean and best, seeds 43, 44, 45,
  100 paired starting tasks per condition: **36 evaluations**, 3,600 episode
  outcomes. These are the existing evaluation seeds, not new training seeds
  or newly held-out data. Intervention outcomes have not been inspected.
- Keep the official task sampling, goal distance of 25 steps, 50 native
  control-step budget, horizon 5 blocks, action block 5, receding horizon 5,
  real context handling, action normalization and native success metric.
- Start each run's CEM RNG at the same prescribed seed. From a common state
  the search is identical until output selection. After execution different
  observations and terminations can naturally produce different searches
  and RNG consumption. Do not claim globally identical pools throughout
  closed-loop trajectories.
- Budget: 9,000 search candidates per planning decision in both modes.
  Four additional candidates (unbounded mean/best and effective mean/best)
  are scored **after** search, for logging only, in both conditions. Report
  actual decision counts, control steps and times, since early success
  changes total episode compute. No learning and no tuned parameter.
- Capture episode identities/successes, final elite plans/costs, both
  candidate outputs, their predicted costs before and after execution
  mapping, and the actual submitted/effective controls.

## Primary criterion and interpretation

Let `S(model, selector, seed)` be the success percentage out of 100 tasks.
Define, for each seed:

`G_mean = S(bilinear, mean) - S(corrected, mean)`

`G_best = S(bilinear, best) - S(corrected, best)`

`I = G_mean - G_best`.

First require `G_mean >= 5` percentage points on each of the three TwoRoom
seeds. If this baseline gap is not reproduced, the intended causal question
is inconclusive for this run and the discrepancy must be investigated.

The primary material-effect criterion is `I >= 5` percentage points on **each**
of those seeds. This is a deliberately limited decision rule, not a claim
of statistical significance. Report all model/selector successes, their
paired discordant counts and exact two-sided McNemar tests. Report a paired
20,000-resample percentile 95% bootstrap interval for `I` within each seed,
keeping all four outcomes of a starting task together; generator seed
`21000 + evaluation_seed`. Intervals are descriptive, no multiplicity or
independence claim is made across reused evaluation seeds.

The released model is a separate reference; do not transfer the primary
verdict to it. PushT is a prespecified secondary boundary check, reported
with the same quantities without transferring its verdict to TwoRoom.

- If the primary criterion holds, final selection makes a material
  contribution to the relative performance in this configuration.
  Distinguish a network improvement from a bilinear degradation using
  the separate success counts. Neither establishes an exclusive mechanism.
- If the baseline replicates but the criterion fails, this prespecified
  final-selection explanation is not supported at the chosen effect size.
  Do not retune this experiment or infer that every planner mechanism has
  been eliminated.
- No outcome demonstrates abstraction discovery, a new optimizer,
  general superiority of bilinear dynamics or performance on other tasks.

Subjective prior that the primary criterion holds, conditional on passing
the feasibility gate and reproducing the baseline: **30%**. This is an
explicit bet, not a calibrated probability. The narrow intervention may
have little effect after 30 CEM contraction steps; its actual spread is
not yet measured. No broad architecture change is justified beforehand.
