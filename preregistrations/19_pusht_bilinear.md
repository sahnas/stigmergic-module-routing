# Preregistration: experiment 19, the bilinear predictor on PushT

Written 2026-10-07T18:25Z, before the embedding cache, any fit, or any evaluation on PushT.

## Question
Does the gain of the ridge-fitted bilinear predictor over LeWorldModel's released predictor, observed on Two-rooms
(experiment 18), recur on PushT, the second task of the release? The mechanism is not under test and will remain
to be established whatever the outcome.

## Fixed inputs (revisions recorded now)
- Released PushT model: Hugging Face `quentinll/lewm-pusht`, revision 22b330c28c27ead4bfd1888615af1340e3fe9052
  (encoder, projector, action encoder with input_dim 10 and emb_dim 192, predictor with num_frames 3 and
  192-dimensional embeddings, pred_proj). Two-rooms model for reference: revision 77adaae0bc31deab21c93740d1f8bb947cd0bdec.
- Evaluation code: lucas-maes/le-wm at commit 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac, `config/eval/pusht.yaml`
  unchanged: horizon 5, receding horizon 5, action block 5 (frameskip 5 comes from this YAML, not from config.json),
  eval budget 50, goal offset 25, environment state set from the dataset row; `eval.num_eval` overridden to 100 as on
  Two-rooms. Library: stable-worldmodel 0.1.1 from PyPI; the environment installs from `[train,format]` plus the
  packages PushT needs, recorded in the kernel.
- Dataset: `pusht_expert_train.h5.zst` from the Hugging Face dataset `quentinll/lewm-pusht`, revision 655cd446b992,
  decompressed to `$STABLEWM_HOME/datasets/`; its size and number of episodes are recorded in the cache metadata.
- Published reference for context: the paper reports 96.0 +/- 2.83 % on PushT for three training seeds on 50
  trajectories, a different protocol from ours (four evaluation seeds, 100 episodes, one checkpoint); the margin
  above the released predictor is therefore expected to be small.

## Procedure, identical to experiment 18 unless stated
1. fp32 cache of the frozen encoder's embeddings for every frame, parity with `model.encode` checked on 64 frames,
   with PushT's own preprocessing from the YAML transform.
2. Windows: history 3, frameskip 5, action blocks of 5 (10-dimensional, PushT's action is 2-dimensional), target 5
   frames ahead, never across an episode boundary; episode-level held-out split of 10 % with generator seed 0 (as on
   Two-rooms); action normalisation by the dataset's mean and standard deviation, as `eval.py` does.
3. Bilinear predictor: linear in the three embeddings and the action block, bilinear in current embedding x action
   block, with a bias; ridge lambda = 1e-2 x N on the training windows; no tuning. Held-out relative one-step error
   reported at contexts of 1, 2 and 3 frames.
4. Corrected neural predictor: the released architecture at random init, the official loss (every position
   supervised), 8 epochs, OneCycle 3e-4, batch 256, training seed 1 (as the Two-rooms variant); errors at contexts
   1, 2, 3 reported.
5. Official evaluation, unchanged, seeds 42, 43, 44, 45, 100 episodes each, for the released predictor, the
   corrected neural predictor and the bilinear predictor; per-episode successes archived.

## Hypothesis and reading, fixed in advance
- Primary criterion: on each of seeds 43, 44 and 45, the bilinear predictor has strictly more successes than the
  released predictor. A tie on any seed fails the criterion. Differences are reported with discordant-episode counts
  and exact McNemar p-values; winning by one episode on a seed is not, by itself, strong statistical evidence, and
  will be described as such.
- Secondary comparison: the corrected neural predictor. Beating the released predictor does not license the claim
  of beating both networks; that claim requires the same criterion against the corrected network.
- Seed 42 is reported as exploratory, as on Two-rooms.
- Reading of a failure: "superiority not confirmed on PushT with this recipe". A tie near 100 % and a collapse are
  different outcomes and will be reported as such, with the error levels.
- No change to the action bounding during the search; if tested later, it will be a separate condition.
- Subjective prior that the primary criterion holds: 40 %. Motivation, intuitive and not measured: PushT is
  contact manipulation, and whether its dynamics is as close to bilinear in this embedding as Two-rooms' has not
  been measured.
