# A bilinear predictor on LeWorldModel's frozen encoder: what was measured, what was corrected, what is open

Hassan Akaou, 12 October 2026. Working summary of the public repository `sahnas/stigmergic-module-routing` (experiments 18 to 21 and their preregistrations). Two pages; every number below is in the repository's raw results.

## Question

On LeWorldModel's frozen encoder, when does a predictor fitted by ridge least squares, linear in the embedding and bilinear in embedding and action, plan better than a neural predictor? The question arose from an error: a neural predictor we retrained reached eleven times lower latent error than the released one and planned far worse (36 % vs 85 % on Two-rooms). An external reader found the cause in our recipe, not in the model: our loss supervised only the last position of a three-frame context, the official loss supervises every position, and the planner starts from one frame. Retrained with the official loss, the network plans like the released predictor. That episode is the method result of this work; the bilinear predictor is what we tested next.

## Protocol

Released LeWorldModel encoder and projector, frozen (Two-rooms revision 77adaae0, PushT revision 22b330c2); fp32 caches of the embeddings of both datasets, parity with `model.encode` checked. Three predictors per task on the same encoder: the released predictor; a network with the released architecture retrained with the official loss (8 epochs on the cache, one training per task); a bilinear predictor fitted once by ridge least squares (lambda 1e-2 N, 2,507 coefficients per output dimension, one minute). Evaluation: the authors' `eval.py` unchanged (commit 8edfeb3), horizon 5 blocks of 5 actions, 100 episodes per evaluation seed, seeds 42 to 45, per-episode successes archived, exact McNemar tests on paired episodes. Seed 42 was exploratory on Two-rooms; seeds 43 to 45 were preregistered before any run, as was the PushT protocol and its primary criterion.

## Results

**Two-rooms (experiment 18).** Successes out of 100 on seeds 42, 43, 44, 45:

| predictor | held-out latent error | 42 | 43 | 44 | 45 |
|---|---|---|---|---|---|
| released | 0.067 | 85 | 83 | 87 | 80 |
| retrained, official loss | 0.005 | 84 | 85 | 89 | 81 |
| bilinear | 0.69 | 97 | 99 | 100 | 96 |

The bilinear predictor is above both networks on all four evaluation seeds, with one training per predictor; discordant episodes against the retrained network 15/2, 14/0, 11/0, 16/1 (p <= 0.002 per seed). Removing the bilinear state-action terms gives 57 % on seed 42; removing the history changes little (96 %); ridge x10 or /10 changes nothing. This is a repetition across evaluation seeds of one fit, not a measure of variability across trainings.

**PushT (experiment 19, preregistered).** The same recipe fails: bilinear 35, 40, 39, 36 against released 89, 86, 89, 85 and retrained 93, 94, 91, 89. The primary criterion (strictly more successes than the released predictor on each of seeds 43 to 45) fails on every seed; this is a collapse, not a tie at the ceiling. Descriptively, the retrained network scores above the released one on all four PushT seeds (87.25 vs 91.75 on average); on Two-rooms the two are indistinguishable.

**What does not explain the Two-rooms advantage.**
- Average latent error does not rank the three predictors on both tasks (reversed on Two-rooms, aligned on PushT).
- Error restricted to the task state, through a linear probe with held-out R2 above 0.9 on every coordinate (experiment 20, preregistered), ranks them on PushT but not on Two-rooms, where the retrained network predicts the agent's position best (0.012 vs 0.048 for the bilinear predictor) and plans worst. This tested a measure, not a mechanism.
- The final selection of the cross-entropy solver (elite mean vs best final candidate; experiment 21, preregistered, 36 paired evaluations, 3,600 episodes) changes the bilinear-minus-retrained gap by -1, 0 and +1 points on the three seeds, against a 5-point criterion. This weakens one precise explanation; it does not rule out other model-planner interactions.
- An earlier "exploitation" reading (27-fold optimism of the solver's chosen plan) was mostly an artefact of comparing the predicted cost of unclipped actions with the executed clipped plan; after clipping, ratios are 0.8 to 1.5 for all three predictors.

## Corrections made along the way

The retraining recipe (last-position loss) and the sentence "the obvious causes are excluded" were both withdrawn publicly (le-wm issue 107, closed by us with the context-length table). A reimplementation of VIScore was found defective by review and is marked uninterpretable. All claims above were reduced to what the repository's numbers support after three rounds of external review.

## Limits

Two tasks, one encoder, one fit per predictor, four evaluation seeds. The mechanism behind the Two-rooms advantage is unknown. Nothing is claimed about bilinear dynamics in general, about other encoders, or about what makes PushT different.

## The question we would ask

Which diagnostic would you run to understand a ranking that reverses between Two-rooms and PushT for the same three predictors on the same encoder family, given that neither average latent error, task-state error nor the solver's final selection accounts for it?

Repository: https://github.com/sahnas/stigmergic-module-routing (preregistrations in `preregistrations/18` to `21`, results in `results/lewm_kaggle/`, kernels in `kaggle/`).
