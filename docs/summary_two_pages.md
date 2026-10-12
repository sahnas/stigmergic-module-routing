# A bilinear predictor on LeWorldModel's frozen encoder: what was measured, what was corrected, what is open

Hassan Akaou, 12 October 2026. Working summary of the public repository `sahnas/stigmergic-module-routing` (experiments 18 to 21 and their preregistrations). Two pages; every number below is in the repository's raw results.

## Question

That predictive accuracy does not guarantee control performance is established: Lambert, Amos, Yadan and Calandra (2020) named it objective mismatch and showed one-step likelihood is not always correlated with control. This summary reports one controlled instance on a released model, not the phenomenon. On LeWorldModel's frozen encoder, when does a predictor fitted by ridge least squares, linear in the embedding and bilinear in embedding and action, plan better than a neural predictor? The question arose from an error: a neural predictor we retrained reached eleven times lower latent error than the released one and planned far worse (36 % vs 85 % on Two-rooms). An external reader found the cause in our recipe, not in the model: our loss supervised only the last position of a three-frame context, the official loss supervises every position, and the planner starts from one frame. Retrained with the official loss, the network plans like the released predictor. That episode is the method result of this work; the bilinear predictor is what we tested next.

## Protocol

Released LeWorldModel encoder and projector, frozen (Two-rooms revision 77adaae0, PushT revision 22b330c2); fp32 caches of the embeddings of both datasets, parity with `model.encode` checked. Three predictor types per task, with task-specific weights, on that task's released encoder: the released predictor; a network with the released architecture retrained with the official loss (8 epochs on the cache, one training per task); a bilinear predictor fitted once by ridge least squares (lambda 1e-2 N, 2,507 coefficients per output dimension; the PushT fit took 131.5 seconds). Evaluation: the authors' `eval.py` unchanged (commit 8edfeb3), horizon 5 blocks of 5 actions, 100 episodes per evaluation seed, seeds 42 to 45, per-episode successes archived, exact McNemar tests on paired episodes. Seed 42 was exploratory on Two-rooms; seeds 43 to 45 were preregistered before any run, as was the PushT protocol and its primary criterion.

## Results

**Two-rooms (experiment 18).** Successes out of 100 on seeds 42, 43, 44, 45:

| predictor | held-out latent error | 42 | 43 | 44 | 45 |
|---|---|---|---|---|---|
| released | 0.067 | 85 | 83 | 87 | 80 |
| retrained, official loss | 0.005 | 84 | 85 | 89 | 81 |
| bilinear | 0.69 | 97 | 99 | 100 | 96 |

The bilinear predictor is above both networks on all four evaluation seeds, with one training per predictor; discordant episodes against the retrained network 15/2, 14/0, 11/0, 16/1 (exact two-sided McNemar p < 0.003 on each seed). Removing the bilinear state-action terms gives 57 % on seed 42; removing the history changes little (96 %); ridge x10 or /10 changes success by at most one percentage point on that seed (96 and 97 against 97). This is a repetition across evaluation seeds of one fit, not a measure of variability across trainings.

**PushT (experiment 19, preregistered).** The same recipe fails: bilinear 35, 40, 39, 36 against released 89, 86, 89, 85 and retrained 93, 94, 91, 89. The primary criterion (strictly more successes than the released predictor on each of seeds 43 to 45) fails on every seed; this is a collapse, not a tie at the ceiling. Descriptively, the retrained network scores above the released one on all four PushT seeds (91.75 % vs 87.25 % on average); on Two-rooms the two have similar observed success rates.

**Diagnostics completed and their scope.**
- Average latent error does not match the planning ranking on Two-rooms (it does on PushT).
- Error restricted to the task state, through a linear probe with held-out R2 above 0.9 on every coordinate (experiment 20, preregistered), matches the planning ranking on PushT but not on Two-rooms, where the retrained network has the lowest measured task-state prediction error (0.012 vs 0.048 for the bilinear predictor) but lower planning success than the bilinear predictor. This tested a measure, not a mechanism.
- The final selection of the cross-entropy solver (elite mean vs best final candidate; experiment 21, preregistered, 36 paired evaluations, 3,600 episodes) reduces the bilinear-minus-retrained gap by -1, 0 and +1 points on the three seeds, against a preregistered criterion of a reduction of at least 5 points on each seed. This weakens one precise explanation; it does not rule out other model-planner interactions.
- An earlier "exploitation" reading (a 27.5-fold optimism of the solver's chosen plan, measured on the defective retrained checkpoint) is no longer relied on. A later action-bounding control on the released, corrected and bilinear predictors showed that comparing predictions for unclipped actions with clipped executions substantially inflated apparent optimism (the corrected network's ratio fell from about 18.1 to 1.45 after clipping). This control did not re-evaluate the defective checkpoint behind the earlier 27.5-fold ratio.

## Corrections made along the way

The retraining recipe (last-position loss) and the sentence "the obvious causes are excluded" were both withdrawn publicly (le-wm issue 107, closed by us with the context-length table). A reimplementation of VIScore was found defective by review and is marked uninterpretable. All claims above were reduced to what the repository's numbers support after three rounds of external review.

## Where this sits

Predictors on frozen encoders are the standard family (DINO-WM; V-JEPA-2-AC). Terver, Yang, Ponce, Bardes and LeCun (2025) ablate predictor architecture, context and optimizer on that family and find capacity saturating on simulated tasks but helping on real robot data; "The Planning Limits of Latent World Models" (2026) finds an 81-fold larger predictor does not extend the useful horizon and locates the limit in the distance between target and imagined horizon. Neither shows a general advantage of small predictors, and we claim none. On LeWorldModel itself, Singh (2026) reproduced TwoRoom independently and then showed that the CEM objective, squared latent distance to the goal, tracks true distance weakly (r = 0.43), saturates and then decreases with distance, and that replacing only the objective, with encoder and predictor frozen, lifts long-offset success from 26 % to 98 %: "the most accurate predictor paired with the worst metric geometry is the worst long-horizon planner". Fu and Ning (2026) study how action effects accumulate into the terminal-cost landscape for sampling-based planners including CEM, on LeWorldModel among others; Nguyen, Xu and Huang (2026) replace CEM on LeWorldModel by an amortized inverse-dynamics controller. "Temporal Straightening for Latent Planning" (2026) relates straight latent trajectories to planning success and distinguishes, in its appendix, linear dynamics from straight trajectories, so our bilinear result does not by itself predict straighter trajectories; SJEPA (2026) proposes simple predictors on frozen encoders as a diagnostic of whether a representation exposes compact dynamics, with experiments on symbolic pendulum variants and visual control left to future work. Our comparison belongs to this family of questions; it does not establish any of these mechanisms. One candidate diagnostic we have not run follows from Singh's result: re-evaluate the three predictors under a proximity-faithful objective such as his decoded-position cost; if the bilinear advantage on Two-rooms disappears, it is an interaction between predictor and the Euclidean latent objective rather than predictive quality.

## Limits

Two tasks, one released encoder per task (shared by that task's three predictors), one fit per predictor, four evaluation seeds. Planning starts and goals are sampled from the full datasets, following the official evaluator, rather than restricted to episodes held out from predictor fitting; the prediction errors are measured on held-out episodes. The mechanism behind the Two-rooms advantage is unknown. Nothing is claimed about bilinear dynamics in general, about other encoders, or about what makes PushT different.

## The question we would ask

Which controlled diagnostic would you run next to distinguish explanations for why the bilinear predictor outperforms both neural baselines on Two-rooms but underperforms them on PushT, under these fixed recipes and task-specific released encoders? We would start with the objective (Singh, 2026) unless you see a more direct test.

Repository: https://github.com/sahnas/stigmergic-module-routing (preregistrations in `preregistrations/18` to `21`, results in `results/lewm_kaggle/`, kernels in `kaggle/`).
