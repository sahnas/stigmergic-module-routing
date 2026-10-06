# Preregistration: experiment 18, an operator predictor on LeWorldModel's frozen encoder, confirmation

Written 2026-10-06T22:39Z, before any of the runs below.
Exploratory run already done (kaggle/lewm-koopman/koopman.py, sha256 3546c74ba8c7198f): a predictor
linear in the last three embeddings and in the 10-d action block, bilinear in (current embedding x action block), with
a bias, fitted by ridge least squares (lambda = 1e-2 x N) on the training windows of the fp32 cache (dataset action
statistics, as eval.py); relative one-step latent error 0.69 on held-out windows; with the authors' eval.py unchanged,
encoder and projector released, success 97/100 on evaluation seed 42 (released predictor: 85/100 on the same seed).
Confirmation code: kaggle/lewm-koopman-confirm/confirm.py (sha256 c64207c70fb7d480).

Hypotheses and priors, written before the runs:
- K1 (fresh evaluation seeds 43, 44, 45, 100 episodes each): the bilinear operator's success is at or above the
  released predictor's on all three seeds. Prior 60 %.
- K2 (exploitation, separate kernel reusing kaggle/lewm-exploit with the operator swapped in): on 30 cloned starts,
  the ratio of true to predicted cost of the plan chosen by the authors' CEM is below 3 for the operator (released
  1.5, retrained 27). Prior 70 %.
- K3 (ablations, seed 42): the purely linear variant (no bilinear term) and the Markov variant (current embedding
  only) both score below the bilinear operator; the bilinear term is the active ingredient. Prior 55 %.
- K4 (regularisation, seed 42): ridge x10 and ridge /10 both stay within 10 points of 97. Prior 65 %.
Reading fixed in advance: if K1 fails, the 97 is a single-seed artefact and is reported as such; if K1 holds and K2
holds, the claim is that a predictor with far higher latent error but a smooth operator structure plans at least as
well as the end-to-end predictor on this task, and the manuscript's planning-gap section is rewritten around it;
K3 and K4 only qualify that claim. The claim is for Two-rooms and this encoder; nothing is claimed for other tasks.
