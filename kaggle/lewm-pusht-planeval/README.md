# Experiment 19: official PushT planning evaluation

Twelve runs: released, corrected retrained and bilinear predictors; seeds
42–45; 100 starting tasks per run. The eight-epoch fit has completed and the
cache passed the independent official-preprocessing comparison.

The script loads the completed `lewm-pusht-fit` output and validates its
provenance, exact released checkpoint hashes and action normalization. It
downloads the preregistered raw dataset and checks out the pinned LeWM code.

The official `eval.py` and all configuration files are kept unchanged. Each
run executes that file in its own process. Hooks only:

- select the frozen released or retrained checkpoint, or install the fitted
  bilinear operator on the released encoder;
- assert the official solver settings and action normalization;
- save the selected episode/start/goal identities and the actual individual
  success flags returned by `World.evaluate`;
- route videos to a separate directory for each predictor and seed.

The CEM algorithm, action bounds, preprocessing, episode sampling, stopping
rules and success criterion are unchanged. The three predictors must receive
identical ordered tasks within each seed, or execution stops. Evaluation
uses the whole dataset as the official script does, not just held-out fit
episodes. These are evaluation seeds for one checkpoint of each model,
not four independently trained models.

Outputs include per-run configurations, logs, videos and `episodes.json`;
`paired_comparisons.json` with discordant counts and exact two-sided McNemar
p-values; and `evaluation_complete.json` only once all twelve runs finish.
Comparisons are saved after every completed seed. No result changes later
runs or any hyperparameter.

Primary verdict: strictly more bilinear successes than released successes
on each of seeds 43, 44 and 45. A tie fails this criterion. The retrained
comparison is secondary and separate. Seed 42 is exploratory. Count wins
are reported with effect sizes and p-values; a one-episode win is not
strong evidence by itself.

Local checks: `python check_local.py` with CPU PyTorch. GPU execution is
limited to 12 hours. The private kernel launch is authorized on 2026-10-08;
publication of subsequent repository changes is separate.

## Completed result, 2026-10-08

All twelve runs completed in 4,558 seconds including setup and data preparation.
Successes on evaluation seeds 42/43/44/45 (100 matched tasks per seed):
released 89/86/89/85; corrected retrained 93/94/91/89; bilinear 35/40/39/36.
The primary criterion fails on all three confirmatory seeds, as does the
separate comparison with the corrected network. The recipe is unchanged.

Raw JSON artifacts and the execution log are archived under
[`results/lewm_kaggle/exp19_eval`](../../results/lewm_kaggle/exp19_eval/).
`python tools/analyze_exp19.py` validates hashes, task identities, individual
outcomes, totals, paired statistics and generated manuscript tables.
