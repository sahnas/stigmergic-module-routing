# Experiment 19: cache validation and predictor fitting

This kernel follows `preregistrations/19_pusht_bilinear.md`. It does not perform
planning evaluation or select hyperparameters using results.

1. Mount the completed `hassanakaou/lewm-pusht-cache` output; check shape,
   finiteness and contiguous episode offsets.
2. Download the exact preregistered model and dataset revisions. Re-encode
   frames 1000–1063 through the pinned official `eval.py` image transform.
   Stop before fitting if maximum absolute cache error exceeds `1e-5`, a
   numerical tolerance fixed before this validation run.
3. Use the seed-0 episode split, full-dataset valid-row `StandardScaler`
   action statistics from `eval.py`, and windows contained within episodes.
4. Fit the experiment-18 bilinear features with ridge `0.01 * N`, including
   regularization of the bias as before. Report context-1/2/3 errors.
5. Train fresh predictor modules, seed 1, every position supervised,
   8 epochs, batch 256, AdamW 3e-4/weight decay 1e-3, OneCycle/pct_start .05,
   gradient clipping 1.0. Keep the encoder and projector frozen. The epoch-8
   checkpoint is mandatory; intermediate checkpoints are recovery artifacts.

The old Two-rooms retraining script used training-frame normalization; this
PushT script uses the full dataset's valid rows, as explicitly specified in
preregistration 19 and as the official evaluator does. No tuning is added.

Outputs include the released and retrained model folders, bilinear weights,
context errors, cache validation, exact package versions, hashes and progress.
`fit_complete.json` is only written after all 8 epochs and checkpoint reload
verification. Its absence means the models are not ready for the planned
12 evaluations. The private kernel is launched with a 12-hour limit.

The downloaded raw dataset stays on temporary storage, outside the output
quota. The 2.34-million-frame cache is reused, not recomputed.

Local checks: `python check_local.py` (NumPy and CPU PyTorch).
