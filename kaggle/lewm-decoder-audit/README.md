# Frozen-decoder error audit after experiment 23 version 1

Exploratory analysis authorized after observing the failed p95 criterion; not
a new confirmatory preregistration. No decoder fit, predictor evaluation, change
to experiment 23's thresholds or planning run. Uses CPU only.

Pin the original decoder, split and manifests by byte hash. Pin cache and dataset
to their experiment-23-v1 hashes. Recompute the frozen validation errors and stop
if median/p95 differ by more than 2e-4 native units or R² differs by more than 2e-6.
Training errors use exactly the original 300,000 selected rows as a descriptive
reference; validation uses all original 89,618 rows. No new model is fitted.

Before inspecting the per-frame errors, fix these descriptive outputs:

- Full error quantiles, signed coordinate biases, rates above 4/8/16 units.
- Per-episode errors and concentration of failures in the top 1/5/10 episodes,
  alongside each group's share of all frames. No independent-frame inference.
- A fixed 16×16 spatial grid (14-unit cells), occupancy and mean residual vectors;
  omit error-map cells with fewer than 30 observations but retain them in CSV.
- Fixed canvas-edge, half-plane, central-x and optional door-center groups.
  The central-x group is not assumed to represent the actual wall geometry.
- Signed error versus x/y modulo 14, matching the encoder patch width, explicitly
  descriptive and potentially confounded by spatial occupancy.
- Images: worst frame from each of 16 distinct episodes; worst frame in each
  occupied 4×4 spatial bin; 16 random tail frames (error >8, RNG 23023); 16 random
  validation frames (RNG 23024). Raw images and exact selection metadata retained.
- Red-excess image centroid versus the stored position as a sampled label/rendering
  sanity check, conditional on the images actually depicting a red agent.
- Complete error traces for the eight episodes containing the largest distinct-
  episode errors. Deliberately chosen extremes are not representative samples.

Audit results guide a later protocol decision; no automatic retuning or second
planning evaluation follows. GitHub publication remains pending. Record this
implementation and its hashes in a local commit before submission, then archive
the submitted Kaggle version, retrieved source parity and output hashes.

Local checks: `python audit.py --check`.
Submission: `kaggle kernels push -p kaggle/lewm-decoder-audit`.
