# Experiment 23: shared decoded-position objective

Changes only LeWM's terminal cost on TwoRoom, using a single fitted decoder shared
by three **existing frozen** predictors. Protocol:
[`23_decoded_position_objective.md`](../../preregistrations/23_decoded_position_objective.md).

The script checks input provenance and fits one decoder, stops if its observation
validation fails, then captures 48 first-decision plans on 8 fixed starts. It runs
9 latent baselines before conditionally running 9 decoded-position comparisons.
Nothing trains or refits a dynamics model. Experiment 22 is not executed.

## Checks

Use Python 3.12 with NumPy, Torch, Gymnasium, einops and loguru. Installing the
`stable-worldmodel==0.1.1` wheel with `--no-deps` suffices for source-based checks;
the full package is installed by the Kaggle runner for real execution.

```sh
python check_local.py --package /path/to/site-packages/stable_worldmodel
python objective.py --check /path/to/site-packages/stable_worldmodel
```

Checks import the hash-verified native model/solver/environment source with only
package-root imports replaced, as in experiment 21. They use synthetic values,
not private weights or research observations. Full GPU operation is checked on
Kaggle before the paired comparisons.

## Submit the committed version

From an authenticated Kaggle environment, after verifying the freeze:

```sh
python tools/verify_exp23_freeze.py
kaggle kernels push -p kaggle/lewm-objective
kaggle kernels status hassanakaou/lewm-objective
```

The kernel metadata is private, enables a T4 and attaches four existing kernels:
the corrected TwoRoom checkpoint, bilinear checkpoint, dataset and fp32 cache.
The standalone script downloads only frozen source helpers/task fixtures from
GitHub and official runtime packages. Missing or changed private inputs stop it.

Record the submitted Kaggle version before retrieving outputs. A successful
submission is not a successful experiment. Use a fresh output directory on every
run. There is no silent resume or automatic retuning after a failed gate.

## Retrieve and verify

```sh
kaggle kernels output hassanakaou/lewm-objective -p /path/to/outputs
python objective.py --analyze /path/to/outputs
```

`failure.json` means execution failed. `complete.json` may instead report
`stopped_decoder_gate`, `stopped_baseline_not_reproduced` or `evaluated`.
An `evaluated` marker is credible only with all 18 × 100 paired outcomes, source
and input hashes, successful gates and matching regenerated analysis. Do not
interpret decoder or implementation failures as negative scientific results.

## Current status

CPU checks pass; no private-data fit, GPU gate or research evaluation has run.
The notebook has not been submitted at the time this implementation was committed.
