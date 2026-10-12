# Experiment 21: final selection does not meet the explanation criterion

Kaggle version 2 completed at **2026-10-11 23:49:33 UTC**: six feasibility
runs, then all 36 planned evaluations (3,600 episode outcomes). The
[preregistered protocol](../../../preregistrations/21_cem_final_selection.md)
is unchanged. Three frozen predictors, two selectors, both tasks and
evaluation seeds 43–45 are included. No model was fitted or retrained.

The feasibility gate passed: mean and best effective plans differed by
more than 1e-3 normalized action units on **8/8** corrected-TwoRoom starts,
against the required 2/8. This checks that the intervention changes plans;
it does not establish an improvement in success.

## Successes out of 100 paired tasks

Each cell lists seeds **43 / 44 / 45**, in that order.

| Task | Predictor | Final elite mean | Best final elite |
|---|---|---|---|
| TwoRoom | Released network | 83 / 87 / 80 | 80 / 87 / 79 |
| TwoRoom | Corrected network | 85 / 89 / 81 | 84 / 89 / 81 |
| TwoRoom | Bilinear | 99 / 100 / 96 | 99 / 100 / 95 |
| PushT | Released network | 85 / 89 / 85 | 85 / 90 / 89 |
| PushT | Corrected network | 94 / 91 / 89 | 94 / 91 / 88 |
| PushT | Bilinear | 41 / 37 / 35 | 41 / 41 / 38 |

The TwoRoom mean-selector baseline gap reproduces on every seed. Define
the interaction as the bilinear-minus-corrected gap under mean selection,
minus that gap under best selection. Positive values mean that the new
selector reduces the bilinear advantage.

| TwoRoom seed | Mean gap, points | Best gap, points | Gap reduction, points | Paired bootstrap 95% interval |
|---|---|---|---|---|
| 43 | 14 | 15 | −1 | [−5, 2] |
| 44 | 11 | 11 | 0 | [−4, 4] |
| 45 | 15 | 14 | +1 | [−2, 5] |

The primary criterion required a reduction of at least **5 points on each
seed**. It is not met on any seed. Returning the best final candidate does
not support this prespecified explanation at the chosen effect size. It
does not improve the corrected network's observed success on TwoRoom.
This is not proof that all planner mechanisms are irrelevant, or that the
two selectors are universally equivalent. The cause of the remaining
TwoRoom advantage is still unknown.

On the prespecified secondary task, PushT, the bilinear predictor remains
below both networks with either selector. This is a fresh paired evaluation
in its archived environment, not a bit-for-bit historical replay of
experiment 19. It makes no claim about newly learned abstractions or
general superiority of bilinear dynamics.

Exact paired McNemar tests, discordant counts and the registered bootstrap
are in [`version2/analysis.json`](version2/analysis.json). Intervals are
descriptive, within evaluation seed; neither the three reused seeds nor
their episodes establish generalization over independently trained models.

## Reproduction and execution audit

```sh
python tools/verify_exp21.py
python kaggle/lewm-cem-selection/selection.py --analyze results/lewm_kaggle/exp21_selection/version2
```

[`version2/verification.json`](version2/verification.json) records the
episode pairing, frozen-source/input checks, initial-search comparison,
final elite membership, native action mapping, executed controls and
recalculated analysis. It also reports actual solve calls, per-row decisions,
search/diagnostic candidates, control steps and elapsed times for every run.
The comparison concerns equal search budget per decision; early successes
and diverging paths can change total episode compute.

Raw episode outcomes, plans, controls, configurations and the complete
kernel log are retained in [`version2/`](version2/). Videos and redundant
per-run text/Hydra logs are not needed by the audit; the original output
listing and download manifest record the selection. The Hugging Face Hub
dependency warnings did not prevent the installed-library check or any
of the 36 evaluations from completing.

## Submission history

The protocol and source were pushed before the accepted Kaggle version 1.
The script pulled back from Kaggle matches the committed bytes. CPU plumbing
checks are in [`kaggle/lewm-cem-selection/plumbing_check.json`](../../../kaggle/lewm-cem-selection/plumbing_check.json).

`launch.json` is a dated submission/status observation, not a live status
or an experiment outcome.

Version 1 ended in `ERROR` at 2026-10-11 18:40:21 UTC. The source, runtime
check, frozen-input manifests, environment, sampled tasks and raw log are
archived in [`version1/`](version1/), with a SHA-256 retrieval manifest.
The first post-search diagnostic grouped eight rows instead of using the
official batch size of one and failed in LeWM's goal-embedding broadcast.
There is no completed episode evaluation, gate verdict or full comparison.
The correction only batches the extra diagnostic by individual row; the
preregistered intervention and criteria are unchanged.

Kaggle version 2 was accepted from the pushed correction commit. Its source
was pulled back and matches the committed bytes exactly. The original
`RUNNING` observation in [`launch_v2.json`](launch_v2.json) is preserved as
dated history. The completed outputs and verification above supersede
that launch status.
