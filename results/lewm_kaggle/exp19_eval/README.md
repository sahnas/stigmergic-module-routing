# Experiment 19: completed PushT planning comparison

The fixed recipe was preregistered in
[`19_pusht_bilinear.md`](../../../preregistrations/19_pusht_bilinear.md).
All twelve evaluations completed on 2026-10-08 at 04:28 UTC. Kaggle execution:
`hassanakaou/lewm-pusht-planeval`, version 1. The retrieved latest output is
identified by its exact script hash in `evaluation_manifest.json`; the
retrieval API did not resolve a version-specific output request.

| Evaluation seed | Released | Corrected neural | Bilinear |
|---|---:|---:|---:|
| 42 (exploratory) | 89 | 93 | 35 |
| 43 | 86 | 94 | 40 |
| 44 | 89 | 91 | 39 |
| 45 | 85 | 89 | 36 |
| Descriptive mean (%) | 87.25 | 91.75 | 37.50 |

Each cell is successes out of 100 identical ordered tasks within a seed.
The primary criterion, a strict bilinear win over the released model on
each of seeds 43, 44 and 45, fails on all three. The separate secondary
comparison with the corrected neural predictor fails on all three as well.
The fixed-recipe series ends with a Two-rooms gain and a PushT disadvantage.
No mechanism for this difference is established.

`episodes.json` in each run folder records the task identity and success
flag, not just a rounded aggregate. Companion files preserve task selection,
the evaluator command, metrics and solver/normalization settings.
`paired_comparisons.json` preserves the discordant counts and exact two-sided
McNemar tests. `evaluation_complete.json` records completion and both
preregistered verdicts. `fit_*.json` preserve the input fit's measurements
and provenance. `execution.log` is the retrieved Kaggle event log.

`retrieval_manifest.json` contains SHA-256 hashes of the locally serialized
JSON downloads. `summary.json` and `paper/exp19_tables.tex` are derived files:

```sh
python tools/analyze_exp19.py
python tools/analyze_exp19.py --write  # regenerate, then review the diff
python paper/check_numbers.py
```

The validator recomputes scores and paired tests from all 1,200 outcome
records, checks within-seed task identity, the recorded solver and action
statistics, frozen model/code revisions, the fit and evaluation source
hashes, and the download hashes. These checks establish internal consistency;
they are not an independent rerun of the simulator.

Scope: one fitted predictor per type; evaluation seeds rather than training
replications; official planning tasks from the full dataset, including fit
episodes; latent errors measured on the episode-level holdout; no modification
of action bounding. The neural-over-released comparison is descriptive, not
a replacement primary hypothesis. Neither a cause of the task difference nor
a general rejection of other bilinear recipes follows from this result.

The large cached embeddings, model weights and videos are not committed here.
Their input revisions and weight hashes are recorded; regenerating the
predictors requires the cache and fit scripts under `kaggle/` and the pinned
public Hugging Face dataset/model. Kaggle kernels remain private.
