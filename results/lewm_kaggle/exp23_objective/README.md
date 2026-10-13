# Experiment 23: stopped at decoder feasibility gate

Kaggle `hassanakaou/lewm-objective`, version 1, reported COMPLETE when checked
on 2026-10-13. Its own `complete.json` reports `stopped_decoder_gate` and
`evaluations: 0`. The stop is the prespecified outcome of a failed decoder
feasibility requirement, not a completed predictor/objective comparison.

On 89,618 held-out frames, with 300,000 training frames and 388 planning episode
IDs excluded from both probe partitions:

| Metric | Observed | Prespecified requirement | Outcome |
| --- | ---: | ---: | --- |
| R², x | 0.995226 | >= 0.90 | Pass |
| R², y | 0.985823 | >= 0.90 | Pass |
| Median Euclidean error | 3.338951 | <= 4 native units | Pass |
| 95th percentile Euclidean error | 8.976610 | <= 8 native units | Fail |

The failed threshold was exceeded by 0.976610 native units. The strong global
R² and acceptable median do not remove this failed tail-error requirement.
The threshold remains unchanged; no further fit, planning gate, baseline or
intervention was run or authorized by this result.

The cache check on 64 training frames passed (maximum absolute embedding error
2.768636e-5 against a 1e-4 limit). Input/source provenance checks and the synthetic
runtime objective checks ran before fitting. This limited cache check is not
an exhaustive validation of the cache or imagined latent decoding.

## Interpretation and next decision

The specified affine decoder does not meet the complete feasibility criterion.
The effect of the decoded-position objective on the predictor success gap is
**untested**. Do not interpret this stop as evidence for or against that effect,
or as a comparison between the neural and bilinear predictors.

A useful next proposal is a separate audit of the held-out decoding errors,
including spatial concentration and the high-error tail. Any modified decoder
recipe or admissibility threshold would require a new documented protocol before
planning evaluation. Experiment 22 remains a separate draft; nothing is launched
automatically.

`version1/` preserves the retrieved raw outputs. `retrieval_v1.json` records file
hashes and the checks performed after download. Reported decoder errors come
from the frozen job; they were not recomputed from the full private cache locally.
`submission_v1/submission.json` retains the historical RUNNING observation and
the independently fetched source parity at submission time. GitHub publication
remains pending; see the pre-execution submission note in `preregistrations/`.
