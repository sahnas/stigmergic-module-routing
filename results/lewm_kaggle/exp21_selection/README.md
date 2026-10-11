# Experiment 21 submission record

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

When outputs are available, verify the source and frozen-input manifests,
runtime smoke, task pairing, effective controls and completion status before
reporting the feasibility gate or the conditional full comparison.
