# Preregistration: experiment 16, reset or keep the weights of a recruited module

Written 2026-10-02T22:40Z, before any run on the test seeds.
Code: src/exp_reset.py (sha256 2e4c396f887e23eb), src/exp_preservation.py (sha256 b112fd314196a825), src/exp_dynamic.py (sha256 c3329b81c02b0eb6), src/exp.py (sha256 6c202cab844bb559).

Question (from a reviewer's remark after experiment 15): since a module trained for one function is a worse
starting point for another than a fresh module, does reinitialising a recruited module (parameters and optimizer
state) improve the system at equal budget? Two settings: A, the full production scenario of experiment 3
(availability over 4 events); B, experiment 14's learned-then-rare scenario with no reserve. Rules: keep (current)
and reset. Nothing else changes; nothing is tuned.

Development seed 953, declared before the test: reset was worse in both settings (A: availability 0.64 vs 0.73,
final R2 0.54 vs 0.72; B: availability -0.03 vs 0.36, other-task R2 0.43 vs 0.94). Likely reason: a recruited
module keeps serving its previous symbols through the traces, so resetting it destroys functions that would have
survived; the negative transfer of experiment 15 is paid, but losing retained functions costs more.
Prior before writing the code: 30 to 60 % for a gain from reset. After the development seed: 15 %.

Note: the first launch crashed on a print statement after writing one row (setting A, keep, seed 1300); the print line was fixed, the row deleted, and the hash above is that of the fixed file. The run functions are unchanged.

Test seeds: 1300 to 1309 (10 seeds, never used; checked against every result file).
Hypotheses, two-sided paired Wilcoxon, threshold 0.05 / 2:
- T1 (setting A): availability, reset minus keep. Expected negative.
- T2 (setting B): availability, reset minus keep. Expected negative.
Descriptive: per-event availability in A, final R2, other-task R2 after in B, alarms.
Reading fixed in advance: if reset is worse, the cost of recruitment is not the starting point of the recruited
module but the loss of what it still carried, which points to protecting modules in use rather than refreshing
them; if reset is better, negative transfer dominates and recruitment should reinitialise.

## Abandoned after 4 of 10 seeds (2026-10-04)

An external review of the repository pointed out that this design applies the reset rule from the start of
learning, where alarms also fire, so the two rules diverge before the event and the comparison does not isolate
the effect of resetting a recruited module. The 4 completed seeds are kept in results/16_reset/reset.jsonl and
are not analysed. The question is re-posed as experiment 16b: a common learned state is duplicated at the first
post-event recruitment and three interventions are compared on identical data streams (keep weights and optimizer
state; keep weights, reset optimizer state; reset both).
