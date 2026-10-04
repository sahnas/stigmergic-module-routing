# Preregistration: experiments 14b (preservation, corrected) and 16b (reset at recruitment), common-state harness

Written 2026-10-04T16:12Z, before any run on the test seeds.
Code: src/exp_common_state.py (sha256 8e962faa971c8297), src/exp_dynamic.py (sha256 c3329b81c02b0eb6), src/exp.py (sha256 6c202cab844bb559).

Harness (answers the external review of 2026-10-04): per seed and capacity, one model learned up to the event
(uniform learning, death of the module carrying frequent primitive a), deep-copied before every condition
(weights, optimizer state, traces, alarm statistics, counters, routing generator); production data (task order
with the rare primitive r weighted 0.02, and all batches) drawn once per seed and shared; recruitments
timestamped so that "first recruit" is the first after the event. Capacity: none (5 active), spare (7 active
from the start, over-provisioning), dormant (5 active + 2 inactive held back until the event, then activated
first on recruitment). Measures on tasks needing r but not a (rare), neither (other), a (killed); availability
over production; recruitments after the event.

Development seed 954, declared before the test. Before-event values are identical across rules within a
capacity (harness check). Capacity none: every rule degrades (availability 0.34 least committed, 0.20 random,
0.44 none); least committed never recovered the killed function (0.01) while random did (0.88) at the price of
the rare skill (-1.18). Capacity dormant: first recruit dormant, availability 0.68, rare 0.66; capacity spare:
availability 0.92, rare 0.95; spares active from the start absorb better than fresh dormant modules on this
seed. 16b (capacity none): reset_opt 0.35, reset_both 0.33, keep 0.34 on availability; no visible effect.
Priors before the development seed: 50 % that dormant protects better than spare at equal total capacity,
25 % that a reset beats keep. After it: 25 % and 15 %.

Test seeds: 1400 to 1409 (10 seeds, never used; checked against every result file).
Hypotheses, two-sided paired Wilcoxon, threshold 0.05 / 6:
- U1 (capacity none): least committed minus none, on availability and on rare R2 after the event.
- U2 (least committed): dormant minus spare, on availability and on rare R2 after the event.
- W1 (16b, capacity none): reset_opt minus keep and reset_both minus keep, on availability.
Descriptive: first post-event recruit (rare specialist? dormant?), killed-function recovery (AUC and final),
other-task R2, number and diversity of post-event recruitments, random rule as control.
Reading fixed in advance: U2 decides whether holding reserves back has any value over simply having more active
modules; W1 decides whether the negative transfer of experiment 15b matters inside the production loop.
