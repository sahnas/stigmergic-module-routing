# Preregistration: experiment 14, does the mechanism preserve a rare specialist?

Written 2026-10-02T22:05Z, before any run on the test seeds.
Code: src/exp_preservation.py (sha256 b112fd314196a825), src/exp_dynamic.py (sha256 c3329b81c02b0eb6), src/exp.py (sha256 6c202cab844bb559).
Production world of experiment 3, one event (death of the module carrying a frequent primitive a), one rare
primitive r whose tasks are sampled with relative weight 0.02. Measures restricted to tasks that need r but not
a (rare), and tasks that need neither (other). Nothing is tuned; the mechanism is as in experiments 3 to 6.
Design: 2 scenarios (rare from the start; learned then rare) x reserves (0 or 2 dormant modules) x recruitment
rule (least committed = current; random; none = alarms fire but nobody is recruited).

Development seed 950, declared before the test. With no reserve, in the learned-then-rare scenario, the rare
skill collapsed (R2 0.90 to about -0.9) under all three rules, including without recruitment: trace sampling
explores other modules for the dead symbol and gradient training through them overwrites their content, so the
sacrifice happens through exploration itself, not only through recruitment. With two dormant reserves the rare
skill was preserved (0.93 to 0.93). In the rare-from-start scenario the rare skill was never acquired
(R2 below 0 before the event).

Test seeds: 1100 to 1109 (10 seeds, never used; checked against every result file).

Hypotheses:
- Q1 (learned then rare, no reserve, least committed): the rare specialist is sacrificed, rare R2 after the
  event below 0.5 in at least 7 seeds of 10, and its module appears among the recruits in at least 7 of 10.
- Q2 (same, rule none): the rare skill collapses as well (rare R2 after below 0.5 in at least 7 of 10):
  exploration alone sacrifices it. Paired comparison least committed minus none on rare R2 after: expected
  near zero; on availability: expected positive. Two-sided Wilcoxon, threshold 0.05 / 2.
- Q3 (learned then rare, two reserves, least committed): the first recruit is a dormant module in at least
  9 of 10 seeds, and rare R2 after stays above 0.8 in at least 8 of 10.
- Q4 (rare from the start, no reserve): the rare skill is not acquired, rare R2 before the event below 0.5 in
  at least 8 of 10 seeds (acquisition, not preservation, is the failure there).
- Random recruitment is a control: it sacrifices the specialist at the chance rate.
Not tested here: criticality (a cost of errors); the mechanism has no notion of cost, which is noted as a limit.
Reading fixed in advance: if Q1 to Q3 hold, the mechanism preserves nothing by itself; only dormant reserves
protect existing functions, and both recruitment and exploration will consume a specialist once reserves are
gone. The reading is independent of whether the skill is rare: a single-function specialist is sacrificed
because it is the least committed by total trace.
