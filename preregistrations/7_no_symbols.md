# Preregistration: step 2, no symbols

> English translation of the French original kept verbatim in `original-fr/`. The hashes below refer to the French files as executed, which are in `original-fr/`. The translated code in `src/` differs only in comments, strings and identifiers; it reproduces stored results bit for bit (see README).

> **Deviation noted at publication.** The descriptive "global capacity" below was specified as clipped at 0 per task, but the stored data keep only per-task means, so the value reported in the README is unclipped. It is descriptive only and was added after seeing the development seeds.

Written 2026-10-01T05:18Z, before any run on the test seeds.
Code: exp_sans_symboles.py (sha256 5c558d088ce911c3), exp.py (sha256 9b0d44c40e14df1d).

Development: seeds 500 and 501 to check the code and tune the reinforcement-learning opponent
(lr_pg in {0.1; 0.5; 2.0}, criterion: learning speed). Kept: lr_pg = 0.1.
The tested mechanism (aco_thresholds) is not tuned.

What the development seeds already showed (to be declared before the test):
aco_thresholds learns the 12 known tasks poorly without symbols (R² 0.26 and 0.42) and learns
new compositions slowly; mono and soft learn new ones fast but destroy the old ones.
No variant aligns its modules with the primitives. I therefore expect N1 and N3 to fail.

Test seeds: 600 to 609 (10 seeds, never used). Variants: mono, soft, rl, aco_thresholds.

Main family (measure defined in the code before development: learning speed on a new
composition, mean R² over 10 evaluations during 200 steps), threshold 0.05 / 3:
- N1: aco_thresholds > soft
- N2: aco_thresholds > rl
- N3: aco_thresholds > mono
Secondary family (interference on the 12 old tasks, lower = better), threshold 0.025:
- I1: aco_thresholds < soft
- I2: aco_thresholds < mono
Two-sided paired Wilcoxon. Confirmed if p below threshold and difference in the stated direction;
contradicted if p below threshold in the other direction; undetermined otherwise.

Descriptive only, measure added AFTER seeing the development seeds: global capacity
= mean R² clipped at 0 over the 13 tasks (12 old + the new one) after the 200 steps.
