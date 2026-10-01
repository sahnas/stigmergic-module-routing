# Preregistration: standard non-stationary bandit opponents

> English translation of the French original kept verbatim in `original-fr/`. The hashes below refer to the French files as executed, which are in `original-fr/`. The translated code in `src/` differs only in comments, strings and identifiers; it reproduces stored results bit for bit (see README).

> **Provenance note added at publication.** This protocol was written and partly run in an earlier execution whose history is absent from the conversation that produced the rest of this repository. Its files were found in the execution environment while preparing publication. The code hashes match. The main test had stopped at 15 of 20 seeds; seeds 95 to 99 were completed with the same code before any analysis. Results are reported in experiment 4 of the README.

Written 2026-10-01T04:01Z, before tuning and before the test seeds.

Code: bandits.py (sha256 1e018844e58d6497), exp_dynamique.py (sha256 c5f0595ecf9201ca), exp.py (sha256 9b0d44c40e14df1d).
Same production scenario as the previous test.

Opponents: SW-UCB, D-UCB, M-UCB (one bandit per symbol, arms = modules, reward = q).
Tuning of the opponents, and of them only, on development seeds 300, 301, 302:
- SW-UCB: window {50, 200, 800} x c {0.05; 0.3}
- D-UCB: gamma {0.98; 0.995; 0.999} x c {0.05; 0.3}
- M-UCB: window {20, 60} x threshold {0.2; 0.4} x c {0.05; 0.3}
Criterion: best mean availability over these 3 seeds, one configuration kept per algorithm.
Evaporation + thresholds (aco_thresholds) keeps its original hyperparameters, without any tuning.

Test seeds: 60 to 79 (20 seeds, never used).
Hypotheses:
- B1: aco_thresholds > best SW-UCB in availability.
- B2: aco_thresholds > best D-UCB in availability.
- B3: aco_thresholds > best M-UCB in availability.
Test: two-sided paired Wilcoxon, threshold 0.05 / 3 = 0.0167, and positive mean difference.
The thesis "beats non-stationary bandits" is retained only if B1, B2 and B3 are all confirmed.
Any other result is reported as is.

## Addendum, written 2026-10-01T04:25Z

An earlier, interrupted run launched this protocol on seeds 60 to 74 (file test_bandits.jsonl)
then started an additional, non-preregistered opponent (exp_rupture.py, reglage.jsonl), discarded.
I did not examine these results before writing this addendum.
Checks done: bandits.py, exp_dynamique.py and exp.py have the same hashes as above;
the kept configurations (bandits_retenus.json) are indeed the best on seeds 300 to 302.
To avoid any contamination, the main test is run on fresh seeds: 80 to 99.
Same code, same configurations, same hypotheses B1 to B3, same threshold.
Seeds 60 to 74 will be reported separately, as a replication.
