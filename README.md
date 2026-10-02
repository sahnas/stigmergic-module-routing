# stigmergic-module-routing

Gradient-free coordination of learning modules through evaporating trails, anomaly alarms and load-aware recruitment. Preregistered toy experiments, with positive and negative results.

A paper describing experiments 1 to 9 is in [`paper/`](paper/) (LaTeX source, bibliography and compiled PDF). It is the version submitted to HAL on 1 October 2026 (hal-05773822, under moderation, with transfer to arXiv requested). Earlier submissions of the same day were withdrawn before moderation and replaced by this one. Experiment 10 was run after that submission and is not in the manuscript.

## Summary

A pool of small neural modules must share a family of tasks. Which module handles which part of a task is not decided by a learned router, but by traces on the edges between task symbols and modules, in the spirit of ant pheromones: traces are reinforced when a module succeeds and evaporate when it is used without success. When a module's performance drops abnormally, an alarm triggers, and the least committed module (the one the routing relies on least) is recruited to take over and relearn the lost function.

What the experiments support, on a toy world:

- Forgetting is necessary for compensation. Without it, the system never recovers from the permanent loss of a module.
- At equal rate, forgetting triggered by solicitation preserves memory, whereas literal time-based evaporation destroys it. A much slower time-based rate was not tested.
- What carries the resilience is load-aware recruitment, i.e. a piece of collective information read at the moment of failure. Here a module's load is its total trace, how much the routing already relies on it, not its computational load: the recruit is the least committed module. Recruiting at random is much worse. Individual alarm thresholds per unit bring nothing measurable over a single collective threshold.
- Without any routing gradient, the mechanism clearly beats simple recency-based routing and change-point bandits (M-UCB family).

What the experiments do not support:

- It does not beat the best standard non-stationary bandits (sliding-window UCB and discounted UCB). Differences are small and not significant; the profiles differ (better early learning and early events for the mechanism, better late recovery for the bandits).
- It does not beat a router learned by gradient when gradients are available. A sparse top-2 mixture of experts ties with it in learning and recovers almost as well; a top-1 router learns better but never recovers from the loss of a module (experiment 8).
- Individual alarm thresholds bring nothing measurable, even with heterogeneous modules (experiment 9).
- It does not discover compositional structure. When task symbols are replaced by opaque identifiers, it fails to learn even the known tasks, and no tested method aligns its modules with the underlying primitives.

In short: a coordination and resilience mechanism, roughly at the level of the best standard non-stationary bandits, not a mechanism for discovering structure or for compositional generalisation.

## The mechanism

- **Modules.** Small regressors (linear map + 32-unit ReLU MLP) that communicate through the observation space, a fixed interface in the spirit of Unix text streams. No shared latent space. Modules learn their content by gradient along the path they are used on.
- **Traces.** One trace per (task symbol, module). A module is chosen by sampling softmax(trace / 0.05). After each batch, all traces of the solicited symbol evaporate by 2 %, and the edge used receives a deposit proportional to the batch quality q (R² clipped to [0, 1]).
- **Alarm.** Each module tracks a moving average of its quality and of its deviation. If q falls below mean minus 3 deviations (after at least 50 uses, outside a 100-step cooldown), an alarm fires.
- **Recruitment.** On alarm, the active module with the lowest total trace (the least committed one; this total trace is what we call its load, not a computational load), off the current path, has its trace for that symbol raised to the incumbent's level. It then competes, learns, and either takes over or fades.

## World

Five fixed primitives P_k on R^8 (random rotation plus a tanh nonlinearity). A task is one primitive or the composition of two, y = P_j(P_i(x)). In experiments 1 to 6 the system receives the task's symbols, so the decomposition is given. In experiment 7 it receives only an opaque task identifier.

## Experiments

All tests are paired two-sided Wilcoxon tests across seeds, with thresholds fixed in advance. Opponents were tuned on separate development seeds; the tested mechanism (`aco_thresholds`) was never tuned and kept the same hyperparameters throughout.

### 1. Core test with symbols (seeds 0 to 9)

Hypotheses in the docstring of `src/exp.py`, written before the test seeds. Continual learning over 17 tasks, then the module carrying one primitive dies permanently.

| Variant | Retention (mean / median) | Collapses | Zero-shot (mean / median) | Tasks with k* after recovery | Rerouting (steps, median) |
|---|---|---|---|---|---|
| monolithic MLP | -0.47 / -0.46 | 10/10 | -0.50 / -0.50 | -0.05 | n/a |
| fixed orchestration | 0.94 / 0.97 | 0/10 | 0.91 / 0.96 | 0.00 | never |
| router learned by gradient | 0.81 / 0.86 | 0/10 | 0.73 / 0.82 | 0.98 | 324 |
| traces without forgetting | 0.75 / 0.96 | 2/10 | 0.70 / 0.94 | 0.00 | never |
| forgetting by recency | 0.76 / 0.95 | 2/10 | 0.68 / 0.94 | 0.75 | 524 |
| evaporation | 0.83 / 0.94 | 2/10 | 0.72 / 0.92 | 0.92 | 555 |
| time-based evaporation | 0.07 / 0.08 | 10/10 | -0.09 / -0.04 | 0.13 | 37 |
| evaporation + thresholds | 0.91 / 0.92 | 0/10 | 0.88 / 0.88 | 0.97 | 4.5 |

H1 (forgetting enables compensation), H2 (thresholds speed it up) and H3 (time-based forgetting at the same rate destroys retention) confirmed, 10/10 seeds, p = 0.002. H4 partial: evaporation beats the monolithic MLP but not the gradient router (p = 0.43). Control: evaporation does not differ from forgetting by recency here. An exploratory advantage of `aco_thresholds` over the gradient router (zero-shot 0.88 vs 0.73) was not preregistered and failed to replicate in experiment 2.

### 2. Confirmation (seeds 20 to 29)

The exploratory advantage over the gradient router is not confirmed: zero-shot +0.008 (95 % CI -0.09 to +0.09, p = 0.63), retention +0.014 (95 % CI -0.04 to +0.07, p = 0.56). On this world the mechanism ties with a router learned by gradient.

### 3. Gradient-free thesis (seeds 40 to 59)

The routing receives no gradient. Production scenario after initial learning: death of a module, arrival of two new modules plus a death, a change in the world, then a simultaneous double death. Seven modules out of ten slots at the start; the system does not know which modules are dead. Measure: availability, the mean R² over the 17 tasks throughout production.

| Variant | Availability | Collapses at initial learning | Final R² | Final zero-shot |
|---|---|---|---|---|
| recency | 0.41 | 10/20 | 0.22 | 0.07 |
| recency + thresholds | 0.51 | 7/20 | 0.61 | 0.45 |
| evaporation | 0.57 | 4/20 | 0.37 | 0.20 |
| evaporation + thresholds | 0.76 | 0/20 | 0.60 | 0.39 |

S1 (full mechanism vs recency): +0.35 (95 % CI +0.29 to +0.40), 20/20. S2 (effect of thresholds): +0.14, 18/20, p = 0.0001. S3 (effect of the forgetting rule, sign not predicted): evaporation beats recency by +0.21, 20/20. On the 7 seeds where both start healthy, the mechanism loses 0.30 of R² across the events, recency 0.77.

### 4. Standard non-stationary bandits (main test seeds 80 to 99, replication 60 to 74)

See the provenance note in the integrity section. Opponents tuned over 20 configurations on development seeds.

| Variant | Availability | Initial R² | Final R² |
|---|---|---|---|
| evaporation + thresholds | 0.74 | 0.94 | 0.65 |
| SW-UCB (window 50) | 0.72 | 0.69 | 0.86 |
| D-UCB (gamma 0.98) | 0.72 | 0.66 | 0.88 |
| M-UCB (change detection) | 0.59 | 0.90 | 0.72 |

B1 vs SW-UCB: +0.02 (95 % CI -0.02 to +0.06), p = 0.39, not confirmed. B2 vs D-UCB: +0.03 (95 % CI -0.01 to +0.06), p = 0.20, not confirmed. B3 vs M-UCB: +0.15, 20/20, confirmed. The replication on 15 other seeds gives the same picture (B1 and B2 positive but p = 0.03, above the preregistered threshold). The preregistered thesis "beats non-stationary bandits" required all three and is therefore not retained.

### 5. Tuned change-point bandit (seeds 60 to 79)

An independent implementation of a change-point bandit (UCB, sliding window, reset on drop), tuned over 16 configurations. R1: mechanism +0.16 (95 % CI +0.11 to +0.22), 18/20, p < 0.0001. R2: the opponent beats recency by +0.16, so it is a serious opponent. The opponent learns better at the start (0.95 vs 0.91) but loses more across events (0.37 vs 0.21). Consistent with B3 in experiment 4.

### 6. Ablation (seeds 80 to 99)

| Variant | Availability | Final R² | Collapses |
|---|---|---|---|
| evaporation + individual thresholds + least-committed recruitment | 0.74 | 0.65 | 0/20 |
| same, random recruitment | 0.52 | 0.48 | 2/20 |
| same, single collective threshold | 0.74 | 0.65 | 0/20 |

M1 (least committed beats random): +0.22 (95 % CI +0.19 to +0.27), 20/20, confirmed. M2 (individual beats collective thresholds): +0.002 (95 % CI -0.03 to +0.03), undetermined; any effect is below 0.03. In this scenario all modules are identical, which may explain why individuality does not matter.

### 7. No symbols (seeds 600 to 609)

Each task now has only an opaque identifier; the system must discover that tasks share primitives. Phase A: interleaved learning of 12 known compositions. Phase B: 200 steps on each of 8 new compositions, from the same state.

| Variant | Known tasks R² | Learning speed on new task | Final R² new task | Interference on old tasks | Module/primitive alignment |
|---|---|---|---|---|---|
| monolithic MLP | 0.95 | 0.96 | 0.98 | 0.60 | n/a |
| router learned by gradient | 0.91 | 0.89 | 0.97 | 2.23 | -8.8 |
| reinforcement-learning router | 0.13 | 0.29 | 0.50 | 0.21 | -0.24 |
| evaporation + thresholds | 0.21 | 0.29 | 0.55 | 0.21 | -0.20 |

N1 and N3 contradicted: the mechanism learns new compositions far more slowly than the gradient router and the monolithic MLP (0/10 seeds). N2 undetermined: tie with the reinforcement-learning router. The lower interference (I1, I2 confirmed) is not a virtue: little was learned in phase A. No variant aligns its modules with the primitives: nobody discovers the decomposition. The MLP and the gradient router learn each task as its own function and overwrite the old ones. These failures were visible on the development seeds and were declared in the preregistration before the test.

A plausible diagnosis, not tested: credit assignment. The only signal is the quality of the whole path, near zero until both modules are right at once, and nothing in the traces indicates that two opaque tasks share a primitive. A local prediction error per module (for example JEPA-style modules) would provide the missing signal.

### 8. Sparse top-k mixture of experts (seeds 700 to 719)

Protocol of experiment 1, against routers trained by gradient with noisy top-k gating, tuned on development seeds 100 to 102. Two opponents were kept after seeing the tuning results, as declared in the preregistration: the best top-1 configuration never recovers from the loss of a module, which would have made the compensation comparison trivial.

| Variant | Retention | Zero-shot | Tasks with k* after recovery |
|---|---|---|---|
| evaporation + thresholds | 0.89 | 0.87 | 0.97 |
| top-1 (noise 0.3, balancing 0.01) | 0.96 | 0.95 | 0.05 |
| top-2 (noise 0.3) | 0.93 | 0.89 | 0.96 |

Against top-2: zero-shot -0.03 and retention -0.04 (not significant at the preregistered threshold, p = 0.009 and 0.012 against 0.0083), recovery +0.011 (18/20, significant but negligible). Against top-1: zero-shot -0.08 and retention -0.07 (significant, top-1 better), recovery +0.93 (20/20). With gradients available, the mechanism brings no learning advantage; its only clear compensation advantage is over top-1 routing, which cannot reroute after a module dies.

### 9. Heterogeneous modules (seeds 720 to 739)

Production scenario of experiment 3 with module hidden sizes {2, 4, 8, 16, 32, 64, 2, 4, 8, 16}. Bandits keep the configurations of experiment 4, as preregistered; nothing is tuned.

| Variant | Availability | Initial R² | Collapses at initial learning | Final R² |
|---|---|---|---|---|
| evaporation + individual thresholds | 0.72 | 0.89 | 0/20 | 0.55 |
| evaporation + collective threshold | 0.72 | 0.89 | 1/20 | 0.53 |
| SW-UCB (window 50) | 0.71 | 0.75 | 2/20 | 0.76 |
| D-UCB (gamma 0.98) | 0.68 | 0.63 | 7/20 | 0.74 |

HE1 (individual thresholds help with heterogeneous modules): +0.002, not confirmed. HE2 vs SW-UCB: +0.01, not significant. HE3 vs D-UCB: +0.04 (95 % CI +0.01 to +0.07), 16/20, p = 0.014, significant. Caution: the bandits were tuned for homogeneous modules, and D-UCB collapses at initial learning on 7 of 20 seeds here, so this edge is likely a tuning artefact rather than a structural advantage. The profiles of experiment 4 persist: better initial learning for the mechanism, better final recovery for the bandits.

### 10. Oracle local credit (seeds 800 to 809, run after the submitted manuscript)

Diagnostic of experiment 7: each slot's quality is computed against the true primitive applied to the module's actual input (an oracle, hence an upper bound). `local_routing` uses it for traces and alarms only; `local_full` also lets each module learn on its local target. Preregistration 10; the development seed and the expected outcome were declared in advance.

| Variant | Known tasks R² | Learning speed on new task | Final R² new task | Interference | Alignment |
|---|---|---|---|---|---|
| global credit (experiment 7 baseline) | 0.18 | 0.31 | 0.59 | 0.21 | -0.24 |
| local credit for routing only | 0.09 | 0.22 | 0.40 | 0.19 | -0.27 |
| local credit for routing and learning | 0.97 | 0.94 | 0.98 | 0.20 | 0.99 |
| router learned by gradient (reference) | 0.90 | 0.89 | 0.96 | 2.29 | -8.6 |

O1 and O3 contradicted: local credit for routing alone makes things worse (-0.09 on known tasks, -0.09 in learning speed, 0/10 seeds). O2 not confirmed. O4 confirmed: with a local learning signal for the modules, the trace mechanism learns new compositions fast (+0.63, 10/10, p = 0.002), aligns modules with primitives (0.99) and keeps interference an order of magnitude below the gradient router (0.20 against 2.29, descriptive). Reading: the lock of experiment 7 is the learning signal of the modules, not the traces. Once modules have their own error, traces indexed by opaque identifiers find correct paths. Removing the oracle, i.e. giving modules a self-supervised local error, is the open question.

## Integrity notes

- **Development seeds.** Every test used seeds separate from the development seeds used to check code and tune opponents. What development revealed is stated in the preregistrations.
- **Protocol found after the fact.** While preparing publication, the files of a preregistered protocol (experiment 4) were found in the execution environment, from an earlier execution absent from the history of the conversation that produced the rest of this work. It had not been reported. Its code hashes match its preregistration; its main test had stopped at 15 of 20 seeds and was completed with the same code before any analysis. It is reported here in full, including the fact that it weakens an earlier claim.
- **Seed reuse.** The preregistrations of experiments 5 and 6 state that their seeds had never been used. Seeds 60 to 74 and 80 to 99 had in fact been used by experiment 4. No decision depended on those results, which were unknown at the time, and the code is deterministic: the tested mechanism gives bit-identical results on the 20 shared seeds in both runs. The statement was nevertheless false; corrections are attached to the translated preregistrations.
- **A second set of files from an unseen execution.** After the v1 manuscript, the preregistrations of experiments 8 and 9, their code, the completed test runs of experiment 8 and the beginning of a third, unpreregistered experiment were again found in the execution environment, from an execution absent from the conversation history. The quoted code hashes match, and the preregistrations were written before the test runs. Experiment 8 was analysed and experiment 9 was run according to their written protocols. The third experiment was not continued; its development-seed tuning is kept in `results/draft_regimes_dev/` and `src/exp_regimes.py`, marked as a draft and not analysed.
- **Provenance check.** `python tools/check_provenance.py` recomputes every code hash quoted in the preregistrations and prints a ledger of seeds per result file, with all overlaps between test seeds of different experiments. Output in `tools/provenance_report.txt`: all hashes match, and the only overlaps are the two disclosed above.
- **Translation.** Everything was first written in French. `original-fr/` contains the files exactly as executed; the hashes quoted in the preregistrations refer to them. The English code in `src/` differs only in comments, strings and identifiers. It was checked by re-running one seed of every experiment family (9 runs): all stored results are reproduced bit for bit. The English analysis scripts reproduce every number of the original outputs.
- **How this was produced.** The research question, the intuitions (plurality of small units, orchestration plus Unix-style composition, saga-style compensation, per-unit triggers, ants, forgetting) and the decisions at each step come from the author. The code, experiments and write-up were produced with Claude (Anthropic), an AI assistant, in conversation with the author.

## Related work

Stigmergy (Grassé, 1959); ant colony optimisation (Dorigo) and AntNet (Di Caro and Dorigo, 1998); response-threshold models of division of labour (Bonabeau, Theraulaz and Deneubourg, 1996; Theraulaz et al., 1998); ant-like task allocation and recruitment in robots (Krieger, Billeter and Keller, Nature, 2000); Artificial Neural Tissue, chemical-like regulation with self-organised task decomposition learned by evolution (Thangavelautham and D'Eleuterio, IEEE TNNLS, 2012); Routing Networks, gradient-free routing between learning function blocks (Rosenbaum et al., 2018); mixtures of experts (Jacobs et al., 1991; Shazeer et al., 2017); shared global workspace between modules (Goyal et al., 2022); non-stationary bandits (Garivier and Moulines, 2011; Cao et al., 2019); pheromone-based fault-aware routing between LLM agents (StigmergyRouter, ACM CAIS 2026) and ant-colony routing for multi-agent LLM systems (AMRO-S, 2026). The specific combination studied here, modules that learn during operation together with the routing, solicitation-triggered forgetting, and load-aware recruitment of a module that relearns a lost function, was not found in these works; this is based on targeted searches, not an exhaustive review.

## Limitations

A single toy world (5 primitives, dimension 8); scenarios designed to probe resilience, hence not neutral; modest tuning budgets for opponents; no sparse top-k mixture of experts or contextual bandit among the opponents; a single evaporation rate, so a much slower time-based evaporation was not tested; homogeneous modules, so total trace is a reasonable proxy for availability only in this setting; task decomposition given in experiments 1 to 6; modules are supervised regressors, not JEPAs; no real data.

## Reproducing

Python 3.12, CPU only. Versions used: torch 2.14.1 (CPU), numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.8.

```
pip install -r requirements.txt
cd src
python exp.py --seeds 0,1,2,3,4,5,6,7,8,9 --out ../results/1_core/results.jsonl
python exp.py --seeds 20,21,22,23,24,25,26,27,28,29 --variants soft,aco_thresholds,aco --out ../results/2_confirmation/confirmation.jsonl
python exp_dynamic.py 40,41,...,59 ../results/3_gradient_free/gradient_free.jsonl
python exp_bandits.py 80,...,99 ../results/4_standard_bandits/main_seeds_80_99.jsonl aco_thresholds,swucb:50:0.05,ducb:0.98:0.3,mucb:60:0.2:0.3
python exp_changepoint.py 60,...,79 ../results/5_changepoint/changepoint_test.jsonl ema,aco_thresholds,changepoint '{"sel":"ucb","c":0.1,"h":20,"b":0.2}'
python exp_ablation.py 80,...,99 ../results/6_ablation/ablation.jsonl aco_thresholds,aco_thresholds_random,aco_global_threshold
python exp_no_symbols.py 600,...,609 ../results/7_no_symbols/no_symbols.jsonl mono,soft,rl,aco_thresholds '{"lr_pg":0.1}'
```

Write seed lists in full (no "..."). Runs append to the output file. About 10 to 70 seconds per seed and variant group on one CPU core. Analyses, from the repository root:

```
python src/analysis_core.py results/1_core/results.jsonl
python src/analysis_confirmation.py results/2_confirmation/confirmation.jsonl
python src/analysis_gradient_free.py results/3_gradient_free/gradient_free.jsonl
python src/analysis_bandits.py results/4_standard_bandits
python src/analysis_changepoint.py results/5_changepoint
python src/analysis_ablation.py results/6_ablation/ablation.jsonl
python src/analysis_no_symbols.py results/7_no_symbols/no_symbols.jsonl
python src/analysis_topk.py results/8_topk_moe
python src/analysis_heterogeneous.py results/9_heterogeneous/heterogeneous.jsonl
python src/analysis_local_credit.py results/10_local_credit/local_credit.jsonl
python tools/check_provenance.py
```

Experiments 8 and 9 (from `src/`): `python exp_topk.py 700,...,719 ../results/8_topk_moe/topk2.jsonl topk '{"k":2,"noise":0.3,"alpha":0.0}'` (and the top-1 configuration), `python exp_heterogeneous.py 720,...,739 ../results/9_heterogeneous/heterogeneous.jsonl aco_thresholds,aco_global_threshold,swucb:50:0.05,ducb:0.98:0.3`.

## Layout

- `src/`: experiment and analysis code (English).
- `results/<n>_<name>/`: raw results (JSON lines), summaries and figures; development and tuning runs are kept alongside.
- `preregistrations/`: English translations of the preregistrations, with corrections attached.
- `tools/`: provenance check and its report.
- `paper/`: manuscript (LaTeX, BibTeX, compiled `.bbl` and PDF).
- `original-fr/`: all files exactly as executed, in French, including smoke tests and a discarded, non-preregistered tuning run.

## Archived versions

The Git history of this repository was squashed into a single commit. Earlier states remain available on Software Heritage: `swh:1:snp:ec5e807533a26aef69001187d77b62911d3cb4c8` (first publication, code and data), `swh:1:snp:78775eeca41089d5cc7dcbc89d25f15d25ad7ec2` (manuscript added), `swh:1:snp:c2c564fadd4d0c73cd3af983ec94d25d23592d77` (references verified), `swh:1:snp:4b4ffa5b9b0b5d2de38c75d5a270979e7f9158ac` (licence added).

## License

Apache License 2.0, see [LICENSE](LICENSE).
