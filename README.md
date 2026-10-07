# stigmergic-module-routing

Gradient-free coordination of learning modules through evaporating trails, anomaly alarms and load-aware recruitment. Preregistered toy experiments, with positive and negative results.

A manuscript draft covering the whole study is in [`paper/`](paper/) (LaTeX source, bibliography and compiled PDF): the toy experiments, the preservation and modular JEPA experiments, the continual pilot on LeWorldModel's frozen representation, and the planning gap with its diagnostics. It is a working draft, kept in step with the results below; 239 constants reported in it are recomputed from the raw results by `paper/check_numbers.py` (experiments 12 to 16 are not covered by that script). Earlier draft states are tagged `v1-draft` and `v2-draft`.

## Summary

Seventeen preregistered experiments, from a synthetic compositional world to the released LeWorldModel, on the idea that small learning modules could coordinate through stigmergic traces (reinforced by use, evaporated when solicited) and recruit the least committed module when one fails. What held, what did not, and what was found on the way:

- **Resilience without a routing gradient (experiments 1 to 6, 9).** Forgetting is necessary to compensate the permanent loss of a module, and at equal rate evaporation triggered by solicitation preserves retention where time-based evaporation destroys it. The resilience comes from recruiting the least committed module; per-module alarm thresholds bring nothing, even with heterogeneous modules. Against standard non-stationary bandits the mechanism matches sliding-window and discounted UCB and beats change-point and recency rules.
- **No advantage with gradients (experiments 1, 2, 8).** It does not beat a router trained by gradient, dense or sparse; a top-1 router cannot reroute after a module dies, a top-2 router can.
- **No structure discovery without a local signal (7, 10, 10b, 11).** With opaque task identifiers nothing is learned. With an oracle local error per module, trace routing finds the shared structure with a clean decomposition and little interference, where dense and sparse gradient routers each fail on speed, alignment or interference. The lock was the modules' learning signal.
- **No protection of functions (14, 14b, 15, 15b, 16b).** Once no spare capacity is left, a specialist is consumed whatever the recruitment rule, through exploration itself; only reserves held back until an event preserve it; a consumed specialist relearns no faster than a fresh module, a module repurposed to another function relearns slower than a fresh one, and resetting a recruited module does not help.
- **Modular JEPA (12, 13).** Competition by local error replicates COMET and beats attention; the prior over the winner needs memory, not the current state; traces add nothing to persistence; surprise-triggered recruitment instantiates a new mechanism without degrading the old ones, but no better than over-provisioning.
- **Continual dynamics on LeWorldModel's frozen representation (17).** Against a 20,000-window reservoir replay with no tuned parameter, banks of specialists with freezing and held-back reserves are not better on any retention measure, at twice the forward passes. The line stops by the preregistered rule.
- **Found on the way, first a mistake.** LeWorldModel's own predictor architecture, retrained from scratch on the frozen released encoder with a loss on the last position of three real frames, has eleven times lower latent error on that setting (0.006 vs 0.067) and plans far worse (36 % vs 85 % on Two-rooms). Six offline diagnostics, all starting from three real frames (normalisation, random-action error, rollouts, candidate ranking, wall crossing, cancelling sequences), could not explain it. An external reviewer read the recipe instead: the planner starts from one frame, the official loss supervises every position, and the retrained network's error at a one-frame context was 1.18. Retrained with the official loss it plans like the released predictor (84, 85, 89, 81 vs 85, 83, 87, 80 on four seeds) with twelve times lower latent error. The gap was ours; the lesson is a methods result.
- **Experiment 18, a bilinear predictor in place of the released one.** A predictor fitted by ridge least squares on the cache in one minute, linear in the last three embeddings and the action block and bilinear in current embedding and action block, has a relative one-step latent error of 0.69 on held-out windows (ten times the released predictor's, a hundred times the corrected neural predictor's); its accuracy at short contexts has not been measured yet. With the authors' evaluation unchanged it succeeds in 97, 99, 100 and 96 of 100 episodes on evaluation seeds 42 to 45, against 85, 83, 87 and 80 for the released predictor and 84, 85, 89 and 81 for the neural predictor retrained with the official loss (preregistered confirmation on the three fresh seeds; discordant episodes against the corrected neural predictor 15/2, 14/0, 11/0, 16/1; the official evaluation samples starts and goals from the whole dataset). Removing the bilinear term gives 57 % on seed 42; removing the history changes little on that seed (96 %); ridge x10 or /10 changes nothing on that seed. In a separate exploitation probe (different goals, contexts of three real frames), the per-start median ratio of true to predicted cost is 1.6 for this predictor, 1.5 for the released one and 13.1 for the defective retrained one, but the bilinear predictor's executed plans cost more than the released model's on 25 of 30 starts. Three predictors on one encoder with latent errors 0.005, 0.067 and 0.69 plan at about 85, 84 and 98: latent error does not rank them. The mechanism of the bilinear gain is unresolved. Claimed for Two-rooms and this encoder only.

Defensible statement of the whole: a distributed backup-capacity heuristic based on module commitment gives resilience comparable to good non-stationary bandits without a routing gradient; it protects no function by itself, discovers no structure without a local learning signal, and is not worth more than replay for continual world models. On the world model itself, on Two-rooms, a ridge-fitted bilinear predictor with ten times the latent error of the end-to-end predictor plans better than it, and than a correctly retrained neural predictor, on its own encoder; average latent error does not rank these predictors by planning success, and why the bilinear predictor wins is not established. Every claim has its preregistration, seeds, code hashes and raw results in this repository; the integrity notes list what went wrong and how it was corrected.

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

### 10. Oracle local credit (seeds 800 to 809)

Diagnostic of experiment 7: each slot's quality is computed against the true primitive applied to the module's actual input (an oracle, hence an upper bound). `local_routing` uses it for traces and alarms only; `local_full` also lets each module learn on its local target. Preregistration 10; the development seed and the expected outcome were declared in advance.

| Variant | Known tasks R² | Learning speed on new task | Final R² new task | Interference | Alignment |
|---|---|---|---|---|---|
| global credit (experiment 7 baseline) | 0.18 | 0.31 | 0.59 | 0.21 | -0.24 |
| local credit for routing only | 0.09 | 0.22 | 0.40 | 0.19 | -0.27 |
| local credit for routing and learning | 0.97 | 0.94 | 0.98 | 0.20 | 0.99 |
| router learned by gradient (reference) | 0.90 | 0.89 | 0.96 | 2.29 | -8.6 |

O1 and O3 contradicted: local credit for routing alone makes things worse (-0.09 on known tasks, -0.09 in learning speed, 0/10 seeds). O2 not confirmed. O4 confirmed: with a local learning signal for the modules, the trace mechanism learns new compositions fast (+0.63, 10/10, p = 0.002), aligns modules with primitives (0.99) and keeps interference an order of magnitude below the gradient router (0.20 against 2.29, descriptive). Reading: the lock of experiment 7 is the learning signal of the modules, not the traces. Once modules have their own error, traces indexed by opaque identifiers find correct paths. Removing the oracle, i.e. giving modules a self-supervised local error, is the open question.

### 10b. Independent re-implementation of experiment 10, plus a gradient router with local losses (seeds 810 to 819)

Written without knowledge of experiment 10 (see the integrity notes), with a slightly different oracle for slot 2 (the module applied to the true intermediate, compared with y). On the three seeds shared with experiment 10 (800 to 802, kept apart in `10b_seeds_800_802_before_discovery.jsonl`), the reference values are identical and the local variants agree within 0.04, except one known-task value that differs by 0.10.

| Variant | Known tasks R² | Speed on new task | Final R² new task | Interference | Alignment |
|---|---|---|---|---|---|
| reference (global credit) | 0.20 | 0.30 | 0.58 | 0.22 | -0.28 |
| level 2: local credit for routing and content | 0.97 | 0.95 | 0.98 | 0.16 | 0.99 |
| dense gradient router, same local losses | 0.99 | 0.95 | 0.99 | 0.66 | -1.65 |

P1 (replication of O4): +0.65, 10/10. P2: tie on speed with the gradient router (-0.008, p = 0.85). P3: interference 0.16 vs 0.66, 0/10 seeds higher for traces. P4: alignment 0.99 vs -1.65, 10/10. Given the same local losses, hard assignment by traces yields a clean decomposition where the dense gradient mixture smears modules and overwrites known tasks.

### 11. Sparse top-k routers with local losses (seeds 820 to 829)

Same setting as 10b level 2, against top-1 and top-2 routers (configurations of experiment 8) given the same local losses. Prior stated before the test: 60 % that a sparse router aligns as well as traces.

| Variant | Known tasks R² | Speed on new task | Interference | Alignment |
|---|---|---|---|---|
| trace routing, local losses | 0.97 | 0.95 | 0.19 | 0.99 |
| top-1 router, local losses | 0.97 | 0.66 | 0.28 | 0.46 |
| top-2 router, local losses | 0.99 | 0.87 | 1.21 | 0.94 |

Against top-1: speed +0.29 and alignment +0.53 for traces (10/10 each), interference not significant. Against top-2: speed +0.08 and alignment +0.05 for traces (10/10 each), interference 1.21 vs 0.19 (10/10). The combination of fast learning, clean alignment and low interference is specific to the trace rule among the routers tested. Caveat: top-k configurations were not re-tuned for local losses.

### 14. Does the mechanism preserve a rare specialist? (seeds 1100 to 1109, not in the manuscript)

Production world of experiment 3, one event (death of the module carrying a frequent primitive), one primitive r whose tasks are sampled with relative weight 0.02; measures on the tasks that need r but not the killed primitive. Two scenarios (rare from the start; learned then rare) x spare modules (0 or 2) x recruitment rule (least committed; random; none, alarms without recruitment). Preregistered after a development seed whose observations are declared in the preregistration.

Three protocol flaws were identified by an external review on 2026-10-04 and are kept visible here rather than hidden: the two "reserve" modules are active from the start (they take part in sampling and training), so that condition measures over-provisioning, not dormancy; the recruitment rule also acts during initial learning, where alarms fire too, so the three rules diverge before the event and their states before it differ (visible in the "rare R² before" column); and the "first recruit" column counts recruitments from the start of learning, not from the event. A corrected replication (14b: common learned state duplicated at the event, rules applied only afterwards, recruitments timestamped, a true dormant condition) is planned; the results below describe full-policy trajectories and should be read as such.

| Scenario, reserves, rule | Rare R² before | Rare R² after | Other R² after | First recruit = rare specialist | Rare skill preserved (R² > 0.8) |
|---|---|---|---|---|---|
| learned then rare, 0, least committed | 0.87 | -0.09 | 0.68 | 1/10 | 2/10 |
| learned then rare, 0, random | 0.62 | -0.44 | 0.79 | 0/10 | 0/10 |
| learned then rare, 0, none | 0.31 | -0.53 | 0.83 | - | 1/10 |
| learned then rare, 2, least committed | 0.77 | 0.68 | 0.98 | 2/10 | 8/10 |
| rare from the start, 0, least committed | -0.46 | -0.53 | 0.93 | 6/10 | 0/10 |

A learned skill that becomes rare is not targeted by recruitment (its traces survive rarity, as the solicitation-triggered rule predicts), but it is destroyed anyway once no reserve is available, under every rule including no recruitment at all: alarms cascade (about 38 recruitments per run), trace sampling explores every module for the orphaned symbol, and gradient training through them overwrites their content. Two spare active modules preserve it (8/10). A skill that is rare from the start is never acquired (10/10) and its module is the preferred target of least-committed recruitment (6/10 vs 1/10 for random). The mechanism allocates work; nothing in it protects a function in use, and only spare capacity protected one here. Criticality (a cost of errors) is not modelled and was not tested.

### 15. Savings of a consumed specialist (seeds 1200 to 1209, not in the manuscript)

After the production phase of experiment 14 (no reserve), the consumed specialist of r, a live module that carried another primitive, and a fresh module relearn the single task (r,) in isolation for 150 steps with the same batches.

| Candidate | R² before relearning | AUC (150 steps) | Steps to R² 0.8 |
|---|---|---|---|
| consumed specialist | +0.07 | 0.80 | 43 |
| fresh module | -0.44 | 0.72 | 58 |
| module that carried another primitive | -0.71 | 0.58 | 83 |

S1 (consumed vs fresh): +0.075 on AUC, 6/10, p = 0.23, not significant; the residual memory exists but is not a reliable advantage. S2 (consumed vs repurposed module): +0.22, 8/10, p = 0.0098 (sign test p = 0.11). Exploratory, not preregistered: the repurposed module is a worse starting point than a fresh one (-0.15 on AUC, 1/10, p = 0.02; +25 steps to 0.8, 9/10, p = 0.006). Reading: cheap reconstruction is not a reliable protection here, spare capacity remains the only observed one, and repurposing a trained module costs more than starting fresh. Protocol flaw found by the external review of 2026-10-04: `torch.manual_seed` does not reset the world's own generator, so the three candidates did not receive the same batches, contrary to the comment in the code; a corrected replication (15b) with identical pre-drawn streams is planned. The toy's primitives are easy to learn from scratch (58 steps), which bounds how much savings can matter; on a real world model the same question has a different cost scale.

### 15b. Savings, corrected protocol (seeds 1210 to 1219)

Same as 15, with the relearning batches and the evaluation batch drawn once per seed and reused for every candidate.

| Candidate | R² before relearning | AUC (150 steps) | Steps to R² 0.8 |
|---|---|---|---|
| consumed specialist | -0.01 | 0.79 | 48 |
| fresh module | -0.40 | 0.73 | 58 |
| module that carried another primitive | -0.83 | 0.53 | 90 |

S1 (consumed vs fresh): +0.06, 6/10, p = 0.32, not significant. S2 (consumed vs repurposed): +0.26, 8/10, p = 0.020, not significant at the preregistered threshold of 0.0167 (it was significant in 15, which had the flawed streams). S3 (repurposed vs fresh, preregistered this time): -0.20, 0/10, p = 0.002, significant. The stable finding across 15 and 15b is the negative transfer: a module trained for one function is a worse starting point for another than a fresh one. The savings of the consumed specialist remain unproven.

### 14b and 16b. Preservation and reset on a common-state harness (seeds 1400 to 1409)

Corrected protocol (see the flaws recorded under 14 and 15): per seed and capacity, one model learned up to the event and deep-copied before every condition (weights, optimizer state, traces, alarm statistics, counters, routing generator); production data drawn once and shared; recruitments timestamped. Capacity: none (5 active), spare (7 active from the start), dormant (5 active + 2 inactive held back until the event). Rules after the event: least committed, random, none. 16b: at the first post-event recruitment, keep / reset optimizer state / reset weights and optimizer state.

| Capacity | Rule or intervention | Availability | Rare R² before | Rare R² after | Killed function recovered | Other R² after | First recruit dormant |
|---|---|---|---|---|---|---|---|
| none | least committed | 0.46 | 0.89 | -0.54 | 0.59 | 0.86 | - |
| none | random | 0.41 | 0.89 | -0.33 | 0.54 | 0.58 | - |
| none | none | 0.47 | 0.89 | -0.54 | 0.58 | 0.84 | - |
| spare | least committed | 0.76 | 0.59 | 0.61 | 0.92 | 0.97 | - |
| dormant | least committed | 0.87 | 0.89 | 0.92 | 0.96 | 0.98 | 10/10 |
| none | 16b reset optimizer | 0.45 | 0.89 | -0.54 | 0.60 | 0.85 | - |
| none | 16b reset weights and optimizer | 0.40 | 0.89 | -0.78 | 0.66 | 0.87 | - |

U1: with no spare capacity, no benefit of least-committed recruitment over no recruitment is established on the criteria tested with this sample (-0.01 on availability, 95 % CI [-0.07, +0.03]; -0.002 on the rare skill, [-0.59, +0.58]; both p = 0.77); the rare skill collapses with and without recruitment, and the first post-event recruit is the rare specialist in 3/10 seeds (random: 2/10), so no targeting is visible. U2: held-back reserves versus spares active from the start: +0.11 on availability (6/10, p = 0.32) and +0.31 on the rare skill (8/10, p = 0.037), neither significant at the threshold of 0.0083. The two conditions do not start from the same learned state (rare R² 0.89 vs 0.59 before the event, because spares active during learning dilute the rare skill), so the final gap cannot be attributed to better retention after the event; the exploratory difference on the change rare_after - rare_before is +0.01 ([-0.07, +0.13], p = 0.77, not preregistered). W1: no gain of availability was shown from resetting the recruited module's optimizer state (-0.01, [-0.05, +0.01], p = 0.46), and resetting its weights as well gave an unfavourable signal that does not pass the preregistered threshold (-0.06, [-0.11, -0.02], 1/10, p = 0.039). The negative transfer measured in isolated relearning (15b) exists, but removing it by a reset did not improve availability here. Reading: in these runs, activable held-back reserves are associated with better preservation; the contributions of the least-committed rule and of the activation policy remain to be separated, and reset is not justified by these results. The "before" fields are measured after the module is switched off and before production; a control replay on all seeds for the none and spare capacities gave identical values with and without the switch on these filtered task groups.

Reproducibility (added 2026-10-04). An external reviewer, with the documented library versions, obtained a different value for one run (seed 1400, spare capacity, no recruitment: rare R² -0.135 instead of the published -0.629) while the neighbouring run matched. The published values are reproduced bit for bit here (`tools/environment.txt`, `tools/environment_detail.txt`: Intel Xeon with AVX-512, MKL-backed PyTorch 2.14.1 CPU, one thread). A replay of all 110 runs with every module's initial weights perturbed by Gaussian noise of standard deviation 1e-6 (`src/replay_common_state.py`, `results/14b_16b_common_state/replay_perturbed.jsonl`) reproduces the reviewer's value for that run (-0.149). This demonstrates a sensitivity to the initial weights and supports a numerical explanation of the difference between machines, without identifying its precise cause: individual trajectories diverge under alarms and sampled routing (7 of 110 runs differ by more than 0.05 on availability, up to 0.318). The six preregistered statistical decisions are unchanged; the estimates vary (the U1 contrast on the rare skill moves from -0.002 to +0.098, both far from significance) (`replay_perturbed_comparison.txt`). Conclusions should be read at the level of the aggregate, not of single runs, and bitwise reproduction should be expected only on the same hardware and BLAS.

## Modular JEPA study (experiments 12 and 13, not in the manuscript)

Design note: [`design/modular_jepa.md`](design/modular_jepa.md), written after a review of the state of the art (MOSAIC, COMET, RIM/NPS, NEO, the JEPA line) and before any result. World: sequential states with five unnamed primitives that persist for a few steps then switch, plus eight noise channels per observation; a tiny JEPA (shared encoder, EMA target encoder) with competing predictors in representation space. Stage 0 (sanity, seed 0): representation-space prediction works and does not collapse (linear probe keeps the state at R² 0.998, drops the noise at 0.007, effective rank 7.9 of 8). Five test seeds per stage, so results are effect sizes with per-seed values; no test reaches a threshold.

### 12. Stage 1: competition and the prior over the winner (seeds 1000 to 1004)

| Variant | Module purity | Relative error, 2 to 3 steps after a switch to a held-out pair |
|---|---|---|
| competition, persistence prior (previous winner) | 0.84 | 0.05 |
| competition, trace prior (ours) | 0.84 | 0.05 |
| competition, state classifier prior (COMET) | 0.84 | 1.52 |
| attention assignment (RIM style) | 0.36 | 0.92 |
| oracle assignment | 1.00 | 0.01 |

Competition by local error replicates COMET and beats attention on purity (5/5). The "when" needs memory of the previous winner, not the current state: the classifier prior fails after a switch (5/5). Traces tie with plain persistence (+0.002), as expected in a world where the previous winner predicts itself.

### 13. Stage 2: a sixth primitive appears (seeds 1010 to 1014)

| Variant | Specialist captured the new primitive | Error on the new primitive | Interference on the old ones |
|---|---|---|---|
| competition, 5 modules | 5/5 | 0.68 | +0.041 |
| competition, 8 modules (over-provisioned) | 1/5 | 0.49 | +0.000 |
| competition + least-committed recruitment (5 active, reserves dormant) | 0/5 | 0.49 | +0.002 |

Recruitment provides the capability COMET states as open, instantiating a mechanism without degrading the others: against fixed five-module competition, lower error on the new primitive (5/5) and lower interference (5/5). Against competition merely over-provisioned with eight modules, it ties on both (differences of 0.002 and 0.001). No economic advantage can be claimed from this implementation: the predictions of all modules, reserves included, are computed before the inactive ones are masked, so a dormant reserve still costs its forward pass (pointed out by an external review on 2026-10-04). A regime with repeated new primitives and scarce spares remains to be tested.

## Continual dynamics on LeWorldModel's frozen representation (experiment 17, pilot, not in the manuscript)

Preregistration: `preregistrations/17_continual_lewm_pilot.md` (with the three development runs declared). Code: `kaggle/lewm-pilot/`. Results: `results/lewm_pilot/`. Setting: the fp32 embedding cache of the released LeWM for Two-rooms, predictors with LeWM's own architecture trained on it; regime A = original actions, regime B = actions rotated by 90 degrees (same images); stream P1 A (1500 steps of 4 episodes), P2 B with A at 2 % (1500 steps), P3 A (500 steps). Errors are relative 5-step latent prediction errors on 60 held-out episodes; "access" is the module chosen before observing the outcome.

| System | A end of P1 | A during B | B end of P2 | A end of P3 | B end of P3 | Forward passes |
|---|---|---|---|---|---|---|
| single predictor | 0.070 | 1.60 | 0.063 | 0.058 | 1.68 | 3500 |
| single + replay (20k windows) | 0.064 | 0.131 | 0.080 | 0.072 | 0.107 | 3500 |
| bank, persistence prior | 0.108 | 0.132 | 0.116 | 0.111 | 0.138 | 6982 |
| bank, trace prior + least-committed recruitment | 0.108 | 0.132 | 0.116 | 0.111 | 0.138 | 6982 |

Five seeds, standard deviations between 0.002 and 0.047 (largest for the forgetting single predictor); a fresh predictor trained on the P3 stream alone reaches 0.16 to 0.17 at the end of P3, so every system retains more of A than a fresh start. The banks behaved as designed on every seed: the A specialist froze before B, B triggered exactly one surprise and one activation, no recruitment, two active and two frozen modules at the end. The two banks are identical to the fourth decimal: with one module per regime, the prior has nothing to decide.

Verdict under the preregistered stop rule. Against replay, the banks are not better on any retention measure: A during B tie (+0.000, better in 2/5 seeds), B at the end worse (+0.031, 0/5), A at the end worse (+0.039, 0/5); they also cap A at the level where it froze (0.108 vs 0.064) and cost twice the forward passes. The trace and recruitment line therefore stops for continual world models, and this pilot is reported as a negative result: in this setting, a reservoir replay buffer of 20,000 windows, with no tuned parameter, preserves both dynamics at least as well as a bank of specialists with freezing and held-back reserves, whose thresholds had to be set on a development seed. The mechanism's one structural asset, instantiating a new predictor on surprise without touching the old one, worked, and was not worth more than replay. Not tested here: planning success under A and B (phase 2 of the design note), which remains the only measure that would turn these latent errors into a claim about control.

- **Development seeds.** Every test used seeds separate from the development seeds used to check code and tune opponents. What development revealed is stated in the preregistrations.
- **Protocol found after the fact.** While preparing publication, the files of a preregistered protocol (experiment 4) were found in the execution environment, from an earlier execution absent from the history of the conversation that produced the rest of this work. It had not been reported. Its code hashes match its preregistration; its main test had stopped at 15 of 20 seeds and was completed with the same code before any analysis. It is reported here in full, including the fact that it weakens an earlier claim.
- **Seed reuse.** The preregistrations of experiments 5 and 6 state that their seeds had never been used. Seeds 60 to 74 and 80 to 99 had in fact been used by experiment 4. No decision depended on those results, which were unknown at the time, and the code is deterministic: the tested mechanism gives bit-identical results on the 20 shared seeds in both runs. The statement was nevertheless false; corrections are attached to the translated preregistrations.
- **A second set of files from an unseen execution.** After the v1 manuscript, the preregistrations of experiments 8 and 9, their code, the completed test runs of experiment 8 and the beginning of a third, unpreregistered experiment were again found in the execution environment, from an execution absent from the conversation history. The quoted code hashes match, and the preregistrations were written before the test runs. Experiment 8 was analysed and experiment 9 was run according to their written protocols. The third experiment was not continued; its development-seed tuning is kept in `results/draft_regimes_dev/` and `src/exp_regimes.py`, marked as a draft and not analysed.
- **Cause of the "unseen executions", identified on 2 October.** When the chat interface reports a response as failed, the attempt's tools keep executing in the shared environment. The files found three times came from such attempts of the same turns, not from any external intervention. Experiments 12 and 13 were produced the same way: their preregistrations precede their test seeds, their code hashes match, and the seeds the cut attempts did not reach were completed with the same code.
- **A third set of files from an unseen execution, and an overwrite.** Experiment 10 (preregistration, code, complete test runs and an amended commit at 2026-10-02T02:50Z) was found in the execution environment while a second, independently written implementation of the same design was being started. That second implementation briefly overwrote the preregistration and code of experiment 10, which were restored byte for byte from the commit; it was then kept as experiment 10b on fresh seeds, with its three runs made before the discovery set apart.
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
python src/analysis_local_credit_b.py results/10_local_credit/10b_seeds_810_819.jsonl
python src/analysis_topk_local.py results/11_topk_local/topk_local.jsonl
python src/analysis_preservation.py results/14_preservation/preservation.jsonl
python src/analysis_savings.py results/15_savings/savings.jsonl
python src/exp_savings_b.py  # 15b; its summary is results/15b_savings/summary.txt
python src/analysis_common_state.py results/14b_16b_common_state/common_state.jsonl
python tools/check_provenance.py
```

Modular JEPA stages, from `src/jepa/`: `python stage0_sanity.py 0`, `python stage1_competition.py 1000,...,1004 ../../results/12_jepa_stage1/stage1.jsonl none,traces,classifier,attention,oracle,random,single`, `python stage2_lifelong.py 1010,...,1014 ../../results/13_jepa_stage2/stage2.jsonl competition5,competition8,recruit5`.

Experiments 8 and 9 (from `src/`): `python exp_topk.py 700,...,719 ../results/8_topk_moe/topk2.jsonl topk '{"k":2,"noise":0.3,"alpha":0.0}'` (and the top-1 configuration), `python exp_heterogeneous.py 720,...,739 ../results/9_heterogeneous/heterogeneous.jsonl aco_thresholds,aco_global_threshold,swucb:50:0.05,ducb:0.98:0.3`.

## Layout

- `src/`: experiment and analysis code (English).
- `results/<n>_<name>/`: raw results (JSON lines), summaries and figures; development and tuning runs are kept alongside.
- `preregistrations/`: English translations of the preregistrations, with corrections attached.
- `tools/`: provenance check and its report.
- `paper/`: manuscript (LaTeX, BibTeX, compiled `.bbl` and PDF).
- `original-fr/`: all files exactly as executed, in French, including smoke tests and a discarded, non-preregistered tuning run.

## Archived versions

The Git history of this repository was squashed into a single commit on 1 October 2026; from the tag `v2-draft` on, the history is linear and complete. Earlier states remain available on Software Heritage (first publication: `swh:1:snp:ec5e807533a26aef69001187d77b62911d3cb4c8`, 1 October 2026 06:07 UTC).

## License

Apache License 2.0, see [LICENSE](LICENSE).
