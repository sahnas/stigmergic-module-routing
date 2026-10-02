# Design note: modular JEPA with trace-based assignment (not yet an experiment)

Written 2026-10-02. This is a design, kept here so that any later experiment can be checked against what was
intended before results existed. Nothing below has been run on test seeds.

## Why

Experiments 10 and 11 showed that, once each module has its own local error, trace routing finds the shared
structure across opaque tasks with a clean decomposition and little interference. The oracle that supplied the
local error corresponds, in a sequential world, to observing the intermediate states. A JEPA module predicts the
next observation in a representation space and therefore has its own local error by construction. The question
becomes: do JEPA modules coordinated by traces discover the primitives of a world without labels, and compose
them on new sequences?

## World (toy, CPU)

- State s_t in R^8, five fixed nonlinear primitives P_k as before. A trajectory applies a primitive per step,
  s_{t+1} = P_{k_t}(s_t), with k_t drawn per step; primitives are never named.
- Observation o_t = [s_t, n_t], where n_t in R^8 is fresh noise at every step. The noise makes prediction in
  observation space impossible to complete, which is what representation-space prediction is for.
- Training sequences use an allowed set of consecutive pairs (k_t, k_{t+1}); held-out pairs are never seen in
  sequence, as in the compositional split of experiments 1 to 11.

## Modules

- A shared encoder f (small MLP) and a target encoder f_bar updated as an exponential moving average of f, as in
  I-JEPA; the target encoder is the common representation space in which modules communicate.
- M predictors g_m (small MLPs) in representation space: g_m(f(o_t)) predicts f_bar(o_{t+1}).
- Local error of module m on a transition: the distance between g_m(f(o_t)) and f_bar(o_{t+1}) (stop-gradient
  on the target). Anti-collapse: EMA target plus a variance term on f(o) (VICReg-style) if collapse is observed
  in stage 1.

## State of the art consulted before fixing the design (2026-10-02)

- MOSAIC (Wolpert and Kawato, 1998; Haruno, Wolpert and Kawato, NeurIPS 1998): paired forward models compete
  through a responsibility signal, a softmax of their prediction errors; a responsibility predictor supplies a
  prior from context. This is the classical form of assignment by local prediction error, with a context prior.
- Competition of experts (Parascandolo et al., 2018) and COMET (Lei, Nolte, Schoelkopf and Posner, 2024):
  winner-takes-all gradient allocation to the mechanism with the lowest prediction error makes recognisable,
  reusable mechanisms emerge; two stabilisers are needed in practice (a warm-start phase with gradients to all
  mechanisms, and selection over a time horizon rather than per step). The "when" is then a classifier trained
  on winner labels. COMET states as an open problem that it cannot instantiate new mechanisms when new data
  arrive.
- Neural Production Systems and Recurrent Independent Mechanisms (Goyal et al., 2021): assignment by
  attention learned jointly with the mechanisms; COMET shows this fails to disentangle mechanisms across
  environments with varying dynamics.
- NEO / Learning to Theorize (Baek et al., ICML 2026): mechanisms as a learned discrete codebook composed into
  programs, with a minimum-description-length objective; a different formalism for the same goal of reusable
  primitives.
- JEPA line: I-JEPA (EMA target encoder), LeJEPA (a principled regulariser replacing the EMA heuristics),
  LeWorldModel (stable end-to-end JEPA world model from pixels), Causal-JEPA (object-level latent masking,
  with LeCun among the authors, ICML 2026). None of them addresses the assignment of competing predictors.

## Assignment (the part under study), fixed after the review above

- Competition: on each transition, modules are compared on their local error in representation space; the
  winner learns (winner-takes-all), with COMET's two stabilisers (warm-start, selection over a short horizon).
  This is the state of the art and is not the object of study.
- The "when" is where the study sits. The world has sticky regimes (a primitive persists for a random number
  of steps, mean 5, then switches according to the allowed pairs), so that the previous winner carries
  information. Four priors over the winner are compared, combined with the local errors as in MOSAIC
  (responsibility proportional to prior times exp(-error / sigma)):
  none (pure competition, COMET's competition phase); a classifier trained on winner labels (COMET's
  composition module); attention learned jointly (RIM / NPS style); and traces keyed by the previous winner,
  with solicitation-triggered evaporation (ours).
- Lifelong discovery, COMET's stated open problem: when no module explains the transition (surprise, all
  errors above each module's own threshold), recruit the least committed module. Test: a sixth primitive
  appears after training; compare plain competition with spare modules against competition plus
  least-committed recruitment, on purity and on interference with the existing mechanisms.
- Baselines: a single predictor; the oracle assignment by the true primitive (upper bound); random assignment.

## Measures

- Latent prediction error on held-out transitions (lower is better), reported against the oracle upper bound.
- Primitive discovery: purity of the winner-to-primitive mapping (COMET's disentanglement matrix), and the
  number of distinct modules used.
- Composition: prediction error right after a switch to a held-out pair (a regime change never seen), and
  two-step prediction on sequences made of held-out pairs.
- Lifelong: after the sixth primitive appears, purity for the new primitive and interference on the old ones.
- Collapse diagnostics: variance and effective rank of f(o) over a batch.

## Stages, each preregistered before its test seeds

0. Sanity (done 2026-10-02, not a hypothesis test): a single JEPA predictor with the true primitive given as
   input learns the dynamics in representation space (relative latent error 0.018), keeps the state (linear
   probe R2 0.998), drops the noise (R2 0.007) and does not collapse (effective rank 7.9 of 8).
1. Discovery and the "when": competition with the four priors, on purity, prediction error and switches to
   held-out pairs.
2. Lifelong: the sixth primitive, with and without least-committed recruitment.

## What would be ours, stated honestly

Competition by local error is MOSAIC and COMET; representation-space prediction is JEPA. What is ours is the
trace prior with solicitation-triggered forgetting as the "when", and least-committed recruitment as the
answer to COMET's open problem of instantiating new mechanisms, both compared against the existing
alternatives under preregistration.

## Risks, stated in advance

Collapse of representations; a purity metric that rewards trivial solutions (one module per everything if the
noise dominates); the shared target encoder reintroducing a shared latent whose forgetting is uncontrolled;
heavy prior work on competing predictors (Recurrent Independent Mechanisms, mixtures of dynamics models), so the
only possible novelty is the assignment rule and its comparison.
