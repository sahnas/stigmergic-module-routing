# Design note: continual dynamics on LeWorldModel's frozen representation (Kaggle)

Written 2026-10-04, before any experiment. Follows the roadmap of 2026-10-03 (section 3) and the lessons of
experiments 14b and 16b (protection comes from held-back capacity, not from a rule).

## Setting

- Representation: the released LeWM for Two-rooms (quentinll/lewm-tworooms), encoder and projector frozen.
  Embeddings (192-d CLS projection) are precomputed once for all 920,809 frames of the Two-rooms dataset,
  with LeWM's own preprocessing (ImageNet normalisation, 224 px). All later experiments run on these
  embeddings, on CPU or a small GPU. The encoder saw regime A only (it was trained on the original Two-rooms
  data); this is documented as the continual-learning claim requires.
- Predictors: small networks mapping a history of 3 embeddings and the (z-scored) action to the next embedding,
  as LeWM's predictor does; trained by gradient on precomputed embeddings. No EMA target is needed since the
  representation is frozen.
- Dynamics A: the original Two-rooms transitions. Dynamics B: the same observations with a remapped action
  (a fixed rotation of the action vector), so that the visual context is identical and only the
  action-to-transition mapping changes; a regime can then be inferred only from prediction errors and history,
  never from appearance. Regime identities are used for evaluation only, never given to the system.

## Scenario (one stream, no labels)

1. Learn A until acquired (criterion fixed in the preregistration: held-out one-step error on A below a
   threshold set from the single-predictor baseline).
2. Make A rare (sampling weight 0.02) while continuing the stream.
3. Introduce B.
4. Bring A back at its original frequency.

A rare-from-the-start variant is kept separate (acquisition, not loss).

## Systems compared, at equal total predictor capacity and equal data and update budgets

- a single predictor trained sequentially (expected catastrophic forgetting);
- a single predictor with a small replay buffer (the standard remedy; the local-forgetting replay of 2023 is
  the reference to read first);
- a bank of predictors with hindsight competition by error and a persistence prior, with explicit
  preservation: specialists frozen once acquired, reserves held back until a surprise (the configuration that
  protected in 14b);
- the same bank with trace assignment and least-committed recruitment (ours), with and without held-back
  reserves.
Reserve cost is measured honestly: a reserve is excluded from forward computation until activated.

## Measures, kept separate

- Content forgetting: an oracle diagnostic evaluates every predictor on held-out A transitions after B
  (does any available module still predict A?).
- Access forgetting: the predictor chosen before observing the outcome (from history and errors only) and its
  error on held-out A transitions; the hindsight winner is reported separately and never substituted.
- Relearning: when A returns, steps to recover the acquisition criterion, against a fresh predictor and
  against the single sequential predictor.
- Planning (phase 2, needs the environment): success rate on Two-rooms under A and under B with each system,
  through LeWM's planner with the bank's predictor substituted.

## Order of work

0. Precompute embeddings (one GPU session); sanity: a single predictor on precomputed embeddings reaches an
   error comparable to LeWM's predictor on held-out A transitions.
1. Preregister and run the scenario with the systems above; 5 seeds minimum, more if cheap.
2. Planning evaluation for the two best systems.
