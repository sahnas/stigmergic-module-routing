# Experiment 21: CEM mean versus best final candidate

Protocol: [`preregistrations/21_cem_final_selection.md`](../../preregistrations/21_cem_final_selection.md).

Frozen weights and original search; one change to the returned action plan.
The kernel runs six 8-task feasibility evaluations, then, only if the fixed
TwoRoom gate passes, 36 paired official-protocol evaluations on both tasks.
It trains no model. An implementation or input failure stops execution.

Local check:

```sh
python check_local.py --package /path/to/stable_worldmodel_0.1.1_source
```

Analysis after retrieving outputs:

```sh
python selection.py --analyze /path/to/retrieved_outputs
```

`complete.json` distinguishes `stopped_at_feasibility_gate` from `evaluated`.
An absent completion marker or `failure.json` means incomplete execution;
it must never be presented as a scientific failure of the hypothesis.

Registration and code are pushed before kernel submission. Results are
pending until retrieval and verification, even if a job has been submitted.

Kaggle version 1 stopped after its first search, in our extra diagnostic:
the pinned LeWM criterion cannot broadcast eight different goal embeddings
against four candidate plans in a single batch. The official solver uses
batch size one. The diagnostic now also scores one row at a time, still
four extra candidates per row after search in both conditions. The original
search, selector, data, models, gate and analysis are unchanged. Local and
Kaggle runtime checks now exercise the actual LeWM cost path with distinct
synthetic goals and short contexts. The failed run is archived under
[`version1`](../../results/lewm_kaggle/exp21_selection/version1/); it completed
no episode evaluation and produced no scientific verdict.
