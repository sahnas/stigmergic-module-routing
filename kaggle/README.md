# LeWorldModel on free Kaggle hardware

Scripts pushed to Kaggle with the API (`kaggle kernels push -p <folder>`), run on Kaggle's free tier (T4 GPU,
4 CPUs, 20 GB disk, 12-hour sessions). Each folder holds the script and its kernel metadata; the kernels are
private under the account hassanakaou. Logs kept in `results/lewm_kaggle/`.

| Kernel | What it does | Outcome |
|---|---|---|
| lewm-smoke | GPU, internet, install check | Both need phone verification on Kaggle; dataset sizes: Two-rooms 3.4 GB archive, Push-T 13.1 GB, Reacher 23.8 GB, Cube 46.2 GB |
| lewm-data-tworoom | Downloads and extracts the Two-rooms dataset as a reusable kernel output | `tworoom.h5`, 12.78 GB, 920,809 frames of 224x224x3, 10,000 episodes |
| lewm-check | Verifies the mounted dataset | Found that a kernel output is not mounted until Kaggle has finished processing it |
| lewm-timing | Installs `stable-worldmodel[train,env]` and trains 60 batches of the official config | Two fixes needed: datasets live under `$STABLEWM_HOME/datasets/`, and `hdf5plugin` must be installed or the HDF5 format is silently unregistered. 1.12 it/s at batch 128, 16-mixed, on a T4. The official config (100 epochs over about 830,000 clips) would take about 165 h at that rate |
| lewm-eval | Planning evaluation on Two-rooms with the released weights | `policy=quentinll/lewm-tworooms` (the current loader takes a HF repo id; the README's `_object.ckpt` conversion is obsolete): success rate 86.0 over 50 episodes (exact binomial 95 % CI about 73 to 94 %); random policy 24.0. The paper reports about 87 %. Note: `eval.py` overrides `world.max_episode_steps` to twice `eval_budget` (100 with the Two-rooms config), so the `max_episode_steps=60` passed on the command line was not the effective limit; the resolved config was not kept and should be in future runs |
| lewm-embed | Precomputes the frozen encoder's 192-d embeddings for all frames, with LeWM's preprocessing | For the continual-dynamics study (`design/kaggle_continual_lewm.md`) |

The README corrections were submitted upstream: https://github.com/lucas-maes/le-wm/pull/106 and
https://github.com/galilai-group/stable-worldmodel/issues/339. Independent reproductions of LeWM already exist
(arXiv 2608.10145 on Two-rooms, and the Qantara suite), which is why no replication report is planned here.
