# Preregistration: experiment 17, continual dynamics on LeWM's frozen representation, pilot (5 seeds)

Written 2026-10-05T06:55Z, before any run on the pilot seeds.
Code: kaggle/lewm-pilot/pilot_lib.py (sha256 096dcbad729c085f), kaggle/lewm-pilot/pilot.py (sha256 5df6f0dd9a6ea5b3); design/kaggle_continual_lewm.md.
Data: fp32 embedding cache of the released LeWM for Two-rooms (quentinll/lewm-tworooms, encoder and projector
frozen), windows aligned with the Two-rooms config (history 3, frameskip 5, action blocks of 5), episode-level
split (10 % test episodes, per seed), actions z-scored on training frames. Regime A: original actions.
Regime B: actions rotated by 90 degrees before normalisation (same images, other action-to-transition mapping).

Stream: steps of 4 consecutive episodes; P1 1500 steps A; P2 1500 steps B with A at 2 % per episode; P3 500
steps A. Predictor architecture: LeWM's own (action encoder, predictor, pred_proj) at random init; AdamW
lr 3e-4, weight decay 1e-3, gradient clipping 1, one gradient step per module per step on its assigned windows.
Systems: single; single_replay (reservoir buffer of 20,000 windows, replayed one for one at every step);
bank_persistence (K = 4, one active at start, hindsight winner per window, persistence prior, freezing when the
module's EMA relative error stays below 0.1 for 20 steps, surprise when the best active module's step error
exceeds 4 times its EMA and 0.3 after a 200-step warm-up and outside a 100-step cooldown following any activation
or recruitment, reserves activated first, then random recruitment among unfrozen active modules);
bank_traces (same, with a trace prior keyed by the previous winner and least-committed recruitment).

Development history, declared: seed 100 was run three times. Run 1 (600/900/300 steps of one episode, absolute
surprise threshold 0.3, freeze 0.02): acquisition budget too small and surprise firing during acquisition, so the
banks split the data across all modules from the start. Run 2 (4 episodes per step, relative surprise after
warm-up, freeze 0.05): A acquired at 0.07 by every system; the banks then consumed the A specialist, because the
freeze threshold was never reached and surprises cascaded while the activated reserve was still learning B.
Run 3 (freeze 0.1, cooldown 100): the A specialist froze before B, one surprise activated one reserve, no
recruitment; results on seed 100: single A 0.070 then 1.54 during B then 0.062 after return, B 1.64 at the end;
replay A 0.14 during B, B 0.11 at the end; banks A 0.136 during B, B 0.13 at the end, but A capped at 0.115 by
the early freeze; bank_traces identical to bank_persistence. Asymmetry declared: the banks' thresholds were set on
this development seed; the replay buffer size was not tuned. Results of run 3 in results/lewm_pilot/.

Pilot seeds: 0 to 4 (5 seeds; descriptive, no test can reach a threshold). Measures at the end of each phase and
every 100 steps, on 60 held-out episodes per regime: access error (module chosen before observing, sequential
within the episode), content error (best module in hindsight), both for A and B; in P3, a fresh predictor
trained on the same P3 stream; forward passes per system; for the banks, active, frozen, surprises, activations,
recruitments.

Expectations written before the pilot: single forgets A during B and B after A's return; replay retains both;
the banks retain A during B at the level where it froze, pay that early freeze on A's accuracy, and retain B
after A's return; bank_traces and bank_persistence coincide (one module per regime leaves the prior nothing to
decide); no bank beats replay on A during B. Stop rule, fixed in advance: if the banks are not better than
replay on any of the three retention measures (A during B, B after A's return, A at the end), the trace and
recruitment line stops for continual world models and the pilot is reported as a negative result; the
confirmatory sample size is then not needed. If a bank is better on at least one measure by more than the
pilot's between-seed spread, a confirmatory run on new seeds is sized and preregistered.


Note added 2026-10-05 after the run: pilot.py was changed after this preregistration was written, by one line, to
make the preregistered 5-seed mode the default (sha256 of the run version: ac410a0fccaea4d4). pilot_lib.py, which
holds the whole protocol, is unchanged: the hash logged by the kernel at run time (results/lewm_pilot/pilot_seeds_0_4.log)
is 096dcbad729c085f, the one quoted above.
