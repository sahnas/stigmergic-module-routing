"""
DRAFT, not preregistered, development seeds only, not analysed: regimes where the mechanism has a structural reason to win.

Production phase with frequent failures and a scarce reserve: 8 successive deaths, each
followed by the arrival of a single new module, then half a pass over the 17 tasks
(30 steps per block). The pool starts with 6 modules (5 primitives + 1 reserve) out of 14 slots.
Two regimes:
  homogeneous   : every module has 32 hidden units (as in experiments 1 to 7)
  heterogeneous : each module draws its hidden size in {4, 8, 16, 32, 64}, so a little-used
                  module may be little used because it is weak
Opponents: SW-UCB and D-UCB (re-tuned for these regimes on development seeds).
The tested mechanism (aco_thresholds) keeps its original hyperparameters.
"""
import json, sys, time
import numpy as np
import torch
import torch.nn as nn
import exp
import exp_dynamic as ed
import exp_bandits as eb
from exp import World, apply_task, mean_r2, K, D

M, M_INIT = 14, 6
exp.M = M; ed.M = M; ed.M_INIT = M_INIT; eb.M = M
N_DEATHS, STEPS_PER_BLOCK = 8, 30
SIZES = [4, 8, 16, 32, 64]


class HetUnit(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.lin = nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, h), nn.ReLU(), nn.Linear(h, D))
        self.dead = False

    def forward(self, x):
        if self.dead:
            return torch.zeros_like(x)
        return self.lin(x) + self.mlp(x)


def build(variant, lr):
    if variant == 'aco_thresholds':
        return ed.StigDyn(lr, 'aco', True)
    return eb.BanditDyn(lr, variant)


def run(variant, seed, regime, lr=3e-3, steps1=150):
    torch.manual_seed(seed)
    world = World(seed)
    model = build(variant, lr)
    model.seed(seed + 3)
    het_rng = np.random.default_rng(seed + 29)
    sizes = list(het_rng.choice(SIZES, size=M)) if regime == 'heterogeneous' else [32] * M
    model.units = [HetUnit(int(h)) for h in sizes]
    model.opt = torch.optim.Adam([p for u in model.units for p in u.parameters()], lr)
    recruits = []
    if hasattr(model, 'recruit'):
        orig = model.recruit
        def logged(s, u, path, _orig=orig):
            before = model.tau[s].copy()
            _orig(s, u, path)
            changed = np.flatnonzero(model.tau[s] != before)
            recruits.extend(int(sizes[r]) for r in changed)
        model.recruit = logged
    rng = np.random.default_rng(seed + 5)
    ev_rng = np.random.default_rng(seed + 17)
    tt = world.train_tasks
    t0 = time.time()

    def train_block(task, n):
        for _ in range(n):
            x, y = world.batch(task)
            model.train_step(task, x, y, world.var[task])

    for _ in range(2):
        for i in rng.permutation(len(tt)):
            train_block(tt[i], steps1)
    res = dict(variant=variant, seed=seed, regime=regime, train_r2_init=mean_r2(model, world, tt))
    curve = []
    for e in range(N_DEATHS):
        k = int(ev_rng.integers(K))
        model.units[model.greedy(k)].dead = True
        idx = np.where(~model.active)[0][:1]
        model.active[idx] = True
        for i in rng.permutation(len(tt)):
            train_block(tt[i], STEPS_PER_BLOCK)
            curve.append(mean_r2(model, world, tt, n=300))
    res['availability'] = float(np.mean(curve))
    res['train_r2_final'] = mean_r2(model, world, tt)
    res['zero_r2_final'] = mean_r2(model, world, world.held)
    res['recruit_sizes'] = recruits
    res['used_sizes'] = sorted({int(sizes[model.greedy(k)]) for k in range(K)})
    res['curve'] = curve
    res['secs'] = round(time.time() - t0, 1)
    return res


if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]
    out = sys.argv[2]
    variants = sys.argv[3].split(',')
    regimes = sys.argv[4].split(',')
    with open(out, 'a') as f:
        for s in seeds:
            for reg in regimes:
                for v in variants:
                    r = run(v, s, reg)
                    f.write(json.dumps(r) + '\n'); f.flush()
                    print(reg, v, s, round(r['train_r2_init'], 3), round(r['availability'], 3), round(r['train_r2_final'], 3), r['secs'], flush=True)
