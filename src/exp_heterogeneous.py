"""
Experiment 9: heterogeneous modules in the production scenario of exp_dynamic.py.

The 10 module slots receive different hidden sizes {2, 4, 8, 16, 32, 64, 2, 4, 8, 16},
in a seed-dependent order, so modules differ in what they can learn. Every variant of
a given seed gets exactly the same assignment. Nothing else changes: same world,
events, metrics and hyperparameters (no tuning for any variant; the bandits keep the
configurations selected in experiment 4).

Variants: aco_thresholds (individual thresholds), aco_global_threshold (single
collective threshold), swucb:50:0.05, ducb:0.98:0.3.

Usage: python exp_heterogeneous.py SEEDS OUT.jsonl VARIANTS
"""
import json, sys
import numpy as np
import torch
import torch.nn as nn
import exp
import exp_dynamic as ed

_Orig = ed.StigDyn
import exp_bandits as eb            # patches ed.StigDyn; restore before the next import
ed.StigDyn = _Orig
import exp_ablation as ea           # captures _Orig as its base class

SIZES = [2, 4, 8, 16, 32, 64, 2, 4, 8, 16]
_schedule = []


class HetUnit(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.lin = nn.Linear(exp.D, exp.D)
        self.mlp = nn.Sequential(nn.Linear(exp.D, h), nn.ReLU(), nn.Linear(h, exp.D))
        self.dead = False
        self.h = h

    def forward(self, x):
        if self.dead:
            return torch.zeros_like(x)
        return self.lin(x) + self.mlp(x)


def unit_factory():
    return HetUnit(_schedule.pop(0))


exp.Unit = unit_factory


def factory(lr, rule, thresholds=False, **kw):
    if rule.split(':')[0] in ('swucb', 'ducb', 'mucb'):
        return eb.BanditDyn(lr, rule, thresholds)
    if rule == 'global_threshold':
        return ea.GlobalThreshold(lr, 'aco', True)
    return _Orig(lr, rule, thresholds, **kw)


ed.StigDyn = factory


def run(variant, seed):
    _schedule.clear()
    _schedule.extend(int(h) for h in np.random.default_rng(seed + 31).permutation(SIZES))
    r = ed.run(variant, seed)
    r['sizes'] = [int(h) for h in np.random.default_rng(seed + 31).permutation(SIZES)]
    return r


if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]
    out = sys.argv[2]
    variants = sys.argv[3].split(',')
    with open(out, 'a') as f:
        for s in seeds:
            for v in variants:
                r = run(v, s)
                f.write(json.dumps(r) + '\n'); f.flush()
                print(v, s, round(r['train_r2_init'], 3), round(r['availability'], 3),
                      round(r['train_r2_final'], 3), r['secs'], flush=True)
