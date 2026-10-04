"""
Experiment 16: reset the recruited module's weights, or keep them?

Experiment 15 found that a module trained for one function is a worse starting point for another than a fresh
module. The simplest fix is to reinitialise a module when it is recruited (its old function is lost anyway).
Two settings, same code paths as experiments 3 and 14, the only change being the recruitment rule:
  A. full production scenario of experiment 3 (4 events, 7 modules on 10 slots), availability
  B. experiment 14, learned then rare, no reserve, least-committed recruitment, availability and R2 after
Rules: 'keep' (current mechanism) and 'reset' (reinitialise parameters and optimizer state on recruitment).
Usage: python exp_reset.py SEEDS OUT.jsonl
"""
import json, sys, time
import numpy as np
import torch
import exp_dynamic as ed
import exp_preservation as ep
from exp import Unit


def reinit(model, r):
    fresh = Unit()
    model.units[r].load_state_dict(fresh.state_dict())
    for p in model.units[r].parameters():
        model.opt.state.pop(p, None)


class StigDynReset(ed.StigDyn):
    def recruit(self, s, u, path):
        before = self.alarms
        load = self.tau.sum(0).copy(); load[path] = np.inf; load[~self.active] = np.inf
        r = int(np.argmin(load))
        super().recruit(s, u, path)
        if self.alarms > before and np.isfinite(load[r]):
            reinit(self, r)


class StigPresReset(ep.StigPres):
    def recruit(self, s, u, path):
        n = len(self.recruits)
        super().recruit(s, u, path)
        if len(self.recruits) > n:
            reinit(self, self.recruits[-1])


def run_A(rule, seed):
    orig = ed.StigDyn
    if rule == 'reset':
        ed.StigDyn = StigDynReset
    try:
        r = ed.run('aco_thresholds', seed)
    finally:
        ed.StigDyn = orig
    r.pop('curve', None); r['setting'] = 'A'; r['rule'] = rule
    return r


def run_B(rule, seed):
    orig = ep.StigPres
    if rule == 'reset':
        ep.StigPres = StigPresReset
    try:
        r = ep.run('learned_then_rare', 0, 'least_committed', seed)
    finally:
        ep.StigPres = orig
    r.pop('recruits', None); r['setting'] = 'B'; r['rule'] = rule
    return r


if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]
    out = sys.argv[2]
    with open(out, 'a') as f:
        for s in seeds:
            for setting, fn in [('A', run_A), ('B', run_B)]:
                for rule in ['keep', 'reset']:
                    r = fn(rule, s)
                    f.write(json.dumps(r) + '\n'); f.flush()
                    extra = {k: round(v, 3) for k, v in r.get('per_event', {}).items()} if setting == 'A' else {k: round(r[k], 3) for k in ('rare_r2_after', 'other_r2_after')}
                    print(setting, rule, s, 'availability', round(r['availability'], 3), 'alarms', r['alarms'], extra, flush=True)
