"""
Ablation: what carries the result in evaporation + thresholds?
Same production scenario (exp_dynamic.py), no hyperparameter changed.

  aco_thresholds         : reference. Alarm threshold specific to each unit, recruitment of the least busy unit.
  aco_thresholds_random  : individual thresholds, but recruitment of a random unit (among those present, off the path).
  aco_global_threshold   : recruitment of the least busy unit, but a single collective threshold (mean and deviation
                           of the quality of the whole colony) instead of one threshold per unit.
"""
import json, sys
import numpy as np
import exp_dynamic as ed
from exp_dynamic import M

_Base = ed.StigDyn


class RandomRecruit(_Base):
    def recruit(self, s, u, path):
        cand = [r for r in range(M) if self.active[r] and r not in path]
        if not cand:
            return
        r = int(self.rng.choice(cand))
        self.tau[s, r] = max(self.tau[s, r], self.tau[s, u])
        self.qbar[r], self.qdev[r], self.uses[r] = 0.0, 0.0, 0
        self.cool[s] = self.cooldown_len
        self.alarms += 1


class GlobalThreshold(_Base):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.g_bar, self.g_dev, self.g_n = 0.0, 0.0, 0

    def check_alarms(self, task, path, q):
        for s, u in zip(task, path):
            if self.uses[u] >= self.min_uses and self.g_n >= self.min_uses and self.cool[s] == 0:
                if q < self.g_bar - self.kappa * max(self.g_dev, 0.02):
                    self.recruit(s, u, path)
            a = 0.05
            if self.g_n == 0:
                self.g_bar = q
            self.g_dev = (1 - a) * self.g_dev + a * abs(q - self.g_bar)
            self.g_bar = (1 - a) * self.g_bar + a * q
            self.g_n += 1
            self.uses[u] += 1


def factory(lr, rule, thresholds):
    if rule == 'random_recruit':
        return RandomRecruit(lr, 'aco', True)
    if rule == 'global_threshold':
        return GlobalThreshold(lr, 'aco', True)
    return _Base(lr, rule, thresholds)


ed.StigDyn = factory
ed.VARIANTS['aco_thresholds_random'] = ('random_recruit', True)
ed.VARIANTS['aco_global_threshold'] = ('global_threshold', True)

if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]
    out = sys.argv[2]
    variants = sys.argv[3].split(',')
    with open(out, 'a') as f:
        for s in seeds:
            for v in variants:
                r = ed.run(v, s)
                f.write(json.dumps(r) + '\n'); f.flush()
                print(v, s, round(r['train_r2_init'], 3), round(r['availability'], 3), r['alarms'], r['secs'], flush=True)
