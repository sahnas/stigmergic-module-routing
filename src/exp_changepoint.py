"""
Strong opponent for the "gradient-free coordination" thesis:
non-stationary bandit with a sliding window + change detection (M-UCB / CUSUM-UCB family).

For each symbol, a bandit over the modules:
  - statistics over the `win` most recent rewards (sliding window),
  - change detection: if the mean of the h most recent rewards drops by more than b
    compared with the h previous ones, the history of that arm is reset,
  - selection: UCB (mean + c * sqrt(log t / n), never-tried arms first)
    or softmax over the means (same temperature as the tested mechanism).

The scenario (events, seeds, metrics) is the one of exp_dynamic.py, unchanged:
only the model factory is replaced.
The opponent is tuned on development seeds; the tested mechanism is not.
"""
import json, sys
import numpy as np
import exp_dynamic as ed
from exp_dynamic import M, K

_Base = ed.StigDyn
CFG = {}


class ChangePoint(_Base):
    def __init__(self, lr, sel='ucb', c=0.1, h=20, b=0.3, win=300, T=0.05):
        super().__init__(lr, 'ema', False)
        self.sel, self.c, self.h, self.b, self.win, self.Tb = sel, c, h, b, win, T
        self.hist = [[[] for _ in range(M)] for _ in range(K)]
        self.sum = np.zeros((K, M)); self.n = np.zeros((K, M))
        self.t = np.zeros(K)
        self.resets = 0

    def _mean(self, s):
        return np.where(self.n[s] > 0, self.sum[s] / np.maximum(self.n[s], 1), np.nan)

    def _valid(self, exclude):
        ok = self.active.copy()
        if exclude is not None:
            ok[exclude] = False
        return ok

    def sample(self, s, exclude=None):
        ok = self._valid(exclude)
        mean = self._mean(s)
        if self.sel == 'ucb':
            bonus = self.c * np.sqrt(np.log(self.t[s] + 2) / np.maximum(self.n[s], 1))
            idx = np.where(self.n[s] > 0, mean + bonus, np.inf)
            idx[~ok] = -np.inf
            best = np.flatnonzero(idx == idx.max())
            return int(self.rng.choice(best))
        v = np.where(self.n[s] > 0, mean, 0.0) / self.Tb
        v[~ok] = -np.inf
        p = np.exp(v - v.max()); p /= p.sum()
        return int(self.rng.choice(M, p=p))

    def greedy(self, s, exclude=None):
        ok = self._valid(exclude)
        v = np.where(self.n[s] > 0, self._mean(s), -1.0)
        v[~ok] = -np.inf
        return int(np.argmax(v))

    def update_traces(self, task, path, q):
        for s, u in zip(task, path):
            H = self.hist[s][u]
            H.append(q); self.sum[s, u] += q; self.n[s, u] += 1
            if len(H) > self.win:
                self.sum[s, u] -= H.pop(0); self.n[s, u] -= 1
            self.t[s] += 1
            if len(H) >= 2 * self.h:
                old = np.mean(H[-2 * self.h:-self.h]); new = np.mean(H[-self.h:])
                if old - new > self.b:
                    H.clear(); self.sum[s, u] = 0.0; self.n[s, u] = 0
                    self.resets += 1


def factory(lr, rule, thresholds):
    if rule == 'changepoint':
        return ChangePoint(lr, **CFG)
    return _Base(lr, rule, thresholds)


ed.StigDyn = factory
ed.VARIANTS['changepoint'] = ('changepoint', False)

if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]
    out = sys.argv[2]
    variants = sys.argv[3].split(',')
    CFG.update(json.loads(sys.argv[4]) if len(sys.argv) > 4 else {})
    with open(out, 'a') as f:
        for s in seeds:
            for v in variants:
                r = ed.run(v, s)
                r['cfg'] = CFG if v == 'changepoint' else None
                f.write(json.dumps(r) + '\n'); f.flush()
                print(v, s, json.dumps(CFG) if v == 'changepoint' else '', round(r['train_r2_init'], 3),
                      round(r['availability'], 3), round(r['train_r2_final'], 3), r['secs'], flush=True)
