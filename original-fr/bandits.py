"""
Adversaires les plus forts pour la these "coordination sans gradient" :
bandits concus pour les environnements non stationnaires, un bandit par symbole,
les bras sont les modules, la recompense est la qualite q du lot.

  SW-UCB : UCB sur fenetre glissante (Garivier et Moulines, 2011)
  D-UCB  : UCB escompte (Garivier et Moulines, 2011)
  M-UCB  : UCB avec detection de rupture par fenetre, redemarrage du bandit (Cao et al., 2019)

Ils tournent dans exactement le meme scenario que exp_dynamique.run (meme monde,
memes evenements, memes graines) : on remplace seulement la fabrique de modeles.
"""
import sys, json, math
from collections import deque
import numpy as np
import exp_dynamique as ED

M = ED.M
_Orig = ED.StigDyn


class BanditDyn(_Orig):
    def __init__(self, lr, rule, seuils=False, **kw):
        super().__init__(lr, 'ema', False)
        parts = rule.split(':')
        self.kind = parts[0]
        K = self.tau.shape[0]
        self.t = np.zeros(K)
        self.cnt = np.zeros((K, M)); self.sm = np.zeros((K, M))
        if self.kind == 'swucb':
            self.window, self.c = int(parts[1]), float(parts[2])
            self.hist = [deque() for _ in range(K)]
        elif self.kind == 'ducb':
            self.gamma, self.c = float(parts[1]), float(parts[2])
        elif self.kind == 'mucb':
            self.w, self.b, self.c = int(parts[1]), float(parts[2]), float(parts[3])
            self.recent = [[deque(maxlen=self.w) for _ in range(M)] for _ in range(K)]
            self.restarts = 0
        else:
            raise ValueError(rule)

    def _cands(self, exclude):
        ok = self.active.copy()
        if exclude is not None:
            ok[exclude] = False
        return np.where(ok)[0]

    def sample(self, s, exclude=None):
        cands = self._cands(exclude)
        n = self.cnt[s, cands]
        untried = cands[n < 1e-9]
        if len(untried):
            return int(self.rng.choice(untried))
        T = n.sum() if self.kind == 'ducb' else (min(self.t[s], self.window) if self.kind == 'swucb' else n.sum())
        idx = self.sm[s, cands] / n + self.c * np.sqrt(np.log(max(T, 2.0)) / n)
        best = np.flatnonzero(idx == idx.max())
        return int(cands[self.rng.choice(best)])

    def greedy(self, s, exclude=None):
        cands = self._cands(exclude)
        n = self.cnt[s, cands]
        mean = np.where(n > 1e-9, self.sm[s, cands] / np.maximum(n, 1e-9), -1.0)
        return int(cands[int(np.argmax(mean))])

    def update_traces(self, task, path, q):
        for s, u in zip(task, path):
            self.t[s] += 1
            if self.kind == 'swucb':
                self.hist[s].append((u, q)); self.cnt[s, u] += 1; self.sm[s, u] += q
                if len(self.hist[s]) > self.window:
                    u0, q0 = self.hist[s].popleft(); self.cnt[s, u0] -= 1; self.sm[s, u0] -= q0
            elif self.kind == 'ducb':
                self.cnt[s] *= self.gamma; self.sm[s] *= self.gamma
                self.cnt[s, u] += 1; self.sm[s, u] += q
            else:
                self.cnt[s, u] += 1; self.sm[s, u] += q
                r = self.recent[s][u]; r.append(q)
                if len(r) == self.w:
                    h = self.w // 2
                    a = list(r)
                    if abs(np.mean(a[:h]) - np.mean(a[h:])) > self.b:
                        self.cnt[s] = 0; self.sm[s] = 0
                        for d in self.recent[s]:
                            d.clear()
                        self.restarts += 1


def factory(lr, rule, seuils=False, **kw):
    if rule.split(':')[0] in ('swucb', 'ducb', 'mucb'):
        return BanditDyn(lr, rule, seuils)
    return _Orig(lr, rule, seuils, **kw)


ED.StigDyn = factory

GRID = (
    [f'swucb:{w}:{c}' for w in (50, 200, 800) for c in (0.05, 0.3)] +
    [f'ducb:{g}:{c}' for g in (0.98, 0.995, 0.999) for c in (0.05, 0.3)] +
    [f'mucb:{w}:{b}:{c}' for w in (20, 60) for b in (0.2, 0.4) for c in (0.05, 0.3)]
)
for rule in GRID:
    ED.VARIANTS[rule] = (rule, False)


if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]
    out = sys.argv[2]
    variants = sys.argv[3].split(',')
    with open(out, 'a') as f:
        for s in seeds:
            for v in variants:
                r = ED.run(v, s)
                f.write(json.dumps(r) + '\n'); f.flush()
                print(v, s, round(r['train_r2_init'], 3), round(r['disponibilite'], 3),
                      round(r['train_r2_final'], 3), r['secs'], flush=True)
