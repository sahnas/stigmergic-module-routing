"""
Ablation : qu'est-ce qui porte le resultat dans evaporation + seuils ?
Scenario de production identique (exp_dynamique.py), aucun hyperparametre modifie.

  aco_seuils         : reference. Seuil d'alarme propre a chaque unite, recrutement de l'unite la moins occupee.
  aco_seuils_hasard  : seuils individuels, mais recrutement d'une unite au hasard (parmi les presentes hors chemin).
  aco_seuil_global   : recrutement du moins occupe, mais un seul seuil collectif (moyenne et ecart
                       de qualite de toute la colonie) au lieu d'un seuil par unite.
"""
import json, sys
import numpy as np
import exp_dynamique as ed
from exp_dynamique import M

_Base = ed.StigDyn


class Hasard(_Base):
    def recruit(self, s, u, path):
        cand = [r for r in range(M) if self.active[r] and r not in path]
        if not cand:
            return
        r = int(self.rng.choice(cand))
        self.tau[s, r] = max(self.tau[s, r], self.tau[s, u])
        self.qbar[r], self.qdev[r], self.uses[r] = 0.0, 0.0, 0
        self.cool[s] = self.cooldown_len
        self.alarms += 1


class Global(_Base):
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


def factory(lr, rule, seuils):
    if rule == 'hasard':
        return Hasard(lr, 'aco', True)
    if rule == 'global_seuil':
        return Global(lr, 'aco', True)
    return _Base(lr, rule, seuils)


ed.StigDyn = factory
ed.VARIANTS['aco_seuils_hasard'] = ('hasard', True)
ed.VARIANTS['aco_seuil_global'] = ('global_seuil', True)

if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]
    out = sys.argv[2]
    variants = sys.argv[3].split(',')
    with open(out, 'a') as f:
        for s in seeds:
            for v in variants:
                r = ed.run(v, s)
                f.write(json.dumps(r) + '\n'); f.flush()
                print(v, s, round(r['train_r2_init'], 3), round(r['disponibilite'], 3), r['alarmes'], r['secs'], flush=True)
