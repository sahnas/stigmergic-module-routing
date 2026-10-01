"""
These testee : "coordination sans gradient".
Le routage ne recoit aucun gradient (decisions discretes, locales). Les modules, eux,
apprennent leur contenu par gradient le long du chemin emprunte.

Adversaire : le routage par simple recence (ema), le statu quo des bandits non stationnaires.
Plan factoriel 2 x 2 : regle d'oubli (ema / aco) x seuils par unite (non / oui).

Scenario "production" apres l'apprentissage initial (memes 17 taches, meme monde jouet) :
  pool de 10 emplacements, 7 modules presents au depart (5 primitives + 2 reserves).
  E1 : mort du module qui porte une primitive a
  E2 : arrivee de 2 modules neufs + mort du module qui porte une primitive b
  E3 : derive du monde, la primitive c change de definition (aucun module ne meurt)
  E4 : mort simultanee des modules qui portent deux primitives d et e
  Chaque evenement est suivi d'un passage sur les 17 taches (60 pas par bloc).
  Le systeme ne sait pas quels modules sont morts : un module mort reste selectionnable.

Mesure principale : disponibilite = R2 moyen sur toutes les taches d'entrainement,
  evalue apres chaque bloc de 60 pas pendant toute la phase de production (aire sous la courbe).
"""
import json, sys, time
import numpy as np
import torch
import exp
exp.M = 10
from exp import World, Stig, apply_task, mean_r2, K, D

M = 10
M_INIT = 7


class StigDyn(Stig):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.active = np.zeros(M, dtype=bool)
        self.active[:M_INIT] = True

    def _mask(self, v, exclude):
        v = v.copy()
        v[~self.active] = -np.inf
        if exclude is not None:
            v[exclude] = -np.inf
        return v

    def sample(self, s, exclude=None):
        logits = self._mask(self.tau[s] / self.T, exclude)
        p = np.exp(logits - logits.max()); p /= p.sum()
        return int(self.rng.choice(M, p=p))

    def greedy(self, s, exclude=None):
        return int(np.argmax(self._mask(self.tau[s], exclude)))

    def recruit(self, s, u, path):
        load = self.tau.sum(0).copy()
        load[path] = np.inf
        load[~self.active] = np.inf
        r = int(np.argmin(load))
        if not np.isfinite(load[r]):
            return
        self.tau[s, r] = max(self.tau[s, r], self.tau[s, u])
        self.qbar[r], self.qdev[r], self.uses[r] = 0.0, 0.0, 0
        self.cool[s] = self.cooldown_len
        self.alarms += 1


VARIANTS = {'ema': ('ema', False), 'ema_seuils': ('ema', True),
            'aco': ('aco', False), 'aco_seuils': ('aco', True)}


def drift(world, c, seed):
    g = np.random.default_rng(seed + 999)
    Q, _ = np.linalg.qr(g.normal(size=(D, D)))
    B = g.normal(size=(D, D)) * 1.5 / np.sqrt(D)
    world.prims[c] = (torch.tensor(Q, dtype=torch.float32), torch.tensor(B, dtype=torch.float32))
    gx = torch.Generator().manual_seed(seed + 77)
    for t in world.train_tasks + world.held:
        if c in t:
            world.var[t] = apply_task(world.prims, t, torch.randn(20000, D, generator=gx)).var(0).mean().item()


def run(variant, seed, lr=3e-3, steps1=150, steps2=60):
    torch.manual_seed(seed)
    world = World(seed)
    rule, seuils = VARIANTS[variant]
    model = StigDyn(lr, rule, seuils)
    model.seed(seed + 3)
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
    res = dict(variant=variant, seed=seed, train_r2_init=mean_r2(model, world, tt))

    prim_order = ev_rng.permutation(K)
    a, b, c, d, e = [int(p) for p in prim_order]
    events = [('E1', lambda: kill(a)),
              ('E2', lambda: (arrive(2), kill(b))),
              ('E3', lambda: drift(world, c, seed)),
              ('E4', lambda: (kill(d), kill(e)))]

    def kill(k):
        model.units[model.greedy(k)].dead = True

    def arrive(n):
        idx = np.where(~model.active)[0][:n]
        model.active[idx] = True

    curve, step = [], 0
    per_event = {}
    for name, ev in events:
        ev()
        vals = []
        for i in rng.permutation(len(tt)):
            train_block(tt[i], steps2)
            step += steps2
            r = mean_r2(model, world, tt, n=300)
            curve.append((step, r)); vals.append(r)
        per_event[name] = float(np.mean(vals))
    res['disponibilite'] = float(np.mean([r for _, r in curve]))
    res['par_evenement'] = per_event
    res['train_r2_final'] = mean_r2(model, world, tt)
    res['zero_r2_final'] = mean_r2(model, world, world.held)
    res['alarmes'] = model.alarms
    res['curve'] = curve
    res['secs'] = round(time.time() - t0, 1)
    return res


if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]
    out = sys.argv[2]
    variants = sys.argv[3].split(',') if len(sys.argv) > 3 else list(VARIANTS)
    with open(out, 'a') as f:
        for s in seeds:
            for v in variants:
                r = run(v, s)
                f.write(json.dumps(r) + '\n'); f.flush()
                print(v, s, round(r['train_r2_init'], 3), round(r['disponibilite'], 3),
                      {k: round(x, 3) for k, x in r['par_evenement'].items()},
                      round(r['train_r2_final'], 3), round(r['zero_r2_final'], 3), r['alarmes'], r['secs'], flush=True)
