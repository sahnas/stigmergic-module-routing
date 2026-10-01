"""
Tested thesis: "gradient-free coordination".
The routing receives no gradient (discrete, local decisions). The modules themselves
learn their content by gradient along the path taken.

Opponent: routing by simple recency (ema), the status quo of non-stationary bandits.
2 x 2 factorial design: forgetting rule (ema / aco) x per-unit thresholds (no / yes).

"Production" scenario after initial learning (same 17 tasks, same toy world):
  pool of 10 slots, 7 modules present at the start (5 primitives + 2 reserves).
  E1: death of the module carrying a primitive a
  E2: arrival of 2 new modules + death of the module carrying a primitive b
  E3: drift of the world, primitive c changes definition (no module dies)
  E4: simultaneous death of the modules carrying two primitives d and e
  Each event is followed by one pass over the 17 tasks (60 steps per block).
  The system does not know which modules are dead: a dead module remains selectable.

Main measure: availability = mean R2 over all training tasks,
  evaluated after each 60-step block throughout the production phase (area under the curve).
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


VARIANTS = {'ema': ('ema', False), 'ema_thresholds': ('ema', True),
            'aco': ('aco', False), 'aco_thresholds': ('aco', True)}


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
    rule, thresholds = VARIANTS[variant]
    model = StigDyn(lr, rule, thresholds)
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
    res['availability'] = float(np.mean([r for _, r in curve]))
    res['per_event'] = per_event
    res['train_r2_final'] = mean_r2(model, world, tt)
    res['zero_r2_final'] = mean_r2(model, world, world.held)
    res['alarms'] = model.alarms
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
                print(v, s, round(r['train_r2_init'], 3), round(r['availability'], 3),
                      {k: round(x, 3) for k, x in r['per_event'].items()},
                      round(r['train_r2_final'], 3), round(r['zero_r2_final'], 3), r['alarms'], r['secs'], flush=True)
