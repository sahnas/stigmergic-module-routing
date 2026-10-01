"""
Step 2: remove the symbols.

Each task is a composition of two primitives, y = P_j(P_i(x)), but the system only receives
an opaque identifier (0..19). It does not know which primitives make up the task. It only
knows that a path has two steps (two slots). To succeed with 8 modules and 12 tasks, it must
discover that tasks share primitives and reuse the same modules.

Phase A: interleaved learning on the 12 known tasks (8000 steps, one random task per step).
Phase B: for each of the 8 never-seen compositions, start again from the same state (copy) and
  learn the new task for 200 steps. The new task is evaluated every 20 steps.
  Main measure: learning speed = mean R2 over these 10 evaluations (area under the curve).
  Secondary measure: interference = drop of the mean R2 on the 12 old tasks after these 200 steps.

Variants:
  mono           : monolithic MLP, x + one-hot task identifier
  soft           : modules + router learned by gradient (softmax mixture) per (task, slot)
  rl             : modules + reinforcement-learning router (REINFORCE with baseline), Routing Networks style
  aco_thresholds : modules + evaporating traces + thresholds + recruitment of the least busy (unchanged)
"""
import copy, json, sys, time
import numpy as np
import torch
import torch.nn as nn
from exp import World, Unit, Stig, Soft, apply_task, D, M, BATCH

NT = 20          # possible task identifiers
PA_STEPS = 8000
PB_STEPS = 200
EVAL_EVERY = 20


def pseudo(tid):
    return (2 * tid, 2 * tid + 1)   # slots 1 and 2 of task tid


class MonoT:
    def __init__(self, lr):
        self.net = nn.Sequential(nn.Linear(D + NT, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, D))
        self.opt = torch.optim.Adam(self.net.parameters(), lr)

    def fwd(self, tid, x):
        c = torch.zeros(x.shape[0], NT); c[:, tid] = 1
        return self.net(torch.cat([x, c], 1))

    def train_step(self, tid, x, y, var):
        loss = ((self.fwd(tid, x) - y) ** 2).mean()
        self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()

    def predict(self, tid, x):
        with torch.no_grad():
            return self.fwd(tid, x)


class SoftT(Soft):
    def __init__(self, lr):
        super().__init__(lr)
        self.R = nn.Parameter(torch.randn(2 * NT, M) * 0.01)
        self.opt = torch.optim.Adam([p for u in self.units for p in u.parameters()] + [self.R], lr)

    def train_step(self, tid, x, y, var):
        super().train_step(pseudo(tid), x, y, var)

    def predict(self, tid, x):
        return super().predict(pseudo(tid), x)


def _resize(st):
    st.tau = np.full((2 * NT, M), 0.0)
    st.cnt = np.ones((2 * NT, M))
    st.cool = np.zeros(2 * NT, dtype=int)


class StigT(Stig):
    def __init__(self, lr):
        super().__init__(lr, 'aco', True)
        _resize(self)

    def train_step(self, tid, x, y, var):
        super().train_step(pseudo(tid), x, y, var)

    def predict(self, tid, x):
        return super().predict(pseudo(tid), x)


class RLT(Stig):
    """Reinforcement-learning router: logits per (task, slot), REINFORCE with a per-slot baseline."""
    def __init__(self, lr, lr_pg=0.5, T=1.0):
        super().__init__(lr, 'ema', False, T=T)
        _resize(self)
        self.lr_pg = lr_pg
        self.base = np.zeros(2 * NT)

    def _p(self, s, exclude=None):
        v = self.tau[s].copy()
        if exclude is not None:
            v[exclude] = -np.inf
        p = np.exp(v - v.max()); return p / p.sum()

    def sample(self, s, exclude=None):
        return int(self.rng.choice(M, p=self._p(s, exclude)))

    def update_traces(self, task, path, q):
        excl = None
        for s, u in zip(task, path):
            p = self._p(s, excl)
            adv = q - self.base[s]
            g = -p; g[u] += 1.0
            self.tau[s] += self.lr_pg * adv * g
            self.base[s] = 0.9 * self.base[s] + 0.1 * q
            excl = u

    def train_step(self, tid, x, y, var):
        super().train_step(pseudo(tid), x, y, var)

    def predict(self, tid, x):
        return super().predict(pseudo(tid), x)


def build(variant, lr, cfg):
    if variant == 'mono':
        return MonoT(lr)
    if variant == 'soft':
        return SoftT(lr)
    if variant == 'rl':
        return RLT(lr, **cfg)
    if variant == 'aco_thresholds':
        return StigT(lr)
    raise ValueError(variant)


def r2(model, world, task, tid, n=300):
    x = world.test_x[task][:n]
    y = apply_task(world.prims, task, x)
    return 1.0 - ((model.predict(tid, x) - y) ** 2).mean().item() / world.var[task]


def alignment(model, world, n=500):
    """For each primitive, best R2 of a module that approximates it directly (descriptive)."""
    if not hasattr(model, 'units'):
        return None
    g = torch.Generator().manual_seed(1)
    x = torch.randn(n, D, generator=g)
    out = []
    for k in range(5):
        y = apply_task(world.prims, (k,), x); var = y.var(0).mean().item()
        best = -9
        with torch.no_grad():
            for u in model.units:
                best = max(best, 1 - ((u(x) - y) ** 2).mean().item() / var)
        out.append(best)
    return float(np.mean(out))


def run(variant, seed, lr=3e-3, cfg=None):
    cfg = cfg or {}
    torch.manual_seed(seed)
    world = World(seed)
    train = [t for t in world.train_tasks if len(t) == 2]       # 12 known compositions
    held = list(world.held)                                       # 8 never-seen compositions
    rng = np.random.default_rng(seed + 5)
    ids = rng.permutation(NT)                                      # opaque identifiers
    tid = {t: int(ids[i]) for i, t in enumerate(train + held)}
    model = build(variant, lr, cfg)
    if hasattr(model, 'seed'):
        model.seed(seed + 3)
    t0 = time.time()
    for _ in range(PA_STEPS):
        t = train[rng.integers(len(train))]
        x, y = world.batch(t)
        model.train_step(tid[t], x, y, world.var[t])
    res = dict(variant=variant, seed=seed, cfg=cfg)
    old_before = float(np.mean([r2(model, world, t, tid[t]) for t in train]))
    res['phaseA_r2'] = old_before
    res['alignment'] = alignment(model, world)
    aucs, finals, interf, starts = [], [], [], []
    state = copy.deepcopy(model)
    for t in held:
        m = copy.deepcopy(state)
        starts.append(r2(m, world, t, tid[t]))
        curve = []
        for step in range(1, PB_STEPS + 1):
            x, y = world.batch(t)
            m.train_step(tid[t], x, y, world.var[t])
            if step % EVAL_EVERY == 0:
                curve.append(r2(m, world, t, tid[t]))
        aucs.append(float(np.mean(curve))); finals.append(curve[-1])
        interf.append(old_before - float(np.mean([r2(m, world, o, tid[o]) for o in train])))
    res['fewshot_auc'] = float(np.mean(aucs))
    res['fewshot_final'] = float(np.mean(finals))
    res['fewshot_start'] = float(np.mean(starts))
    res['interference'] = float(np.mean(interf))
    res['secs'] = round(time.time() - t0, 1)
    return res


if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]
    out = sys.argv[2]
    variants = sys.argv[3].split(',')
    cfg = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    with open(out, 'a') as f:
        for s in seeds:
            for v in variants:
                r = run(v, s, cfg=cfg if v == 'rl' else {})
                f.write(json.dumps(r) + '\n'); f.flush()
                print(v, s, json.dumps(cfg) if v == 'rl' else '', {k: (round(val, 3) if isinstance(val, float) else val)
                      for k, val in r.items() if k not in ('variant', 'seed', 'cfg')}, flush=True)
