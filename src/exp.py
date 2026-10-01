"""
Minimal test of the hypothesis: stigmergic coordination between small units,
with forgetting of trails and alarm thresholds specific to each unit.

What is tested (and only that):
  the coordination and compensation mechanism. Units are small supervised
  regressors that communicate through the observation space (a fixed interface,
  like Unix text streams), not JEPAs. No shared latent space.

World: K primitives P_k (rotation + nonlinearity) on R^D. A task is a single
primitive (k) or a composition (i, j): y = P_j(P_i(x)). The system receives
the symbols of the task, not a learned decomposition.

Protocol:
  Phase 1 (continual learning): training tasks presented in successive blocks
    (2 passes, random order). Measures: retention on the training tasks,
    zero-shot generalisation on 8 never-seen compositions.
  Phase 2 (damage): the unit that carries primitive k* dies permanently
    (zero output, cannot relearn), followed by a short recovery phase.
    Measures: performance on the tasks involving k*, zero-shot on the held-out
    compositions involving k*, retention of the other tasks.

Hypotheses fixed before the test seeds:
  H1 compensation: forgetting of trails (aco) recovers better than memory without
     forgetting (cumulative) and than fixed orchestration (oracle).
  H2 thresholds: aco + per-unit thresholds + reserve recruitment recovers better than aco alone.
  H3 global temporal forgetting (literal ants) degrades retention compared with local aco.
  H4 generalisation: stigmergic routing >= router learned by gradient (soft) and > monolithic.
  Control: ema (forgetting by recency, on the edge used only), to tell whether forgetting
     "by absence of reinforcement" adds anything beyond recency.
"""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

D, K, M, H = 8, 5, 8, 32
BATCH = 32


# ----------------------------------------------------------------------------- world
def make_world(seed):
    g = np.random.default_rng(seed)
    prims = []
    for _ in range(K):
        Q, _ = np.linalg.qr(g.normal(size=(D, D)))
        B = g.normal(size=(D, D)) * 1.5 / math.sqrt(D)
        prims.append((torch.tensor(Q, dtype=torch.float32), torch.tensor(B, dtype=torch.float32)))
    pairs = [(i, j) for i in range(K) for j in range(K) if i != j]
    while True:
        perm = g.permutation(len(pairs))
        held = [pairs[p] for p in perm[:8]]
        train = [pairs[p] for p in perm[8:]]
        if len({i for i, _ in train}) == K and len({j for _, j in train}) == K:
            break
    counts = [sum(k in p for p in held) for k in range(K)]
    kstar = int(np.argmax(counts))
    singles = [(k,) for k in range(K)]
    return prims, singles + train, held, kstar


def apply_task(prims, task, x):
    for k in task:
        Q, B = prims[k]
        x = x @ Q.T + 0.5 * torch.tanh(x @ B.T)
    return x


class World:
    def __init__(self, seed):
        self.prims, self.train_tasks, self.held, self.kstar = make_world(seed)
        g = torch.Generator().manual_seed(seed + 7)
        self.var, self.test_x = {}, {}
        for t in self.train_tasks + self.held:
            xb = torch.randn(20000, D, generator=g)
            self.var[t] = apply_task(self.prims, t, xb).var(0).mean().item()
            self.test_x[t] = torch.randn(1000, D, generator=g)
        self.g = torch.Generator().manual_seed(seed + 11)

    def batch(self, task):
        x = torch.randn(BATCH, D, generator=self.g)
        return x, apply_task(self.prims, task, x)


# ----------------------------------------------------------------------------- models
class Unit(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, D))
        self.dead = False

    def forward(self, x):
        if self.dead:
            return torch.zeros_like(x)
        return self.lin(x) + self.mlp(x)


class Mono:
    """Monolithic MLP that receives x and the symbols of the task."""
    def __init__(self, lr):
        self.l1, self.l2, self.l3 = nn.Linear(D + 2 * K + 1, 128), nn.Linear(128, 128), nn.Linear(128, D)
        self.mask = torch.ones(128)
        self.params = list(self.l1.parameters()) + list(self.l2.parameters()) + list(self.l3.parameters())
        self.opt = torch.optim.Adam(self.params, lr)

    def code(self, task, n):
        c = torch.zeros(n, 2 * K + 1)
        c[:, task[0]] = 1
        c[:, K + (task[1] if len(task) == 2 else K)] = 1
        return c

    def fwd(self, task, x):
        z = torch.cat([x, self.code(task, x.shape[0])], 1)
        h = torch.relu(self.l1(z)) * self.mask
        h = torch.relu(self.l2(h)) * self.mask
        return self.l3(h)

    def train_step(self, task, x, y, var):
        loss = ((self.fwd(task, x) - y) ** 2).mean()
        self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()

    def predict(self, task, x):
        with torch.no_grad():
            return self.fwd(task, x)

    def damage(self, kstar, g):
        idx = torch.randperm(128, generator=g)[:16]  # permanent loss of 1/8 of the neurons
        self.mask[idx] = 0.0

    def route(self, k):
        return None


class Oracle:
    """Fixed orchestration: symbol k -> unit k, forever."""
    def __init__(self, lr):
        self.units = [Unit() for _ in range(M)]
        self.opt = torch.optim.Adam([p for u in self.units for p in u.parameters()], lr)

    def fwd(self, task, x):
        for k in task:
            x = self.units[k](x)
        return x

    def train_step(self, task, x, y, var):
        loss = ((self.fwd(task, x) - y) ** 2).mean()
        if loss.requires_grad:
            self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()

    def predict(self, task, x):
        with torch.no_grad():
            return self.fwd(task, x)

    def damage(self, kstar, g):
        self.units[kstar].dead = True

    def route(self, k):
        return k


class Soft:
    """Router learned by gradient (softmax mixture, mixture-of-experts style)."""
    def __init__(self, lr):
        self.units = [Unit() for _ in range(M)]
        self.R = nn.Parameter(torch.randn(K, M) * 0.01)
        self.opt = torch.optim.Adam([p for u in self.units for p in u.parameters()] + [self.R], lr)

    def slot(self, s, h):
        w = torch.softmax(self.R[s], 0)
        outs = torch.stack([u(h) for u in self.units], 0)
        return torch.einsum('m,mbd->bd', w, outs)

    def fwd(self, task, x):
        for s in task:
            x = self.slot(s, x)
        return x

    def train_step(self, task, x, y, var):
        loss = ((self.fwd(task, x) - y) ** 2).mean()
        self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()

    def predict(self, task, x):
        with torch.no_grad():
            return self.fwd(task, x)

    def damage(self, kstar, g):
        self.units[int(torch.argmax(self.R[kstar]))].dead = True

    def route(self, k):
        return int(torch.argmax(self.R[k]))


class Stig:
    """
    Routing by traces on the edges symbol -> unit. Choice sampled ~ softmax(tau / T).
    Trace update rules (q = batch quality, R2 clipped to [0, 1]):
      cumulative : cumulative mean of q on the edge used, nothing fades (perfect memory)
      ema        : moving average of q on the edge used only (forgetting by recency)
      aco        : evaporation of all edges of the active symbol + deposit on the edge used
                   (forgetting by absence of reinforcement, pheromone style)
      global     : evaporation of all edges of all symbols at every step (literal ants)
    thresholds (option): each unit tracks its own usual quality; an abnormal drop
      (below its mean minus kappa deviations) triggers the global mode: recruitment of
      the least engaged unit (reserve) for that symbol.
    """
    def __init__(self, lr, rule, thresholds=False, rho=0.02, T=0.05, tau0=0.0, tau_min=0.0,
                 kappa=3.0, cooldown=100, min_uses=50):
        self.units = [Unit() for _ in range(M)]
        self.opt = torch.optim.Adam([p for u in self.units for p in u.parameters()], lr)
        self.rule, self.thresholds, self.rho, self.T = rule, thresholds, rho, T
        self.tau = np.full((K, M), tau0)
        self.cnt = np.ones((K, M))
        self.tau_min = tau_min
        self.rng = np.random.default_rng(0)
        self.kappa, self.cooldown_len, self.min_uses = kappa, cooldown, min_uses
        self.qbar, self.qdev, self.uses = np.zeros(M), np.zeros(M), np.zeros(M, dtype=int)
        self.cool = np.zeros(K, dtype=int)
        self.alarms = 0

    def seed(self, s):
        self.rng = np.random.default_rng(s)

    def sample(self, s, exclude=None):
        logits = self.tau[s] / self.T
        if exclude is not None:
            logits = logits.copy(); logits[exclude] = -np.inf
        p = np.exp(logits - logits.max()); p /= p.sum()
        return int(self.rng.choice(M, p=p))

    def greedy(self, s, exclude=None):
        v = self.tau[s].copy()
        if exclude is not None:
            v[exclude] = -np.inf
        return int(np.argmax(v))

    def path(self, task, greedy=False):
        f = self.greedy if greedy else self.sample
        u1 = f(task[0])
        if len(task) == 1:
            return [u1]
        return [u1, f(task[1], exclude=u1)]

    def fwd(self, path, x):
        for u in path:
            x = self.units[u](x)
        return x

    def train_step(self, task, x, y, var):
        path = self.path(task)
        pred = self.fwd(path, x)
        loss = ((pred - y) ** 2).mean()
        if loss.requires_grad:
            self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()
        q = float(np.clip(1.0 - loss.item() / var, 0.0, 1.0))
        self.update_traces(task, path, q)
        if self.thresholds:
            self.check_alarms(task, path, q)

    def update_traces(self, task, path, q):
        r = self.rho
        if self.rule == 'global':
            self.tau *= (1 - r)
        elif self.rule == 'aco':
            for s in set(task):
                self.tau[s] *= (1 - r)
        for s, u in zip(task, path):
            if self.rule in ('aco', 'global'):
                self.tau[s, u] += r * q
            elif self.rule == 'ema':
                self.tau[s, u] = (1 - r) * self.tau[s, u] + r * q
            elif self.rule == 'cumulative':
                self.tau[s, u] = (self.tau[s, u] * self.cnt[s, u] + q) / (self.cnt[s, u] + 1)
                self.cnt[s, u] += 1
        np.maximum(self.tau, self.tau_min, out=self.tau)
        self.cool = np.maximum(self.cool - 1, 0)

    def check_alarms(self, task, path, q):
        for s, u in zip(task, path):
            if self.uses[u] >= self.min_uses and self.cool[s] == 0:
                thr = self.qbar[u] - self.kappa * max(self.qdev[u], 0.02)
                if q < thr:
                    self.recruit(s, u, path)
            a = 0.05
            if self.uses[u] == 0:
                self.qbar[u] = q
            self.qdev[u] = (1 - a) * self.qdev[u] + a * abs(q - self.qbar[u])
            self.qbar[u] = (1 - a) * self.qbar[u] + a * q
            self.uses[u] += 1

    def recruit(self, s, u, path):
        load = self.tau.sum(0).copy()
        load[path] = np.inf
        r = int(np.argmin(load))
        self.tau[s, r] = max(self.tau[s, r], self.tau[s, u])
        self.qbar[r], self.qdev[r], self.uses[r] = 0.0, 0.0, 0
        self.cool[s] = self.cooldown_len
        self.alarms += 1

    def predict(self, task, x):
        with torch.no_grad():
            return self.fwd(self.path(task, greedy=True), x)

    def damage(self, kstar, g):
        self.units[self.greedy(kstar)].dead = True

    def route(self, k):
        return self.greedy(k)


def build(variant, lr, hp):
    if variant == 'mono':
        return Mono(lr)
    if variant == 'oracle':
        return Oracle(lr)
    if variant == 'soft':
        return Soft(lr)
    rule, thresholds = {'cumulative': ('cumulative', False), 'ema': ('ema', False), 'aco': ('aco', False),
                    'global': ('global', False), 'aco_thresholds': ('aco', True)}[variant]
    return Stig(lr, rule, thresholds, **hp)


# ----------------------------------------------------------------------------- protocol
def r2(model, world, task, n=1000):
    x = world.test_x[task][:n]
    y = apply_task(world.prims, task, x)
    mse = ((model.predict(task, x) - y) ** 2).mean().item()
    return 1.0 - mse / world.var[task]


def mean_r2(model, world, tasks, n=1000):
    return float(np.mean([r2(model, world, t, n) for t in tasks])) if tasks else float('nan')


def run(variant, seed, lr=3e-3, hp=None, steps1=150, passes1=2, steps2=60, passes2=2):
    hp = hp or {}
    torch.manual_seed(seed)
    world = World(seed)
    model = build(variant, lr, hp)
    if isinstance(model, Stig):
        model.seed(seed + 3)
    order_rng = np.random.default_rng(seed + 5)
    tt, held, ks = world.train_tasks, world.held, world.kstar
    t0 = time.time()

    # Phase 1: continual learning in blocks
    for _ in range(passes1):
        for i in order_rng.permutation(len(tt)):
            task = tt[i]
            for _ in range(steps1):
                x, y = world.batch(task)
                model.train_step(task, x, y, world.var[task])

    res = dict(variant=variant, seed=seed, kstar=ks)
    res['train_r2'] = mean_r2(model, world, tt)
    res['zero_r2'] = mean_r2(model, world, held)
    k_train = [t for t in tt if ks in t]
    k_held = [t for t in held if ks in t]
    other = [t for t in tt if ks not in t]
    res['k_train_pre'] = mean_r2(model, world, k_train)
    res['k_held_pre'] = mean_r2(model, world, k_held)
    res['other_pre'] = mean_r2(model, world, other)
    if isinstance(model, Stig):
        routes = [model.route(k) for k in range(K)]
        res['n_distinct_units'] = len(set(routes))
        res['alarms_phase1'] = model.alarms

    # Phase 2: permanent death of the unit carrying k*, then recovery
    dead_route = model.route(ks)
    model.damage(ks, torch.Generator().manual_seed(seed + 13))
    res['k_train_dmg'] = mean_r2(model, world, k_train)
    curve, latency, step = [], None, 0
    for _ in range(passes2):
        for i in order_rng.permutation(len(tt)):
            task = tt[i]
            for _ in range(steps2):
                x, y = world.batch(task)
                model.train_step(task, x, y, world.var[task])
                step += 1
                if latency is None and dead_route is not None and model.route(ks) != dead_route:
                    latency = step
            curve.append((step, mean_r2(model, world, k_train, n=300)))
    res['k_train_rec'] = mean_r2(model, world, k_train)
    res['k_held_rec'] = mean_r2(model, world, k_held)
    res['other_rec'] = mean_r2(model, world, other)
    res['train_r2_rec'] = mean_r2(model, world, tt)
    res['zero_r2_rec'] = mean_r2(model, world, held)
    res['reroute_latency'] = latency
    res['curve'] = curve
    if isinstance(model, Stig):
        res['alarms_total'] = model.alarms
    res['secs'] = round(time.time() - t0, 1)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', default='mono,oracle,soft,cumulative,ema,aco,global,aco_thresholds')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--hp', default='{}')
    ap.add_argument('--out', default='results.jsonl')
    a = ap.parse_args()
    hp = json.loads(a.hp)
    seeds = [int(s) for s in a.seeds.split(',')]
    with open(a.out, 'a') as f:
        for seed in seeds:
            for v in a.variants.split(','):
                r = run(v, seed, a.lr, hp)
                r['hp'] = hp; r['lr'] = a.lr
                f.write(json.dumps(r) + '\n'); f.flush()
                print(v, seed, {k: (round(val, 3) if isinstance(val, float) else val)
                                for k, val in r.items() if k not in ('curve', 'hp')}, flush=True)
