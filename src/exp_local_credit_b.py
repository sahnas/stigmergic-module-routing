"""
Experiment 10b: independent re-implementation of experiment 10 (oracle local credit), with a different oracle definition, plus a gradient router given the same local content losses.

Diagnosis to test: without symbols, the mechanism fails because the only signal is the quality of
the whole path (conjunctive, not decomposable), and traces indexed by opaque identifiers cannot
represent that two tasks share a primitive. An oracle supplies, for each slot of the path, a local
quality measured as if the other slot were perfect (slot 1 against the true intermediate P_i(x),
slot 2 applied to the true intermediate and compared with y).

  aco_thresholds   : reference, as in experiment 7 (global credit, global content loss)
  aco_local_route  : level 1, local credit for the traces and alarms only; modules still learn on the global loss
  aco_local_both   : level 2, local credit for traces and alarms, and modules learn on their local targets
  soft_local_both  : router learned by gradient, with the same local content losses (level 2 for the gradient router)

Everything else (world, phases, steps, evaluations, seeds handling) is copied verbatim from exp_no_symbols.run.
The oracle is the experimental manipulation: it is also used while learning the new compositions in phase B.
Usage: python exp_local_credit_b.py SEEDS OUT.jsonl VARIANTS
"""
import copy, json, sys, time
import numpy as np
import torch
from exp import World, apply_task, M
import exp_no_symbols as ns
from exp_no_symbols import NT, PA_STEPS, PB_STEPS, EVAL_EVERY, pseudo, StigT, SoftT, r2, alignment


def clipq(mse, var):
    return float(np.clip(1.0 - mse / var, 0.0, 1.0))


class StigLocal(StigT):
    def __init__(self, lr, content_local):
        super().__init__(lr)
        self.content_local = content_local
        self.ctx = None

    def train_step(self, tid, x, y, var):
        world, task_of = self.ctx
        task = task_of[tid]
        z = apply_task(world.prims, task[:1], x)
        varz = world.var[task[:1]]
        pt = pseudo(tid)
        path = self.path(pt)
        u1, u2 = path
        h1 = self.units[u1](x)
        if self.content_local:
            loss = ((h1 - z) ** 2).mean() + ((self.units[u2](z) - y) ** 2).mean()
        else:
            loss = ((self.units[u2](h1) - y) ** 2).mean()
        if loss.requires_grad:
            self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()
        with torch.no_grad():
            q1 = clipq(((self.units[u1](x) - z) ** 2).mean().item(), varz)
            q2 = clipq(((self.units[u2](z) - y) ** 2).mean().item(), var)
        qs = [q1, q2]
        r = self.rho
        for s in set(pt):
            self.tau[s] *= (1 - r)
        for s, u, q in zip(pt, path, qs):
            self.tau[s, u] += r * q
        np.maximum(self.tau, self.tau_min, out=self.tau)
        self.cool = np.maximum(self.cool - 1, 0)
        for s, u, q in zip(pt, path, qs):
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


class SoftLocal(SoftT):
    def __init__(self, lr):
        super().__init__(lr)
        self.ctx = None

    def train_step(self, tid, x, y, var):
        world, task_of = self.ctx
        task = task_of[tid]
        z = apply_task(world.prims, task[:1], x)
        s1, s2 = pseudo(tid)
        loss = ((self.slot(s1, x) - z) ** 2).mean() + ((self.slot(s2, z) - y) ** 2).mean()
        self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()


def build(variant, lr, cfg):
    if variant == 'aco_local_route':
        return StigLocal(lr, content_local=False)
    if variant == 'aco_local_both':
        return StigLocal(lr, content_local=True)
    if variant == 'soft_local_both':
        return SoftLocal(lr)
    return ns.build(variant, lr, cfg)


def run(variant, seed, lr=3e-3, cfg=None):
    cfg = cfg or {}
    torch.manual_seed(seed)
    world = World(seed)
    train = [t for t in world.train_tasks if len(t) == 2]
    held = list(world.held)
    rng = np.random.default_rng(seed + 5)
    ids = rng.permutation(NT)
    tid = {t: int(ids[i]) for i, t in enumerate(train + held)}
    model = build(variant, lr, cfg)
    if hasattr(model, 'ctx'):
        model.ctx = (world, {tid[t]: t for t in tid})
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
    with open(out, 'a') as f:
        for s in seeds:
            for v in variants:
                r = run(v, s)
                f.write(json.dumps(r) + '\n'); f.flush()
                print(v, s, {k: (round(val, 3) if isinstance(val, float) else val) for k, val in r.items() if k not in ('variant', 'seed', 'cfg')}, flush=True)
