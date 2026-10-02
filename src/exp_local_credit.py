"""
Experiment 10: oracle local credit in the no-symbols world of experiment 7.

Diagnostic of the failure of experiment 7: is the lock the credit assignment (one global quality
for a two-module path) or the traces themselves (indexed by opaque task identifiers)?
Each module's quality is computed against the true output of the primitive its slot should compute,
applied to the module's actual input. This is an oracle: it is an upper bound, not a mechanism.

  aco_thresholds : baseline of experiment 7 (global credit for routing, global loss for content)
  local_routing  : local quality per slot for traces and alarms; content still learns on the global loss
  local_full     : local quality for traces and alarms, and each module also learns on its local target
                   (slot 2 receives slot 1's output detached, so learning stays local)
  soft           : router learned by gradient, for reference on the same seeds (not in the hypotheses)

Usage: python exp_local_credit.py SEEDS OUT.jsonl VARIANTS
"""
import copy, json, sys, time
import numpy as np
import torch
from exp import Stig, apply_task, M, D
import exp_no_symbols as ns
from exp_no_symbols import World, StigT, SoftT, MonoT, pseudo, r2, alignment, PA_STEPS, PB_STEPS, EVAL_EVERY


class StigLocal(StigT):
    def __init__(self, lr, full=False):
        super().__init__(lr)
        self.full = full
        self.oracle = None   # set by run(): (prims, {tid: task})

    def train_step(self, tid, x, y, var):
        prims, tid2task = self.oracle
        task = tid2task[tid]
        syms = pseudo(tid)
        path = self.path(syms)
        qs, losses = [], []
        h = x
        for slot, (k, u) in enumerate(zip(task, path)):
            inp = h.detach() if (self.full and slot > 0) else h
            out = self.units[u](inp)
            with torch.no_grad():
                target = apply_task(prims, (k,), inp.detach())
                tv = target.var(0).mean().item()
                q = float(np.clip(1.0 - ((out.detach() - target) ** 2).mean().item() / max(tv, 1e-8), 0.0, 1.0))
            qs.append(q)
            if self.full:
                losses.append(((out - target) ** 2).mean())
            h = out
        loss = sum(losses) if self.full else ((h - y) ** 2).mean()
        if loss.requires_grad:
            self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()
        self.update_traces_local(syms, path, qs)
        self.check_alarms_local(syms, path, qs)

    def update_traces_local(self, syms, path, qs):
        r = self.rho
        for s in set(syms):
            self.tau[s] *= (1 - r)
        for s, u, q in zip(syms, path, qs):
            self.tau[s, u] += r * q
        np.maximum(self.tau, self.tau_min, out=self.tau)
        self.cool = np.maximum(self.cool - 1, 0)

    def check_alarms_local(self, syms, path, qs):
        for s, u, q in zip(syms, path, qs):
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


def build(variant, lr):
    if variant == 'local_routing':
        return StigLocal(lr, full=False)
    if variant == 'local_full':
        return StigLocal(lr, full=True)
    return ns.build(variant, lr, {})


def run(variant, seed, lr=3e-3):
    torch.manual_seed(seed)
    world = World(seed)
    train = [t for t in world.train_tasks if len(t) == 2]
    held = list(world.held)
    rng = np.random.default_rng(seed + 5)
    ids = rng.permutation(ns.NT)
    tid = {t: int(ids[i]) for i, t in enumerate(train + held)}
    model = build(variant, lr)
    if hasattr(model, 'seed'):
        model.seed(seed + 3)
    if isinstance(model, StigLocal):
        model.oracle = (world.prims, {v: k for k, v in tid.items()})
    t0 = time.time()
    for _ in range(PA_STEPS):
        t = train[rng.integers(len(train))]
        x, y = world.batch(t)
        model.train_step(tid[t], x, y, world.var[t])
    res = dict(variant=variant, seed=seed)
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
    res.update(fewshot_auc=float(np.mean(aucs)), fewshot_final=float(np.mean(finals)), fewshot_start=float(np.mean(starts)),
               interference=float(np.mean(interf)), secs=round(time.time() - t0, 1))
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
                print(v, s, {k: (round(val, 3) if isinstance(val, float) else val) for k, val in r.items() if k not in ('variant', 'seed')}, flush=True)
