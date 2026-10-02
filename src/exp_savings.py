"""
Experiment 15: savings. After a specialist has been consumed (experiment 14, learned-then-rare scenario, no
reserve, least-committed recruitment), does it relearn its function faster than a fresh module?

For each seed: replay the production phase of experiment 14 exactly (same code path, same RNG consumption),
then train, in isolation on the single-primitive task (r,), three candidate modules with their own optimizer:
  consumed_specialist : the module that carried r before the event, overwritten since
  other_specialist    : a live module that carried another primitive b (b != r, b != a) before the event
  fresh               : a newly initialised module
Measures: R2 on (r,) before relearning, area under the R2 curve over RELEARN steps (evaluated every 10 steps),
and the first step at which R2 reaches 0.8. A fresh module learns the single task in under 100 steps here, so the
window is short (150 steps, evaluated every 5) to keep the measure sensitive to early savings.
Usage: python exp_savings.py SEEDS OUT.jsonl
"""
import copy, json, sys, time
import numpy as np
import torch
import exp_preservation as ep
from exp_preservation import StigPres, RARE_W, STEPS1, STEPS2, BLOCKS
from exp import World, Unit, K, apply_task

RELEARN, EVAL_EVERY = 150, 5


def r2_unit(unit, world, task, n=500):
    with torch.no_grad():
        x, y = world.batch(task, n) if 'n' in world.batch.__code__.co_varnames else world.batch(task)
        p = unit(x)
        return float(1 - ((p - y) ** 2).mean() / y.var(0).mean())


def relearn(unit, world, task, lr=3e-3):
    unit = copy.deepcopy(unit); unit.dead = False
    opt = torch.optim.Adam(unit.parameters(), lr)
    curve, hit = [], None
    for step in range(1, RELEARN + 1):
        x, y = world.batch(task)
        loss = ((unit(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % EVAL_EVERY == 0:
            r = r2_unit(unit, world, task); curve.append(r)
            if hit is None and r >= 0.8:
                hit = step
    return curve, hit


def run(seed, lr=3e-3):
    torch.manual_seed(seed)
    world = World(seed)
    model = StigPres(lr, K, 'least_committed')
    model.seed(seed + 3)
    rng = np.random.default_rng(seed + 5)
    ev_rng = np.random.default_rng(seed + 17)
    tt = world.train_tasks
    r_prim, a_prim = [int(p) for p in ev_rng.permutation(K)[:2]]
    b_prim = [k for k in range(K) if k not in (r_prim, a_prim)][0]
    w = np.array([RARE_W if r_prim in t else 1.0 for t in tt]); w /= w.sum()
    t0 = time.time()

    def train_block(task, n):
        for _ in range(n):
            x, y = world.batch(task)
            model.train_step(task, x, y, world.var[task])

    for _ in range(2):
        for i in rng.permutation(len(tt)):
            train_block(tt[int(i)], STEPS1)
    spec = {k: model.greedy(k) for k in range(K)}
    spec_r, spec_b = spec[r_prim], spec[b_prim]
    single_r = (r_prim,)
    res = dict(seed=seed, rare_prim=r_prim, killed_prim=a_prim, other_prim=b_prim, spec_r=spec_r, spec_b=spec_b,
               r2_specialist_before_event=r2_unit(model.units[spec_r], world, single_r))
    model.units[spec[a_prim]].dead = True
    for _ in range(2):
        order = rng.choice(len(tt), size=BLOCKS, p=w)
        for i in order:
            train_block(tt[int(i)], STEPS2)
    res['n_recruits'] = len(model.recruits)
    res['spec_r_recruited'] = spec_r in model.recruits
    torch.manual_seed(seed + 101)
    cands = {'consumed_specialist': model.units[spec_r], 'other_specialist': model.units[spec_b], 'fresh': Unit()}
    for name, unit in cands.items():
        res[f'{name}_r2_start'] = r2_unit(unit, world, single_r)
        torch.manual_seed(seed + 202)   # same batches for every candidate
        curve, hit = relearn(unit, world, single_r, lr)
        res[f'{name}_auc'] = float(np.mean(curve)); res[f'{name}_steps_to_0.8'] = hit if hit is not None else RELEARN + EVAL_EVERY
        res[f'{name}_final'] = curve[-1]
    res['secs'] = round(time.time() - t0, 1)
    return res


if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]
    out = sys.argv[2]
    with open(out, 'a') as f:
        for s in seeds:
            r = run(s)
            f.write(json.dumps(r) + '\n'); f.flush()
            print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if 'auc' in k or 'start' in k or 'steps' in k or k in ('spec_r_recruited', 'n_recruits', 'r2_specialist_before_event')}, flush=True)
