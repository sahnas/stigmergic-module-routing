"""
Experiment 14: does least-committed recruitment sacrifice a rare specialist?

Production world of experiment 3 (10 slots, symbols given, no routing gradient), one event only:
the death of the module carrying a frequent primitive a. One primitive r is rare: the tasks that
contain it are sampled with relative weight 0.02.
  scenario 'rare_from_start' : r is rare during initial learning and production
  scenario 'learned_then_rare': uniform initial learning, r becomes rare in production
  reserves 0 or 2 dormant modules (5 or 7 active modules at the start)
  recruitment rule: 'least_committed' (current mechanism, lowest total trace), 'random' (random active
  module off the path), 'none' (alarm fires but nobody is recruited)
Measures: which module is recruited at the first alarm (the specialist of r? a dormant module?), R2 on
the tasks containing r before the event and at the end, R2 on the other tasks, availability.
Usage: python exp_preservation.py SEEDS OUT.jsonl
"""
import json, sys, time
import numpy as np
import torch
import exp_dynamic as ed
from exp_dynamic import StigDyn, M
from exp import World, mean_r2, K

RARE_W, STEPS1, STEPS2, BLOCKS = 0.02, 100, 40, 17


class StigPres(StigDyn):
    def __init__(self, lr, n_active, rule):
        super().__init__(lr, 'aco', True)
        self.active[:] = False
        self.active[:n_active] = True
        self.recruit_rule = rule   # not 'rule': Stig uses self.rule for the forgetting rule
        self.recruits = []

    def recruit(self, s, u, path):
        self.alarms += 1
        if self.recruit_rule == 'none':
            self.cool[s] = self.cooldown_len
            return
        load = self.tau.sum(0).copy()
        load[path] = np.inf
        load[~self.active] = np.inf
        if self.recruit_rule == 'least_committed':
            r = int(np.argmin(load))
        else:
            cands = np.where(np.isfinite(load))[0]
            r = int(self.rng.choice(cands))
        if not np.isfinite(load[r]):
            return
        self.recruits.append(r)
        self.tau[s, r] = max(self.tau[s, r], self.tau[s, u])
        self.qbar[r], self.qdev[r], self.uses[r] = 0.0, 0.0, 0
        self.cool[s] = self.cooldown_len


def run(scenario, reserves, rule, seed, lr=3e-3):
    torch.manual_seed(seed)
    world = World(seed)
    model = StigPres(lr, K + reserves, rule)
    model.seed(seed + 3)
    rng = np.random.default_rng(seed + 5)
    ev_rng = np.random.default_rng(seed + 17)
    tt = world.train_tasks
    r_prim, a_prim = [int(p) for p in ev_rng.permutation(K)[:2]]
    rare_tasks = [t for t in tt if r_prim in t and a_prim not in t]      # tasks that need r but not the killed primitive
    other_tasks = [t for t in tt if r_prim not in t and a_prim not in t]  # tasks that need neither
    w = np.array([RARE_W if r_prim in t else 1.0 for t in tt]); w /= w.sum()
    t0 = time.time()

    def train_block(task, n):
        for _ in range(n):
            x, y = world.batch(task)
            model.train_step(task, x, y, world.var[task])

    def one_pass(n, weighted):
        if weighted:
            order = rng.choice(len(tt), size=BLOCKS, p=w)
        else:
            order = rng.permutation(len(tt))
        for i in order:
            train_block(tt[int(i)], n)

    for _ in range(2):
        one_pass(STEPS1, weighted=(scenario == 'rare_from_start'))
    spec = {k: model.greedy(k) for k in range(K)}
    spec_r = spec[r_prim]
    res = dict(scenario=scenario, reserves=reserves, rule=rule, seed=seed, rare_prim=r_prim, killed_prim=a_prim,
               spec_r=spec_r, spec_r_total_trace=float(model.tau.sum(0)[spec_r]),
               rare_r2_before=mean_r2(model, world, rare_tasks), other_r2_before=mean_r2(model, world, other_tasks))
    model.units[spec[a_prim]].dead = True
    curve = []
    for _ in range(2):
        order = rng.choice(len(tt), size=BLOCKS, p=w)
        for i in order:
            train_block(tt[int(i)], STEPS2)
            curve.append(mean_r2(model, world, tt, n=300))
    first = model.recruits[0] if model.recruits else None
    res.update(availability=float(np.mean(curve)), alarms=model.alarms, n_recruits=len(model.recruits),
               recruits=model.recruits, distinct_recruits=len(set(model.recruits)),
               first_recruit=first, first_is_rare_specialist=(first == spec_r) if first is not None else None,
               first_is_dormant=(first is not None and first >= K),
               spec_r_recruited=(spec_r in model.recruits), spec_r_total_trace_after=float(model.tau.sum(0)[spec_r]),
               rare_r2_after=mean_r2(model, world, rare_tasks), other_r2_after=mean_r2(model, world, other_tasks),
               secs=round(time.time() - t0, 1))
    return res


if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]
    out = sys.argv[2]
    with open(out, 'a') as f:
        for s in seeds:
            for scenario in ['rare_from_start', 'learned_then_rare']:
                for reserves in [0, 2]:
                    for rule in ['least_committed', 'random', 'none']:
                        r = run(scenario, reserves, rule, s)
                        f.write(json.dumps(r) + '\n'); f.flush()
                        print(scenario, reserves, rule, s, 'recruit', r['first_recruit'], 'rare_spec', r['spec_r'],
                              'spec_r_recruited', r['spec_r_recruited'], 'rare', round(r['rare_r2_before'], 2), '->', round(r['rare_r2_after'], 2),
                              'other', round(r['other_r2_before'], 2), '->', round(r['other_r2_after'], 2),
                              'avail', round(r['availability'], 3), r['secs'], flush=True)
