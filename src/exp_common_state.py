"""
Experiments 14b and 16b on a common-state harness.

Harness: for each seed and capacity condition, one model is trained up to the event (uniform learning, then the
death of the module carrying a frequent primitive a), then deep-copied before every condition, so that weights,
optimizer state, traces, alarm statistics, counters and the routing generator are identical at the event. The
production data (task order and batches) are drawn once per seed and reused by every condition. Recruitments are
timestamped (production step), so "first recruit" means the first one after the event.

Capacity: 'none' (5 active modules), 'spare' (7 active from the start; over-provisioning, formerly mislabelled
"dormant"), 'dormant' (5 active + 2 inactive, held back during learning; after the event, recruitment activates an inactive
module first).
14b rules after the event: 'least_committed' (current), 'random', 'none' (alarms without recruitment).
16b (capacity 'none', rule least_committed): intervention at the first post-event recruitment:
'keep' (weights and optimizer state kept), 'reset_opt' (optimizer state cleared), 'reset_both' (weights
reinitialised and optimizer state cleared). Later recruitments follow the mechanism unchanged.
Usage: python exp_common_state.py SEEDS OUT.jsonl
"""
import copy, json, sys, time
import numpy as np
import torch
import exp_dynamic as ed
from exp_dynamic import StigDyn, M
from exp import World, mean_r2, K, Unit, apply_task, BATCH, D

RARE_W, STEPS1, STEPS2, BLOCKS = 0.02, 100, 40, 17
CAPACITY = {'none': (5, 0), 'spare': (7, 0), 'dormant': (5, 2)}


class StigCS(StigDyn):
    def __init__(self, lr, n_active, n_dormant):
        super().__init__(lr, 'aco', True)
        self.active[:] = False
        self.active[:n_active] = True
        self.n_dormant_total = n_dormant
        self.dormant = np.zeros(M, dtype=bool); self.dormant[n_active:n_active + n_dormant] = True
        self.recruit_rule = 'least_committed'
        self.intervention, self.intervened = 'keep', False
        self.log, self.step_counter, self.event_step = [], 0, None

    def recruit(self, s, u, path):
        self.alarms += 1
        if self.recruit_rule == 'none':
            self.cool[s] = self.cooldown_len; return
        if self.dormant.any() and self.event_step is not None:   # reserves are held back until the event
            r = int(np.flatnonzero(self.dormant)[0]); self.dormant[r] = False; self.active[r] = True; kind = 'dormant'
        else:
            load = self.tau.sum(0).copy(); load[path] = np.inf; load[~self.active] = np.inf
            if self.recruit_rule == 'least_committed':
                r = int(np.argmin(load))
            else:
                r = int(self.rng.choice(np.where(np.isfinite(load))[0]))
            if not np.isfinite(load[r]):
                return
            kind = 'active'
        post = self.event_step is not None and self.step_counter > self.event_step
        self.log.append(dict(step=self.step_counter, module=r, symbol=int(s), kind=kind, post_event=post))
        if post and not self.intervened and self.intervention != 'keep':
            self.intervened = True
            if self.intervention == 'reset_both':
                self.units[r].load_state_dict(Unit().state_dict())
            for p in self.units[r].parameters():
                self.opt.state.pop(p, None)
        self.tau[s, r] = max(self.tau[s, r], self.tau[s, u])
        self.qbar[r], self.qdev[r], self.uses[r] = 0.0, 0.0, 0
        self.cool[s] = self.cooldown_len

    def train_step(self, task, x, y, var):
        self.step_counter += 1
        super().train_step(task, x, y, var)


def draw_production(world, tt, w, seed):
    g = torch.Generator().manual_seed(seed + 404)
    rng = np.random.default_rng(seed + 505)
    stream = []
    for _ in range(2):
        for i in rng.choice(len(tt), size=BLOCKS, p=w):
            t = tt[int(i)]
            xs = torch.randn(STEPS2, BATCH, D, generator=g)
            stream.append((t, [(xs[j], apply_task(world.prims, t, xs[j])) for j in range(STEPS2)]))
    return stream


def learn_to_event(seed, capacity, lr=3e-3):
    torch.manual_seed(seed)
    world = World(seed)
    n_active, n_dormant = CAPACITY[capacity]
    model = StigCS(lr, n_active, n_dormant); model.seed(seed + 3)
    rng = np.random.default_rng(seed + 5)
    ev_rng = np.random.default_rng(seed + 17)
    tt = world.train_tasks
    r_prim, a_prim = [int(p) for p in ev_rng.permutation(K)[:2]]
    for _ in range(2):
        for i in rng.permutation(len(tt)):
            t = tt[int(i)]
            for _ in range(STEPS1):
                x, y = world.batch(t); model.train_step(t, x, y, world.var[t])
    spec = {k: model.greedy(k) for k in range(K)}
    model.units[spec[a_prim]].dead = True
    model.event_step = model.step_counter
    w = np.array([RARE_W if r_prim in t else 1.0 for t in tt]); w /= w.sum()
    return world, model, dict(r_prim=r_prim, a_prim=a_prim, spec=spec), draw_production(world, tt, w, seed)


def produce(model, world, info, stream, rule, intervention):
    model = copy.deepcopy(model)
    model.recruit_rule, model.intervention = rule, intervention
    tt = world.train_tasks
    r, a = info['r_prim'], info['a_prim']
    rare = [t for t in tt if r in t and a not in t]; other = [t for t in tt if r not in t and a not in t]
    killed = [t for t in tt if a in t]
    res = dict(rare_before=mean_r2(model, world, rare, 300), other_before=mean_r2(model, world, other, 300))
    curve, kcurve = [], []
    for t, batches in stream:
        for x, y in batches:
            model.train_step(t, x, y, world.var[t])
        curve.append(mean_r2(model, world, tt, 300)); kcurve.append(mean_r2(model, world, killed, 300))
    post = [e for e in model.log if e['post_event']]
    first = post[0] if post else None
    res.update(availability=float(np.mean(curve)), killed_auc=float(np.mean(kcurve)), killed_after=kcurve[-1],
               rare_after=mean_r2(model, world, rare, 300), other_after=mean_r2(model, world, other, 300),
               alarms=model.alarms, post_recruits=len(post), distinct_post=len({e['module'] for e in post}),
               first_post=first, first_is_rare_specialist=(first['module'] == info['spec'][r]) if first else None,
               first_is_dormant=(first['kind'] == 'dormant') if first else None)
    return res


if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]
    out = sys.argv[2]
    with open(out, 'a') as f:
        for s in seeds:
            t0 = time.time()
            for capacity in ['none', 'spare', 'dormant']:
                world, model, info, stream = learn_to_event(s, capacity)
                base = dict(seed=s, capacity=capacity, rare_prim=info['r_prim'], killed_prim=info['a_prim'])
                for rule in ['least_committed', 'random', 'none']:
                    r = dict(base, exp='14b', rule=rule, intervention='keep', **produce(model, world, info, stream, rule, 'keep'))
                    f.write(json.dumps(r) + '\n'); f.flush()
                if capacity == 'none':
                    for itv in ['reset_opt', 'reset_both']:
                        r = dict(base, exp='16b', rule='least_committed', intervention=itv, **produce(model, world, info, stream, 'least_committed', itv))
                        f.write(json.dumps(r) + '\n'); f.flush()
            print(s, 'done', round(time.time() - t0), 's', flush=True)
