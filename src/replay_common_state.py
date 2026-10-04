"""Robustness replay of experiments 14b/16b: same code and seeds, but every module's initial weights are
perturbed by Gaussian noise of standard deviation 1e-6 (a proxy for cross-platform floating-point differences),
to check whether the aggregate conclusions depend on bitwise reproduction. On this single-core machine, changing
the thread count reproduces the published values bit for bit, so it cannot serve as a perturbation.
Usage: python replay_common_state.py SEEDS OUT.jsonl"""
import json, sys, time
import torch
import exp_common_state as cs


class StigCSPerturbed(cs.StigCS):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        g = torch.Generator().manual_seed(999)
        with torch.no_grad():
            for u in self.units:
                for p in u.parameters():
                    p.add_(1e-6 * torch.randn(p.shape, generator=g))


cs.StigCS = StigCSPerturbed
seeds = [int(s) for s in sys.argv[1].split(',')]
out = sys.argv[2]
with open(out, 'a') as f:
    for s in seeds:
        t0 = time.time()
        for capacity in ['none', 'spare', 'dormant']:
            world, model, info, stream = cs.learn_to_event(s, capacity)
            base = dict(seed=s, capacity=capacity, rare_prim=info['r_prim'], killed_prim=info['a_prim'])
            for rule in ['least_committed', 'random', 'none']:
                f.write(json.dumps(dict(base, exp='14b', rule=rule, intervention='keep', **cs.produce(model, world, info, stream, rule, 'keep'))) + '\n'); f.flush()
            if capacity == 'none':
                for itv in ['reset_opt', 'reset_both']:
                    f.write(json.dumps(dict(base, exp='16b', rule='least_committed', intervention=itv, **cs.produce(model, world, info, stream, 'least_committed', itv))) + '\n'); f.flush()
        print(s, 'done', round(time.time() - t0), 's', flush=True)
