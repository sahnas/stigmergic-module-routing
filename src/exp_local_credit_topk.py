"""
Experiment 11: sparse top-k routers given the same local content losses as experiment 10b (level 2),
in the no-symbols world of experiment 7.

Question: is the clean decomposition obtained by trace routing in 10b (alignment 0.99, low interference)
a property of traces, or of hard assignment in general? A top-1 router is a hard assignment learned by
gradient; a top-2 router is a near-hard mixture.

  topk1_local_both : noisy top-1 gating (configuration of experiment 8), local content losses per slot
  topk2_local_both : noisy top-2 gating (configuration of experiment 8), local content losses per slot
  aco_local_both   : reference from experiment 10b (trace routing, local content losses)

Usage: python exp_local_credit_topk.py SEEDS OUT.jsonl VARIANTS
"""
import json, sys
import torch
import torch.nn as nn
from exp import apply_task, M
from exp_topk import TopK
from exp_no_symbols import NT, pseudo
import exp_local_credit_b as b


class TopKLocal(TopK):
    def __init__(self, lr, k, noise, alpha):
        super().__init__(lr, k=k, noise=noise, alpha=alpha)
        self.R = nn.Parameter(torch.randn(2 * NT, M) * 0.01)
        self.opt = torch.optim.Adam([p for u in self.units for p in u.parameters()] + [self.R], lr)
        self.ctx = None

    def train_step(self, tid, x, y, var):
        world, task_of = self.ctx
        task = task_of[tid]
        z = apply_task(world.prims, task[:1], x)
        s1, s2 = pseudo(tid)
        loss = ((self.slot(s1, x, train=True) - z) ** 2).mean() + ((self.slot(s2, z, train=True) - y) ** 2).mean()
        if self.alpha > 0:
            P = torch.softmax(self.R, 1)
            loss = loss + self.alpha * M * (P.mean(0) ** 2).sum()
        self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()

    def predict(self, tid, x):
        with torch.no_grad():
            return self.fwd(pseudo(tid), x, train=False)


_build = b.build


def build(variant, lr, cfg):
    if variant == 'topk1_local_both':
        return TopKLocal(lr, k=1, noise=0.3, alpha=0.01)
    if variant == 'topk2_local_both':
        return TopKLocal(lr, k=2, noise=0.3, alpha=0.0)
    return _build(variant, lr, cfg)


b.build = build

if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]
    out = sys.argv[2]
    variants = sys.argv[3].split(',')
    with open(out, 'a') as f:
        for s in seeds:
            for v in variants:
                r = b.run(v, s)
                f.write(json.dumps(r) + '\n'); f.flush()
                print(v, s, {k: (round(val, 3) if isinstance(val, float) else val) for k, val in r.items() if k not in ('variant', 'seed', 'cfg')}, flush=True)
