"""
Experiment 8: sparse top-k mixture-of-experts router, the standard gradient-trained
opponent missing from experiments 1 and 2. Same world and protocol as exp.py
(continual learning with symbols, then permanent death of the module carrying k*).

Router: one logit vector per task symbol (routing is per symbol, as for every other
variant). During training, Gaussian noise is added to the logits (noisy gating).
The k modules with the largest logits are used, weighted by their full-softmax
probabilities (Switch-style, so the router also gets a gradient when k = 1).
Optional auxiliary balancing loss across symbols: alpha * M * sum_m (mean_s p[s, m])^2.

Usage: python exp_topk.py SEEDS OUT.jsonl VARIANTS '{"k":2,"noise":1.0,"alpha":0.01}'
"""
import json, sys
import torch
import exp
from exp import Soft, M, K


class TopK(Soft):
    def __init__(self, lr, k=2, noise=1.0, alpha=0.0):
        super().__init__(lr)
        self.k, self.noise, self.alpha = k, noise, alpha

    def slot(self, s, h, train=False):
        logits = self.R[s]
        if train and self.noise > 0:
            logits = logits + torch.randn(M) * self.noise
        p = torch.softmax(logits, 0)
        top = torch.topk(logits, self.k).indices
        out = 0
        for i in top.tolist():
            out = out + p[i] * self.units[i](h)
        return out

    def fwd(self, task, x, train=False):
        for s in task:
            x = self.slot(s, x, train)
        return x

    def train_step(self, task, x, y, var):
        loss = ((self.fwd(task, x, train=True) - y) ** 2).mean()
        if self.alpha > 0:
            P = torch.softmax(self.R, 1)
            loss = loss + self.alpha * M * (P.mean(0) ** 2).sum()
        self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()

    def predict(self, task, x):
        with torch.no_grad():
            return self.fwd(task, x, train=False)


CFG = {}
_build = exp.build


def build(variant, lr, hp):
    if variant == 'topk':
        return TopK(lr, **CFG)
    return _build(variant, lr, hp)


exp.build = build

if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]
    out = sys.argv[2]
    variants = sys.argv[3].split(',')
    CFG.update(json.loads(sys.argv[4]) if len(sys.argv) > 4 else {})
    with open(out, 'a') as f:
        for s in seeds:
            for v in variants:
                r = exp.run(v, s)
                r['cfg'] = dict(CFG) if v == 'topk' else None
                f.write(json.dumps(r) + '\n'); f.flush()
                print(v, s, CFG if v == 'topk' else '', round(r['train_r2'], 3), round(r['zero_r2'], 3),
                      round(r['k_train_rec'], 3), r['secs'], flush=True)
