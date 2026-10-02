"""
Stage 2 (experiment 13): lifelong discovery, COMET's stated open problem. After stage-1 training on five
primitives, a sixth primitive appears and training continues. Does the system give it a module of its own
without damaging the existing mechanisms?

  competition5 : five active predictors, no reserve. Plain competition continues (COMET): the hindsight winner
                 learns, whichever module it is.
  competition8 : eight active predictors from the start (three spares trained as generalists during warm-start).
                 Plain competition continues.
  recruit5     : five active predictors plus three dormant reserves, excluded from the competition and never
                 trained. Surprise detection: the instantaneous best error among active modules exceeds the
                 running level of instantaneous best errors by three deviations (collective threshold; stats
                 frozen at the end of stage 1). When more than 10 % of a batch's transitions are surprises, one
                 dormant reserve is activated and becomes the forced winner on surprise transitions; once its
                 own error is below the threshold, it competes like the others. This instantiates a new
                 mechanism, which COMET cannot do.
All use the persistence prior at prediction time (stage 1 found traces and persistence indistinguishable here).
Instantaneous (not smoothed) errors are used for surprise detection so that switches to known primitives,
which the right module predicts at once, are not mistaken for novelty.

Phases: stage 1 training (3000 steps, five primitives) -> error statistics -> lifelong phase (1000 steps, sixth
primitive added, allowed to switch with every existing one) -> evaluation.

Measures:
  new_purity        : fraction of the sixth primitive's transitions explained by its majority module
  new_module_prior  : fraction of all transitions that this majority module explained before the lifelong
                      phase (0 = a free or reserve module was used, high = a specialist was hijacked)
  new_module_was_specialist : 1 if that module was the majority module of an old primitive
  interference      : increase of the relative latent error on the old primitives (after - before)
  rel_error_new     : relative latent error on the sixth primitive after the lifelong phase
  old_purity_after  : module purity on the old primitives after the lifelong phase
Usage: python stage2_lifelong.py SEEDS OUT.jsonl VARIANTS
"""
import copy, json, math, sys, time
import numpy as np
import torch, torch.nn as nn
import world_seq
from world_seq import SeqWorld, D, NOISE, K
import stage1_competition as s1
from stage1_competition import Model, sticky_sequences, purity, B, T, STEPS, HORIZON

torch.set_num_threads(1)
LIFE_STEPS, KAPPA = 1000, 3.0


class World6(SeqWorld):
    """Stage-1 world plus a sixth primitive, added only when extend() is called."""
    def extend(self, seed):
        g = np.random.default_rng(seed + 777)
        Q, _ = np.linalg.qr(g.normal(size=(D, D)))
        Bm = g.normal(size=(D, D)) * 1.5 / math.sqrt(D)
        self.prims.append((torch.tensor(Q, dtype=torch.float32), torch.tensor(Bm, dtype=torch.float32)))
        for i in range(K):
            self.allowed[i] = self.allowed[i] + [K]
        self.allowed[K] = list(range(K))


class Lifelong(Model):
    def __init__(self, variant, seed):
        super().__init__('none', seed)
        self.variant_life = variant
        self.active = np.ones(self.m, dtype=bool)
        if variant in ('competition5', 'recruit5'):
            self.active[K:] = False
        self.mu, self.dev, self.nwin = 0.0, 0.0, 0
        self.surprises, self.activations = 0, 0
        self.recruit = None

    def masked(self, e):
        e = e.clone()
        e[torch.tensor(~self.active)] = float('inf')
        return e

    def train_step(self, obs, ks, lifelong=False):
        if self.variant_life == 'competition8':
            return super().train_step(obs, ks)
        n = obs.shape[0]
        e, preds, var, z = self.errors(obs)
        em = self.masked(e.detach())
        w = torch.argmin(self.smooth(em), 0)
        if self.step < s1.WARM:
            loss = e[torch.tensor(self.active)].mean()
        else:
            if lifelong and self.variant_life == 'recruit5':
                best = em.min(0).values
                thr = self.mu + KAPPA * max(self.dev, 1e-3 * max(self.mu, 1e-6))
                surprise = best > thr
                frac = float(surprise.float().mean())
                if self.recruit is None and frac > 0.10 and (~self.active).any():
                    self.recruit = int(np.flatnonzero(~self.active)[0])
                    self.active[self.recruit] = True
                    self.activations += 1
                if self.recruit is not None:
                    w = torch.where(surprise, torch.full_like(w, self.recruit), w)
                    self.surprises += int(surprise.sum())
                    if float((e[self.recruit].detach() > thr).float().mean()) < 0.5:
                        self.recruit = None            # integrated: competes like the others
            loss = e.gather(0, w[None]).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        with torch.no_grad():
            for pt, pe in zip(self.tgt.parameters(), self.enc.parameters()):
                pt.mul_(s1.EMA).add_(pe, alpha=1 - s1.EMA)
        self.step += 1

    def warm_stats(self, obs, ks):
        """error statistics of the instantaneous best error at the end of stage 1, without learning."""
        with torch.no_grad():
            e, _, _, _ = self.errors(obs)
            best = self.masked(e).min(0).values.reshape(-1).tolist()
        a = 0.05
        for v in best:
            if self.nwin == 0:
                self.mu = v
            self.dev = (1 - a) * self.dev + a * abs(v - self.mu)
            self.mu = (1 - a) * self.mu + a * v
            self.nwin += 1

    @torch.no_grad()
    def evaluate(self, obs, ks, switch_at):
        """as in stage 1, with dormant modules excluded from hindsight and persistence."""
        n, T1 = obs.shape[:2]
        e, preds, var, z = self.errors(obs)
        em = self.masked(e)
        hw = torch.argmin(self.smooth(em), 0)
        inst = torch.argmin(em, 0)
        rel, acc = np.zeros((n, T1 - 1)), np.zeros((n, T1 - 1))
        prev = torch.full((n,), self.m, dtype=torch.long)
        first = int(np.flatnonzero(self.active)[0])
        for t in range(T1 - 1):
            c = torch.where(prev == self.m, torch.full_like(prev, first), prev)
            rel[:, t] = (e[c, torch.arange(n), t] / var).numpy()
            acc[:, t] = (c == hw[:, t]).numpy()
            prev = inst[:, t]
        return rel, acc, hw.numpy()


def errors_by_primitive(model, world, rng, n=256, prims=None):
    with torch.no_grad():
        obs, ks = sticky_sequences(world, n, T, rng)
        rel, acc, hw = model.evaluate(obs, ks, None)
    out = {}
    for k in range(len(world.prims)):
        mask = ks == k
        if mask.any():
            out[k] = float(rel[mask].mean())
    return out, hw, ks


def run(variant, seed):
    torch.manual_seed(seed)
    world = World6(seed)
    rng = np.random.default_rng(seed + 5)
    model = Lifelong(variant, seed)
    t0 = time.time()
    for _ in range(STEPS):
        obs, ks = sticky_sequences(world, B, T, rng)
        model.train_step(obs, ks)
    for _ in range(50):
        obs, ks = sticky_sequences(world, B, T, rng)
        model.warm_stats(obs, ks)
    err_before, hw_before, ks_before = errors_by_primitive(model, world, rng)
    # which module explains which old primitive, before the sixth appears
    maj = {k: int(np.bincount(hw_before[ks_before == k]).argmax()) for k in range(K)}
    explained_before = {m: float(((hw_before == m)).mean()) for m in range(model.m)}
    world.extend(seed)
    for _ in range(LIFE_STEPS):
        obs, ks = sticky_sequences(world, B, T, rng)
        model.train_step(obs, ks, lifelong=True)
    err_after, hw_after, ks_after = errors_by_primitive(model, world, rng)
    res = dict(variant=variant, seed=seed)
    new = ks_after == K
    counts = np.bincount(hw_after[new], minlength=model.m)
    mnew = int(counts.argmax())
    res['new_purity'] = float(counts.max() / counts.sum())
    res['new_module_prior'] = explained_before[mnew]
    res['new_module_was_specialist'] = int(mnew in maj.values())
    res['rel_error_new'] = err_after[K]
    old_before = float(np.mean([err_before[k] for k in range(K)]))
    old_after = float(np.mean([err_after[k] for k in range(K)]))
    res['interference'] = old_after - old_before
    res['old_error_before'], res['old_error_after'] = old_before, old_after
    cov, pur, nd = purity(hw_after[~new], ks_after[~new])
    res['old_purity_after'] = pur
    res['surprises'] = model.surprises
    res['activations'] = model.activations
    res['active_modules'] = int(model.active.sum())
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
                print(v, s, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items() if k not in ('variant', 'seed')}, flush=True)
