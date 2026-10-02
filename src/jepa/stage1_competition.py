"""
Stage 1: competing JEPA predictors; the object of study is the prior over the winner (the "when").

World: sticky regimes. A primitive persists with probability 0.8 per step, then switches to a primitive
allowed by the training pairs; held-out pairs are never used as switches during training.
Observation = [state, fresh noise]. Representation: shared encoder f and EMA target encoder.

Learning (competition variants, as in COMET): all M predictors predict every transition in representation
space; the winner of a transition is the predictor with the lowest error summed over a short horizon
(3 consecutive steps); only winners receive gradient, after a warm-start phase where all do.

Priors over the winner, used at prediction time (before the outcome is known):
  none       : stick with the previous hindsight winner (persistence)
  classifier : an MLP on f(o_t) trained on hindsight winners (COMET's composition module)
  traces     : tau[previous winner, m], evaporated on the solicited row and deposited on the hindsight winner
  attention  : no competition; a soft mixture over predictors with attention learned by the prediction loss
  oracle     : the true primitive (M = K); random: uniform; single: one predictor

Measures on test sequences: purity of hindsight winners against true primitives; relative latent error of the
module chosen by the prior, overall and on the steps after a switch (first step, then steps 2 and 3), for
switches on seen pairs and on held-out pairs; prior accuracy against the hindsight winner off switches.

Usage: python stage1_competition.py SEEDS OUT.jsonl VARIANTS
"""
import copy, json, math, sys, time
import numpy as np
import torch, torch.nn as nn
from world_seq import SeqWorld, D, NOISE, K

torch.set_num_threads(1)
Z, H, M, B, T = 8, 32, 8, 32, 12
STEPS, WARM, HORIZON, STAY = 3000, 300, 3, 0.8
EMA, LR, RHO = 0.99, 3e-3, 0.05


def sticky_sequences(world, n, T, rng, pairs=None, switch_at=None):
    """n sequences of T+1 observations; returns obs (n, T+1, D+NOISE) and primitives per step (n, T)."""
    s = torch.randn(n, D, generator=world.g)
    obs = [world.observe(s)]
    ks = np.zeros((n, T), dtype=int)
    if pairs is None:
        k = rng.integers(K, size=n)
    else:
        idx = rng.integers(len(pairs), size=n)
        k = np.array([pairs[t][0] for t in idx]); k2 = np.array([pairs[t][1] for t in idx])
    for t in range(T):
        if pairs is None:
            for i in range(n):
                if rng.random() > STAY:
                    k[i] = rng.choice(world.allowed[int(k[i])])
        elif t == switch_at:
            k = k2
        ks[:, t] = k
        s = torch.stack([world.step(s[i:i + 1], int(k[i]))[0] for i in range(n)])
        obs.append(world.observe(s))
    return torch.stack(obs, 1), ks


class Bank(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(m, Z, H) / math.sqrt(Z)); self.b1 = nn.Parameter(torch.zeros(m, H))
        self.W2 = nn.Parameter(torch.randn(m, H, Z) / math.sqrt(H)); self.b2 = nn.Parameter(torch.zeros(m, Z))

    def forward(self, z):  # (N, Z) -> (m, N, Z)
        h = torch.relu(torch.einsum('nz,mzh->mnh', z, self.W1) + self.b1[:, None, :])
        return torch.einsum('mnh,mhz->mnz', h, self.W2) + self.b2[:, None, :]


class Model:
    def __init__(self, variant, seed):
        self.variant = variant
        self.m = {'single': 1, 'oracle': K}.get(variant, M)
        self.enc = nn.Sequential(nn.Linear(D + NOISE, H), nn.ReLU(), nn.Linear(H, Z))
        self.tgt = copy.deepcopy(self.enc)
        for p in self.tgt.parameters():
            p.requires_grad_(False)
        self.bank = Bank(self.m)
        params = list(self.enc.parameters()) + list(self.bank.parameters())
        self.clf = nn.Sequential(nn.Linear(Z, H), nn.ReLU(), nn.Linear(H, self.m)) if variant == 'classifier' else None
        if self.clf is not None:
            params += list(self.clf.parameters())
        self.keys = nn.Parameter(torch.randn(self.m, Z) / math.sqrt(Z)) if variant == 'attention' else None
        if self.keys is not None:
            params.append(self.keys)
        self.opt = torch.optim.Adam(params, LR)
        self.tau = np.zeros((self.m + 1, self.m))   # row m = "start of sequence"
        self.rng = np.random.default_rng(seed + 3)
        self.step = 0

    def errors(self, obs):
        """obs (n, T+1, .) -> errors (m, n, T), preds (m, n, T, Z), target var (scalar)."""
        n, T1 = obs.shape[:2]
        z = self.enc(obs[:, :-1].reshape(-1, D + NOISE))
        with torch.no_grad():
            zt = self.tgt(obs[:, 1:].reshape(-1, D + NOISE))
        preds = self.bank(z)                                   # (m, N, Z)
        e = ((preds - zt[None]) ** 2).mean(-1).reshape(self.m, n, T1 - 1)
        return e, preds.reshape(self.m, n, T1 - 1, Z), zt.var(0).mean().item(), z.reshape(n, T1 - 1, Z)

    @staticmethod
    def smooth(e):
        """sum of errors over a window of HORIZON steps ending at t (causal)."""
        out = e.clone()
        for d in range(1, HORIZON):
            out[..., d:] += e[..., :-d]
        return out

    def hindsight(self, e):
        return torch.argmin(self.smooth(e), 0)                 # (n, T)

    def train_step(self, obs, ks):
        n = obs.shape[0]
        e, preds, var, z = self.errors(obs)
        if self.variant == 'attention':
            a = torch.softmax(z.reshape(-1, Z) @ self.keys.T / math.sqrt(Z), -1)      # (N, m)
            mix = torch.einsum('nm,mnz->nz', a, preds.reshape(self.m, -1, Z))
            with torch.no_grad():
                zt = self.tgt(obs[:, 1:].reshape(-1, D + NOISE))
            loss = ((mix - zt) ** 2).mean()
            w = torch.argmax(a, -1).reshape(n, -1)
        else:
            if self.variant == 'oracle':
                w = torch.tensor(ks)
            elif self.variant == 'random':
                w = torch.tensor(self.rng.integers(self.m, size=ks.shape))
            elif self.variant == 'single':
                w = torch.zeros(ks.shape, dtype=torch.long)
            else:
                w = self.hindsight(e.detach())
            if self.step < WARM and self.variant not in ('oracle', 'single'):
                loss = e.mean()
            else:
                loss = e.gather(0, w[None]).mean()
            if self.clf is not None:
                logits = self.clf(z.detach().reshape(-1, Z))
                loss = loss + nn.functional.cross_entropy(logits, w.reshape(-1))
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        with torch.no_grad():
            for pt, pe in zip(self.tgt.parameters(), self.enc.parameters()):
                pt.mul_(EMA).add_(pe, alpha=1 - EMA)
        if self.variant == 'traces':
            wn = w.numpy()
            for i in range(n):
                prev = self.m
                for t in range(wn.shape[1]):
                    self.tau[prev] *= (1 - RHO); self.tau[prev, wn[i, t]] += RHO
                    prev = wn[i, t]
        self.step += 1

    def choose(self, z, prev_w, ks_true):
        """module chosen by the prior at prediction time; z (n, Z), prev_w (n,) hindsight winner at t-1 (or m at start)."""
        n = z.shape[0]
        if self.variant == 'oracle':
            return torch.tensor(ks_true)
        if self.variant == 'random':
            return torch.tensor(self.rng.integers(self.m, size=n))
        if self.variant == 'single':
            return torch.zeros(n, dtype=torch.long)
        if self.variant == 'classifier':
            return torch.argmax(self.clf(z), -1)
        if self.variant == 'traces':
            return torch.tensor(np.argmax(self.tau[prev_w.numpy()], -1))
        if self.variant == 'attention':
            return torch.argmax(torch.softmax(z @ self.keys.T / math.sqrt(Z), -1), -1)
        return torch.where(prev_w == self.m, torch.zeros_like(prev_w), prev_w)   # none: persistence

    @torch.no_grad()
    def evaluate(self, obs, ks, switch_at):
        """returns per-step relative error of the prior's choice, prior accuracy, hindsight winners."""
        n, T1 = obs.shape[:2]
        e, preds, var, z = self.errors(obs)
        hw = self.hindsight(e) if self.variant not in ('oracle', 'single', 'random') else (torch.tensor(ks) if self.variant == 'oracle' else torch.zeros(n, T1 - 1, dtype=torch.long))
        if self.variant == 'attention':
            hw = torch.argmin(e, 0)
        inst = torch.argmin(e, 0)                                   # instantaneous hindsight, available one step later
        rel, acc = np.zeros((n, T1 - 1)), np.zeros((n, T1 - 1))
        prev = torch.full((n,), self.m, dtype=torch.long)
        for t in range(T1 - 1):
            c = self.choose(z[:, t], prev, ks[:, t])
            if self.variant == 'attention':
                a = torch.softmax(z[:, t] @ self.keys.T / math.sqrt(Z), -1)
                zt = self.tgt(obs[:, t + 1]); mix = torch.einsum('nm,mnz->nz', a, preds[:, :, t])
                rel[:, t] = (((mix - zt) ** 2).mean(-1) / var).numpy()
            else:
                rel[:, t] = (e[c, torch.arange(n), t] / var).numpy()
            acc[:, t] = (c == hw[:, t]).numpy()
            prev = inst[:, t]
        return rel, acc, hw.numpy()


def purity(hw, ks):
    """coverage: fraction of transitions explained by the majority module of each true primitive;
    module purity: fraction explained by the majority primitive of each module; number of modules used."""
    hw, ks = hw.reshape(-1), ks.reshape(-1)
    cov = sum(np.bincount(hw[ks == k]).max() for k in range(K) if (ks == k).any()) / len(hw)
    pur = sum(np.bincount(ks[hw == m]).max() for m in set(hw.tolist())) / len(hw)
    return cov, pur, len(set(hw.tolist()))


def run(variant, seed):
    torch.manual_seed(seed)
    world = SeqWorld(seed)
    rng = np.random.default_rng(seed + 5)
    model = Model(variant, seed)
    t0 = time.time()
    for _ in range(STEPS):
        obs, ks = sticky_sequences(world, B, T, rng)
        model.train_step(obs, ks)
    res = dict(variant=variant, seed=seed)
    # test 1: training-distribution sequences
    obs, ks = sticky_sequences(world, 256, T, rng)
    rel, acc, hw = model.evaluate(obs, ks, None)
    res['coverage'], res['module_purity'], res['distinct_modules'] = purity(hw, ks)
    res['rel_error_all'] = float(rel.mean())
    sw = np.zeros_like(ks, dtype=bool); sw[:, 1:] = ks[:, 1:] != ks[:, :-1]
    res['prior_acc_off_switch'] = float(acc[~sw].mean())
    # test 2: forced switches on seen pairs and on held-out pairs at step 6
    for name, pairs in [('seen', world.train_pairs), ('held', world.held_pairs)]:
        obs, ks = sticky_sequences(world, 256, T, rng, pairs=pairs, switch_at=6)
        rel, acc, hw = model.evaluate(obs, ks, 6)
        res[f'{name}_switch_step1'] = float(rel[:, 6].mean())
        res[f'{name}_switch_steps23'] = float(rel[:, 7:9].mean())
        res[f'{name}_before_switch'] = float(rel[:, 3:6].mean())
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
