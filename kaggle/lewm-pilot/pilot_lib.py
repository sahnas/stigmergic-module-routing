"""
Continual dynamics on LeWM's frozen representation: pilot library (design/kaggle_continual_lewm.md).

Stream: episodes of the Two-rooms dataset, each turned into sequential windows (history 3 at frameskip 5,
action blocks of 5, target 5 steps later). Regime A: original actions; regime B: actions rotated by 90 degrees
before the fixed z-score normalisation (same images, other action-to-transition mapping).
Phases: P1 A only; P2 B with A at 2 %; P3 A only. Regime labels are used for evaluation only.
Systems (same predictor architecture, same episodes, same number of gradient steps per episode):
  single            one predictor
  single_replay     one predictor with a reservoir buffer replayed on every episode
  bank_persistence  K predictors, hindsight competition, persistence prior, freezing, held-back reserves,
                    random recruitment among unfrozen active modules when no reserve is left
  bank_traces       same, with a trace prior keyed by the previous winner and least-committed recruitment
Measures on held-out episodes: content (oracle best module), access (module chosen before observing),
relearning during P3 (compared with a fresh predictor on the same P3 stream).
"""
import copy, json, math, time
import numpy as np
import torch
import torch.nn as nn

H, FS = 3, 5
SPAN = H * FS
R90 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)   # a_B = R a_A, bounds preserved


class Data:
    def __init__(self, emb, act, ep_len, ep_off, seed, test_frac=0.1):
        self.emb = torch.as_tensor(emb, dtype=torch.float32)
        rng = np.random.default_rng(seed)
        eps = np.arange(len(ep_len)); self.test_eps = np.sort(rng.choice(eps, size=max(1, len(eps) // 10), replace=False))
        self.train_eps = np.array([e for e in eps if e not in set(self.test_eps.tolist())])
        tr_frames = np.concatenate([np.arange(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e])) for e in self.train_eps])
        act = np.nan_to_num(np.asarray(act, dtype=np.float32))
        self.a_mean, self.a_std = act[tr_frames].mean(0), act[tr_frames].std(0) + 1e-8
        self.act_raw = act
        self.actA = torch.as_tensor((act - self.a_mean) / self.a_std, dtype=torch.float32)
        self.actB = torch.as_tensor((act @ R90.T - self.a_mean) / self.a_std, dtype=torch.float32)
        self.ep_len, self.ep_off = ep_len, ep_off
        self.windows = {int(e): np.arange(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e]) - SPAN) for e in eps if int(ep_len[e]) > SPAN}
        self.train_eps = np.array([e for e in self.train_eps if int(e) in self.windows]); self.test_eps = np.array([e for e in self.test_eps if int(e) in self.windows])

    def to(self, dev):
        self.emb, self.actA, self.actB = self.emb.to(dev), self.actA.to(dev), self.actB.to(dev); self.dev = dev; return self

    def episode(self, e, regime):
        idx = torch.as_tensor(self.windows[int(e)], device=self.dev)
        act = self.actA if regime == 'A' else self.actB
        hist = torch.stack([self.emb[idx + k * FS] for k in range(H)], 1)
        blocks = torch.stack([torch.cat([act[idx + k * FS + j] for j in range(FS)], 1) for k in range(H)], 1)
        return hist, blocks, self.emb[idx + H * FS]


def rel(p, y, var):
    return ((p - y) ** 2).mean(-1) / var          # per-window relative error


class System:
    def __init__(self, make, lr, K=1):
        self.make, self.lr = make, lr
        self.mods = [make() for _ in range(K)]
        self.opts = [torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-3) for m in self.mods]
        self.forwards = 0

    def step_on(self, i, h, a, y, mask=None):
        m, opt = self.mods[i], self.opts[i]
        if mask is not None and mask.sum() == 0:
            return
        hh, aa, yy = (h, a, y) if mask is None else (h[mask], a[mask], y[mask])
        loss = ((m(hh, aa) - yy) ** 2).mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); self.forwards += 1

    def predict_access(self, h, a, y, var):
        """error of the module chosen before observing the outcome, per window (sequential within the episode)"""
        raise NotImplementedError

    def errors_all(self, h, a, y, var):
        with torch.no_grad():
            return torch.stack([rel(m(h, a), y, var) for m in self.mods], 0)   # (K, n)


class Single(System):
    def __init__(self, make, lr, replay=0, replay_size=20000, seed=0):
        super().__init__(make, lr, 1)
        self.replay, self.buf, self.seen, self.rng = replay, [], 0, np.random.default_rng(seed)

    def train_episode(self, h, a, y, var):
        if self.replay:
            if self.buf:
                j = self.rng.choice(len(self.buf), size=min(len(self.buf), len(h)), replace=False)
                bh = torch.stack([self.buf[t][0] for t in j]); ba = torch.stack([self.buf[t][1] for t in j]); by = torch.stack([self.buf[t][2] for t in j])
                self.step_on(0, torch.cat([h, bh]), torch.cat([a, ba]), torch.cat([y, by]))
            else:
                self.step_on(0, h, a, y)
            for t in range(len(h)):           # reservoir sampling
                self.seen += 1
                if len(self.buf) < self.replay: self.buf.append((h[t], a[t], y[t]))
                else:
                    r = self.rng.integers(self.seen)
                    if r < self.replay: self.buf[r] = (h[t], a[t], y[t])
        else:
            self.step_on(0, h, a, y)
        return dict(surprise=0, activated=0)

    def predict_access(self, h, a, y, var):
        with torch.no_grad():
            return rel(self.mods[0](h, a), y, var)


class Bank(System):
    def __init__(self, make, lr, K=4, prior='persistence', freeze_thr=0.02, freeze_eps=20, surprise_thr=0.3, rho=0.05, seed=0):
        super().__init__(make, lr, K)
        self.K, self.prior, self.freeze_thr, self.freeze_eps, self.surprise_thr, self.rho = K, prior, freeze_thr, freeze_eps, surprise_thr, rho
        self.active = np.zeros(K, bool); self.active[0] = True
        self.frozen = np.zeros(K, bool); self.ema = np.full(K, np.nan); self.good_run = np.zeros(K, int)
        self.tau = np.zeros((K, K)); self.prev = 0; self.rng = np.random.default_rng(seed)
        self.n_activated, self.n_recruits, self.n_surprise = 0, 0, 0

    def choose(self, prev):
        act = np.where(self.active)[0]
        if self.prior == 'traces' and self.tau[prev, act].max() > 0:
            return int(act[np.argmax(self.tau[prev, act])])
        return int(prev) if self.active[prev] else int(act[0])

    def train_episode(self, h, a, y, var):
        act = np.where(self.active)[0]
        err = self.errors_all(h, a, var=var, y=y)[act]           # (n_active, n)
        self.forwards += len(act)
        best = err.min(0).values                                  # hindsight per window
        winner = torch.as_tensor(act, device=err.device)[err.argmin(0)]
        surprise = int(best.mean().item() > self.surprise_thr)
        activated = 0
        if surprise:
            self.n_surprise += 1
            reserves = np.where(~self.active)[0]
            if len(reserves):
                r = int(reserves[0]); self.active[r] = True; self.n_activated += 1; activated = 1
            else:
                cands = np.where(self.active & ~self.frozen)[0]
                if len(cands):
                    r = int(cands[np.argmin(self.tau.sum(0)[cands])]) if self.prior == 'traces' else int(self.rng.choice(cands))
                    self.n_recruits += 1
                else:
                    r = None
            if r is not None:
                winner = torch.full_like(winner, r)
        for i in np.where(self.active)[0]:
            mask = winner == i
            if mask.any():
                e_i = rel(self.mods[i](h[mask], a[mask]).detach(), y[mask], var).mean().item() if not self.frozen[i] else err[list(act).index(i)][mask].mean().item()
                self.ema[i] = e_i if np.isnan(self.ema[i]) else 0.9 * self.ema[i] + 0.1 * e_i
                if not self.frozen[i]:
                    self.step_on(i, h, a, y, mask)
                    self.good_run[i] = self.good_run[i] + 1 if self.ema[i] < self.freeze_thr else 0
                    if self.good_run[i] >= self.freeze_eps:
                        self.frozen[i] = True
        # traces keyed by the previous winner: evaporate the row, reinforce the episode's majority winner
        w = int(torch.mode(winner).values.item())
        self.tau[self.prev] *= (1 - self.rho); self.tau[self.prev, w] += self.rho * max(0.0, 1.0 - best.mean().item())
        self.prev = w
        return dict(surprise=surprise, activated=activated)

    def predict_access(self, h, a, y, var):
        """sequential: choose by prior from the previous window's hindsight winner, observe, move on"""
        act = np.where(self.active)[0]
        err = self.errors_all(h, a, var=var, y=y)                 # (K, n), K forward passes
        out = torch.empty(len(h), device=h.device); prev = self.prev
        for t in range(len(h)):
            m = self.choose(prev); out[t] = err[m, t]
            prev = int(act[err[act, t].argmin().item()])
        return out


def evaluate(system, data, eps, regime, var, n_eps=60):
    acc, con = [], []
    for e in eps[:n_eps]:
        h, a, y = data.episode(e, regime)
        acc.append(system.predict_access(h, a, y, var).mean().item())
        con.append(system.errors_all(h, a, y, var).min(0).values.mean().item())
    return float(np.mean(acc)), float(np.mean(con))


def run_pilot(data, make, seed, n1=2000, n2=3000, n3=1000, p_rare=0.02, lr=3e-4, systems=('single', 'single_replay', 'bank_persistence', 'bank_traces'), log_every=100, var=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed + 1)
    train = data.train_eps.copy(); rng.shuffle(train)
    sched = [(train[i % len(train)], 'A') for i in range(n1)] + [(train[(n1 + i) % len(train)], 'A' if rng.random() < p_rare else 'B') for i in range(n2)] + [(train[(n1 + n2 + i) % len(train)], 'A') for i in range(n3)]
    if var is None:
        with torch.no_grad():
            ys = torch.cat([data.episode(e, 'A')[2] for e in data.test_eps[:60]]); var = ys.var(0).mean().item()
    out = {}
    for name in systems:
        torch.manual_seed(seed)
        sysm = Single(make, lr, replay=0) if name == 'single' else Single(make, lr, replay=20000, seed=seed) if name == 'single_replay' else Bank(make, lr, prior='persistence', seed=seed) if name == 'bank_persistence' else Bank(make, lr, prior='traces', seed=seed)
        fresh = Single(make, lr); log = []; t0 = time.time(); marks = {}
        for step, (e, regime) in enumerate(sched):
            h, a, y = data.episode(e, regime)
            sysm.train_episode(h, a, y, var)
            if step >= n1 + n2:
                fresh.train_episode(h, a, y, var)
            phase = 'P1' if step < n1 else 'P2' if step < n1 + n2 else 'P3'
            if (step + 1) % log_every == 0 or step + 1 in (n1, n1 + n2, len(sched)):
                accA, conA = evaluate(sysm, data, data.test_eps, 'A', var); accB, conB = evaluate(sysm, data, data.test_eps, 'B', var)
                rec = dict(step=step + 1, phase=phase, accessA=accA, contentA=conA, accessB=accB, contentB=conB, forwards=sysm.forwards)
                if step >= n1 + n2:
                    rec['freshA'] = evaluate(fresh, data, data.test_eps, 'A', var)[0]
                if isinstance(sysm, Bank):
                    rec.update(active=int(sysm.active.sum()), frozen=int(sysm.frozen.sum()), surprises=sysm.n_surprise, activated=sysm.n_activated, recruits=sysm.n_recruits)
                log.append(rec)
                if step + 1 in (n1, n1 + n2, len(sched)): marks[phase] = rec
        out[name] = dict(log=log, marks=marks, seconds=round(time.time() - t0))
        print(name, {k: {m: round(v[m], 4) for m in ('accessA', 'contentA', 'accessB', 'contentB')} for k, v in marks.items()}, 'P3 fresh', round(marks['P3'].get('freshA', float('nan')), 4), f"{out[name]['seconds']}s", flush=True)
    return dict(seed=seed, var=var, n1=n1, n2=n2, n3=n3, p_rare=p_rare, systems=out)
