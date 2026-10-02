"""Sequential toy world for the modular JEPA study: states, five primitives, per-step noise channels."""
import math
import numpy as np
import torch

D, NOISE, K = 8, 8, 5


class SeqWorld:
    def __init__(self, seed, n_held_pairs=8):
        g = np.random.default_rng(seed)
        self.prims = []
        for _ in range(K):
            Q, _ = np.linalg.qr(g.normal(size=(D, D)))
            B = g.normal(size=(D, D)) * 1.5 / math.sqrt(D)
            self.prims.append((torch.tensor(Q, dtype=torch.float32), torch.tensor(B, dtype=torch.float32)))
        pairs = [(i, j) for i in range(K) for j in range(K) if i != j]
        while True:
            perm = g.permutation(len(pairs))
            held = [pairs[p] for p in perm[:n_held_pairs]]
            train = [pairs[p] for p in perm[n_held_pairs:]]
            if len({i for i, _ in train}) == K and len({j for _, j in train}) == K:
                break
        self.train_pairs, self.held_pairs = train, held
        self.allowed = {i: [j for (a, j) in train if a == i] for i in range(K)}
        self.g = torch.Generator().manual_seed(seed + 11)
        self.rng = np.random.default_rng(seed + 13)

    def step(self, s, k):
        Q, B = self.prims[k]
        return s @ Q.T + 0.5 * torch.tanh(s @ B.T)

    def observe(self, s):
        return torch.cat([s, torch.randn(s.shape[0], NOISE, generator=self.g)], 1)

    def transitions(self, n, held=False):
        """n transitions (o_t, o_t+1, k) sampled i.i.d.; the primitive is drawn uniformly."""
        s = torch.randn(n, D, generator=self.g)
        k = torch.tensor(self.rng.integers(K, size=n))
        s1 = torch.stack([self.step(s[i:i + 1], int(k[i]))[0] for i in range(n)])
        return self.observe(s), self.observe(s1), k

    def two_step(self, n, pairs):
        """n two-step sequences using the given primitive pairs: returns o0, o1, o2, (i, j)."""
        s0 = torch.randn(n, D, generator=self.g)
        idx = self.rng.integers(len(pairs), size=n)
        ij = [pairs[t] for t in idx]
        s1 = torch.stack([self.step(s0[t:t + 1], i)[0] for t, (i, _) in enumerate(ij)])
        s2 = torch.stack([self.step(s1[t:t + 1], j)[0] for t, (_, j) in enumerate(ij)])
        return self.observe(s0), self.observe(s1), self.observe(s2), ij
