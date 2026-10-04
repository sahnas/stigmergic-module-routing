"""Local smoke test of pilot_lib on a synthetic stand-in (random embeddings with a linear-plus-tanh dynamics in
the action), to check that the code runs end to end. Not a result."""
import numpy as np, torch, torch.nn as nn, json
import pilot_lib as P
rng = np.random.default_rng(0)
n_ep, L, D = 60, 40, 16
W = rng.normal(size=(D, D)) * 0.3; U = rng.normal(size=(D, 2))
emb = np.zeros((n_ep * L, D), np.float32); act = rng.uniform(-1, 1, size=(n_ep * L, 2)).astype(np.float32)
ep_len = np.full(n_ep, L); ep_off = np.arange(n_ep) * L
for e in range(n_ep):
    z = rng.normal(size=D)
    for t in range(L):
        emb[e * L + t] = z; z = np.tanh(z @ W + act[e * L + t] @ U.T)
data = P.Data(emb, act, ep_len, ep_off, seed=0).to('cpu')
class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(P.H * (D + 10), 64), nn.GELU(), nn.Linear(64, D))
    def forward(self, h, a): return self.net(torch.cat([h.flatten(1), a.flatten(1)], 1)) + h[:, -1]
res = P.run_pilot(data, MLP, seed=0, n1=60, n2=90, n3=30, lr=3e-3, log_every=30)
print(json.dumps({k: v['marks']['P3'] for k, v in res['systems'].items()}, indent=1)[:1500])
