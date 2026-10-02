"""Stage 0 (sanity, not a hypothesis test): one JEPA predictor with the true primitive given as input.
Checks that representation-space prediction works in this world and does not collapse.
Usage: python stage0_sanity.py SEED [variance_weight]"""
import copy, sys
import torch, torch.nn as nn
from world_seq import SeqWorld, D, NOISE, K

torch.set_num_threads(1)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
VW = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
torch.manual_seed(seed)
world = SeqWorld(seed)
H, Z = 32, 8
enc = nn.Sequential(nn.Linear(D + NOISE, H), nn.ReLU(), nn.Linear(H, Z))
tgt = copy.deepcopy(enc)
for p in tgt.parameters():
    p.requires_grad_(False)
pred = nn.Sequential(nn.Linear(Z + K, H), nn.ReLU(), nn.Linear(H, Z))
opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), 3e-3)
EMA = 0.99


def forward(o0, k):
    z0 = enc(o0)
    kk = torch.nn.functional.one_hot(k, K).float()
    return pred(torch.cat([z0, kk], 1)), z0


def diagnostics(z):
    zc = z - z.mean(0)
    var = zc.var(0).mean().item()
    sv = torch.linalg.svdvals(zc)
    p = sv / sv.sum()
    erank = torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()
    return var, erank


for step in range(1, 4001):
    o0, o1, k = world.transitions(64)
    with torch.no_grad():
        z1 = tgt(o1)
    zhat, z0 = forward(o0, k)
    loss = ((zhat - z1) ** 2).mean()
    if VW > 0:
        std = torch.sqrt(z0.var(0) + 1e-4)
        loss = loss + VW * torch.relu(1 - std).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        for pt, pe in zip(tgt.parameters(), enc.parameters()):
            pt.mul_(EMA).add_(pe, alpha=1 - EMA)
    if step % 1000 == 0:
        with torch.no_grad():
            o0, o1, k = world.transitions(1000)
            z1 = tgt(o1); zhat, z0 = forward(o0, k)
            err = ((zhat - z1) ** 2).mean().item() / (z1.var(0).mean().item() + 1e-9)
            var, er = diagnostics(z0)
            # linear probe: does the representation keep the state and drop the noise?
            X = torch.cat([z0, torch.ones(len(z0), 1)], 1)
            for name, target in [('state', o0[:, :D]), ('noise', o0[:, D:])]:
                W = torch.linalg.lstsq(X, target).solution
                r2 = 1 - ((X @ W - target) ** 2).mean().item() / target.var(0).mean().item()
                print(f'step {step} | relative latent prediction error {err:.3f} | latent var {var:.3f} eff. rank {er:.2f} | probe {name} R2 {r2:.3f}')
