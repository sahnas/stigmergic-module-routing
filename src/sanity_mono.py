# Control: can the monolithic MLP learn these tasks when everything is shuffled (iid)?
import numpy as np, torch, sys
from exp import World, Mono, mean_r2
for lr in [1e-3, 3e-3]:
    for seed in [100, 101]:
        torch.manual_seed(seed); w = World(seed); m = Mono(lr); rng = np.random.default_rng(seed)
        tt = w.train_tasks
        for step in range(5100):
            t = tt[rng.integers(len(tt))]; x, y = w.batch(t); m.train_step(t, x, y, w.var[t])
        print('iid lr', lr, 'seed', seed, 'train', round(mean_r2(m, w, tt), 3), 'zero', round(mean_r2(m, w, w.held), 3))
