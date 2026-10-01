# Reglage de l'adversaire sur graines de developpement uniquement (300, 301, 302).
import json, sys, itertools
import exp_rupture as er
import exp_dynamique as ed
configs = [dict(sel='ucb', c=c, h=h, b=b) for c, h, b in itertools.product([0.03, 0.1, 0.3], [20, 50], [0.2, 0.4])]
configs += [dict(sel='softmax', h=h, b=b) for h, b in itertools.product([20, 50], [0.2, 0.4])]
part = int(sys.argv[1]); nparts = int(sys.argv[2])
seeds = [300, 301, 302]
with open('grid_rupture.jsonl', 'a') as f:
    for i, cfg in enumerate(configs):
        if i % nparts != part:
            continue
        for s in seeds:
            er.CFG.clear(); er.CFG.update(cfg)
            r = ed.run('rupture', s)
            f.write(json.dumps({'cfg': cfg, 'seed': s, 'disponibilite': r['disponibilite'],
                                'train_r2_init': r['train_r2_init']}) + '\n'); f.flush()
        print(i, cfg, flush=True)
