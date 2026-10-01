# Analyse du protocole preregistre dans preregistration_bandits.md (B1 a B3).
import json, numpy as np
from scipy import stats
cfg = json.load(open('bandits_retenus.json'))
def load(f):
    return [json.loads(l) for l in open(f)]
def g(rows, v, k='disponibilite'):
    return np.array([r[k] for r in sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed'])], float)
def boot(d):
    b = np.random.default_rng(0).choice(d, (10000, len(d))).mean(1); return np.percentile(b, [2.5, 97.5])
for titre, f in [('TEST PRINCIPAL, graines 80 a 99', 'bandits_test2.jsonl'), ('REPLICATION, graines 60 a 74 (execution interrompue)', 'test_bandits.jsonl')]:
    rows = load(f)
    print(titre, '| graines', sorted({r['seed'] for r in rows})[0], 'a', sorted({r['seed'] for r in rows})[-1], 'n =', len(g(rows, 'aco_seuils')))
    for v in ['aco_seuils'] + list(cfg.values()):
        di, ti, tf = g(rows, v), g(rows, v, 'train_r2_init'), g(rows, v, 'train_r2_final')
        print(f'  {v:18s} dispo {di.mean():.3f} (med {np.median(di):.3f}) | init {ti.mean():.3f} | final {tf.mean():.3f}')
    for name, (k, v) in zip(['B1', 'B2', 'B3'], cfg.items()):
        d = g(rows, 'aco_seuils') - g(rows, v); p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
        verdict = 'CONFIRMEE' if (p < 0.05 / 3 and d.mean() > 0) else ('CONTREDITE' if p < 0.05 / 3 else 'NON CONFIRMEE')
        print(f'  {name} aco_seuils - {v}: diff {d.mean():+.3f} IC95 [{lo:+.3f}, {hi:+.3f}] positif {int((d>0).sum())}/{len(d)} p={p:.5f} -> {verdict}')
    print()
