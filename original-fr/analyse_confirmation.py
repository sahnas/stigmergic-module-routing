import json, numpy as np
from scipy import stats
rows = [json.loads(l) for l in open('confirmation.jsonl')]
def g(v, k):
    return np.array([r[k] for r in sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed'])], float)
print('graines', sorted({r['seed'] for r in rows}))
for v in ['soft', 'aco_seuils', 'aco']:
    tr, ze, kr = g(v, 'train_r2'), g(v, 'zero_r2'), g(v, 'k_train_rec')
    print(f'{v:11s} retention moy {tr.mean():.3f} med {np.median(tr):.3f} min {tr.min():.3f} | '
          f'zero-shot moy {ze.mean():.3f} med {np.median(ze):.3f} min {ze.min():.3f} | k* recup moy {kr.mean():.3f}')
print()
for name, k in [('C1', 'zero_r2'), ('C2', 'train_r2')]:
    a, b = g('aco_seuils', k), g('soft', k); d = a - b
    p = stats.wilcoxon(a, b).pvalue
    boot = np.random.default_rng(0).choice(d, (10000, len(d))).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ok = p < 0.025 and d.mean() > 0
    print(f'{name} {k}: diff moy {d.mean():+.3f} (IC95 bootstrap {lo:+.3f} a {hi:+.3f}), '
          f'aco_seuils gagne {int((d > 0).sum())}/10, p={p:.4f} -> {"CONFIRMEE" if ok else "NON CONFIRMEE"}')
    print('   par graine', np.round(d, 3))
