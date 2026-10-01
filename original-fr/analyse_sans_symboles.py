import json, numpy as np
from scipy import stats
rows = [json.loads(l) for l in open('sans_symboles.jsonl')]
V = ['mono', 'soft', 'rl', 'aco_seuils']
def g(v, k):
    return np.array([r[k] if r[k] is not None else np.nan for r in sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed'])], float)
def boot(d):
    b = np.random.default_rng(0).choice(d, (10000, len(d))).mean(1); return np.percentile(b, [2.5, 97.5])
print('graines', sorted({r['seed'] for r in rows}))
for v in V:
    pa, auc, fin, itf, al = g(v, 'phaseA_r2'), g(v, 'fewshot_auc'), g(v, 'fewshot_final'), g(v, 'interference'), g(v, 'alignement')
    comp = (fin + 12 * (pa - itf)) / 13
    print(f'{v:11s} phaseA {pa.mean():.3f} | vitesse {auc.mean():.3f} | final nouvelle {fin.mean():.3f} | interference {itf.mean():.3f} '
          f'| capacite globale (non tronquee) {comp.mean():.3f} | alignement {np.nanmean(al) if not np.all(np.isnan(al)) else float("nan"):.3f}')
print()
def test(name, a, b, k, sens, alpha):
    d = g(a, k) - g(b, k); p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
    good = d.mean() > 0 if sens > 0 else d.mean() < 0
    verdict = 'CONFIRMEE' if (p < alpha and good) else ('CONTREDITE' if p < alpha else 'INDETERMINEE')
    print(f'{name}: {a} - {b} sur {k}: diff {d.mean():+.3f} IC95 [{lo:+.3f}, {hi:+.3f}] {a} plus haut sur {int((d>0).sum())}/10 p={p:.4f} -> {verdict}')
for n, b in [('N1', 'soft'), ('N2', 'rl'), ('N3', 'mono')]:
    test(n, 'aco_seuils', b, 'fewshot_auc', +1, 0.05 / 3)
for n, b in [('I1', 'soft'), ('I2', 'mono')]:
    test(n, 'aco_seuils', b, 'interference', -1, 0.025)
