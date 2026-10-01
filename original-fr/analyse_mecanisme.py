import json, numpy as np
from scipy import stats
rows = [json.loads(l) for l in open('mecanisme.jsonl')]
def g(v, k):
    return np.array([r[k] for r in sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed'])], float)
def ev(v, e):
    return np.array([r['par_evenement'][e] for r in sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed'])])
def boot(d):
    b = np.random.default_rng(0).choice(d, (10000, len(d))).mean(1); return np.percentile(b, [2.5, 97.5])
print('graines', sorted({r['seed'] for r in rows}))
for v in ['aco_seuils', 'aco_seuils_hasard', 'aco_seuil_global']:
    di, ti, tf, al = g(v, 'disponibilite'), g(v, 'train_r2_init'), g(v, 'train_r2_final'), g(v, 'alarmes')
    print(f'{v:18s} dispo {di.mean():.3f} (med {np.median(di):.3f}) | init {ti.mean():.3f} effondr {int((ti<0.5).sum())}/20 | final {tf.mean():.3f} | alarmes {al.mean():.0f}')
    print('                   par evenement', {e: round(float(ev(v, e).mean()), 3) for e in ['E1', 'E2', 'E3', 'E4']})
for name, b in [('M1 moins occupe vs hasard', 'aco_seuils_hasard'), ('M2 seuils individuels vs collectif', 'aco_seuil_global')]:
    d = g('aco_seuils', 'disponibilite') - g(b, 'disponibilite'); p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
    verdict = 'CONFIRMEE' if (p < 0.025 and d.mean() > 0) else ('CONTREDITE' if (p < 0.025 and d.mean() < 0) else 'INDETERMINEE')
    print(f'{name}: diff {d.mean():+.3f} IC95 [{lo:+.3f}, {hi:+.3f}] positif {int((d>0).sum())}/20 p={p:.4f} -> {verdict}')
    for e in ['E1', 'E2', 'E3', 'E4']:
        de = ev('aco_seuils', e) - ev(b, e)
        print(f'   {e}: diff {de.mean():+.3f}, positif {int((de>0).sum())}/20')
