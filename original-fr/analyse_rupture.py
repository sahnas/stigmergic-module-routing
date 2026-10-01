import json, numpy as np
from scipy import stats
rows = [json.loads(l) for l in open('rupture_test.jsonl')]
def g(v, k):
    return np.array([r[k] for r in sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed'])], float)
def ev(v, e):
    return np.array([r['par_evenement'][e] for r in sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed'])])
def boot(d):
    b = np.random.default_rng(0).choice(d, (10000, len(d))).mean(1); return np.percentile(b, [2.5, 97.5])
print('graines', sorted({r['seed'] for r in rows}), 'n par variante', len(g('ema', 'seed')), '\n')
for v in ['ema', 'rupture', 'aco_seuils']:
    di, ti, tf, zf = g(v, 'disponibilite'), g(v, 'train_r2_init'), g(v, 'train_r2_final'), g(v, 'zero_r2_final')
    print(f'{v:11s} dispo moy {di.mean():.3f} med {np.median(di):.3f} min {di.min():.3f} | init {ti.mean():.3f} effondr. {int((ti<0.5).sum())}/20 '
          f'| final {tf.mean():.3f} | zero-shot final {zf.mean():.3f}')
    print('            par evenement', {e: round(float(ev(v, e).mean()), 3) for e in ['E1', 'E2', 'E3', 'E4']})
print()
for name, a, b in [('R1', 'aco_seuils', 'rupture'), ('R2', 'rupture', 'ema')]:
    d = g(a, 'disponibilite') - g(b, 'disponibilite'); p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
    if name == 'R1':
        verdict = 'MAINTENUE' if (p < 0.025 and d.mean() > 0) else ('INVALIDEE' if (p < 0.025 and d.mean() < 0) else 'INDETERMINEE')
    else:
        verdict = 'CONFIRMEE' if (p < 0.025 and d.mean() > 0) else 'NON CONFIRMEE'
    print(f'{name} {a} - {b}: diff moy {d.mean():+.3f} IC95 [{lo:+.3f}, {hi:+.3f}] positif {int((d>0).sum())}/20 p={p:.5f} -> {verdict}')
# descriptif : graines saines au depart pour les deux
both = (g('rupture', 'train_r2_init') >= 0.8) & (g('aco_seuils', 'train_r2_init') >= 0.8)
d = g('aco_seuils', 'disponibilite')[both] - g('rupture', 'disponibilite')[both]
print(f'\nDescriptif : graines saines au depart pour les deux (n={both.sum()}) diff dispo {d.mean():+.3f}, positif {int((d>0).sum())}/{both.sum()}')
for v in ['rupture', 'aco_seuils']:
    print(f'  perte init->final {v}: {np.mean((g(v,"train_r2_init")-g(v,"train_r2_final"))[both]):.3f}')
