import json, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else 'results.jsonl'
rows = [json.loads(l) for l in open(path)]
V = ['mono', 'oracle', 'soft', 'cumul', 'ema', 'aco', 'global', 'aco_seuils']
by = {v: sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed']) for v in V}
seeds = sorted({r['seed'] for r in rows})


def arr(v, key):
    return np.array([np.nan if r.get(key) is None else r[key] for r in by[v]], dtype=float)


def ci(x):
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan, len(x)
    m = x.mean(); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, h, len(x)


cols = [('train_r2', 'retention fin phase 1'), ('zero_r2', 'zero-shot compositions'),
        ('k_train_dmg', 'k* juste apres dommage'), ('k_train_rec', 'k* apres recuperation'),
        ('k_held_rec', 'zero-shot k* apres recup'), ('other_rec', 'reste apres recup'),
        ('reroute_latency', 'latence de reroutage (pas)'), ('secs', 'temps (s)')]
print(f'graines : {seeds}\n')
for key, label in cols:
    print(label)
    for v in V:
        m, h, n = ci(arr(v, key))
        med = np.nanmedian(arr(v, key)) if n else np.nan
        print(f'  {v:11s} moyenne {m:8.3f} +/- {h:6.3f}   mediane {med:8.3f}   n={n}')
    print()

for v in ['cumul', 'ema', 'aco', 'global', 'aco_seuils']:
    nd = arr(v, 'n_distinct_units'); al = arr(v, 'alarms_total'); lat = arr(v, 'reroute_latency')
    print(f'{v:11s} unites distinctes pour 5 primitives : {np.nanmean(nd):.1f}   alarmes (total) : {np.nanmean(al):.1f}'
          f'   reroutage reussi : {int(np.sum(~np.isnan(lat)))}/10')
print()

tests = [
    ('H1', 'aco', 'cumul', 'k_train_rec'), ('H1', 'aco', 'oracle', 'k_train_rec'),
    ('H2', 'aco_seuils', 'aco', 'k_train_rec'), ('H2', 'aco_seuils', 'aco', 'reroute_latency'),
    ('H3', 'aco', 'global', 'train_r2'),
    ('H4', 'aco', 'soft', 'zero_r2'), ('H4', 'aco', 'mono', 'zero_r2'),
    ('ctrl', 'aco', 'ema', 'k_train_rec'), ('ctrl', 'aco', 'ema', 'train_r2'),
    ('cout', 'aco', 'cumul', 'train_r2'),
]
alpha = 0.05 / len(tests)
print(f'Tests apparies de Wilcoxon (bilateraux), seuil Bonferroni {alpha:.4f}')
for h, a, b, key in tests:
    xa, xb = arr(a, key), arr(b, key)
    ok = ~np.isnan(xa) & ~np.isnan(xb)
    d = xa[ok] - xb[ok]
    if ok.sum() < 5 or np.all(d == 0):
        print(f'  {h:4s} {a} - {b} sur {key}: n={ok.sum()} insuffisant'); continue
    p = stats.wilcoxon(xa[ok], xb[ok]).pvalue
    wins = int(np.sum(d > 0))
    print(f'  {h:4s} {a:10s} - {b:7s} {key:16s} diff moyenne {d.mean():+.3f} (mediane {np.median(d):+.3f}) '
          f'{a} > {b} sur {wins}/{ok.sum()} graines  p={p:.4f} {"*" if p < alpha else ""}')

# Figure : recuperation apres la mort de l'unite qui porte k*
plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(figsize=(11, 5))
styles = {'oracle': ('#937860', ':'), 'soft': ('#8172B3', '-.'), 'cumul': ('#C44E52', '--'),
          'ema': ('#DD8452', '--'), 'aco': ('#4C72B0', '-'), 'aco_seuils': ('#55A868', '-')}
labels = {'oracle': 'orchestration fixe', 'soft': 'routeur appris (soft)', 'cumul': 'traces sans oubli',
          'ema': 'oubli par récence', 'aco': 'évaporation (fourmis)', 'aco_seuils': 'évaporation + seuils'}
for v, (c, ls) in styles.items():
    curves = [r['curve'] for r in by[v]]
    steps = [s for s, _ in curves[0]]
    Y = np.array([[y for _, y in cv] for cv in curves])
    m = Y.mean(0); lo, hi = np.percentile(Y, 25, 0), np.percentile(Y, 75, 0)
    ax.plot(steps, m, color=c, linestyle=ls, linewidth=2, label=labels[v])
    ax.fill_between(steps, lo, hi, color=c, alpha=0.12)
ax.set_xlabel("pas d'apprentissage après la mort de l'unité")
ax.set_ylabel('R² moyen sur les tâches impliquant k*')
ax.set_ylim(-0.3, 1.02)
ax.set_title("Compensation après perte définitive d'une unité\n10 graines de test, moyenne et quartiles")
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
plt.tight_layout()
plt.savefig('recuperation.png', dpi=150, bbox_inches='tight')
print('\nfigure : recuperation.png')
