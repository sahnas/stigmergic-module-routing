import json, numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open('sans_gradient.jsonl')]
V = ['ema', 'ema_seuils', 'aco', 'aco_seuils']
def g(v, k):
    return np.array([r[k] for r in sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed'])], float)
def ev(v, e):
    return np.array([r['par_evenement'][e] for r in sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed'])])
def boot(d):
    b = np.random.default_rng(0).choice(d, (10000, len(d))).mean(1)
    return np.percentile(b, [2.5, 97.5])

print('graines', sorted({r['seed'] for r in rows}), '\n')
for v in V:
    di, ti, tf, zf, al = g(v, 'disponibilite'), g(v, 'train_r2_init'), g(v, 'train_r2_final'), g(v, 'zero_r2_final'), g(v, 'alarmes')
    print(f'{v:11s} dispo moy {di.mean():.3f} med {np.median(di):.3f} | init moy {ti.mean():.3f} effondrements init {int((ti<0.5).sum())}/20 '
          f'| final {tf.mean():.3f} | zero-shot final {zf.mean():.3f} | alarmes {al.mean():.0f}')
    print('            par evenement', {e: round(ev(v, e).mean(), 3) for e in ['E1', 'E2', 'E3', 'E4']})
print()
alpha = 0.05 / 3
S1 = g('aco_seuils', 'disponibilite') - g('ema', 'disponibilite')
S2 = ((g('ema_seuils', 'disponibilite') - g('ema', 'disponibilite')) + (g('aco_seuils', 'disponibilite') - g('aco', 'disponibilite'))) / 2
S3 = ((g('aco', 'disponibilite') - g('ema', 'disponibilite')) + (g('aco_seuils', 'disponibilite') - g('ema_seuils', 'disponibilite'))) / 2
for name, d, sens in [('S1 aco_seuils - ema', S1, 1), ('S2 effet des seuils', S2, 1), ('S3 effet regle (aco - ema)', S3, 0)]:
    p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
    ok = p < alpha and (sens == 0 or d.mean() > 0)
    print(f'{name:28s} diff moy {d.mean():+.3f} IC95 [{lo:+.3f}, {hi:+.3f}] positif {int((d>0).sum())}/20 p={p:.5f} -> {"CONFIRMEE" if ok else "NON CONFIRMEE"}')

# figure : disponibilite au fil de la production
fig, ax = plt.subplots(figsize=(11, 5))
st = {'ema': ('#DD8452', '--', 'récence (statu quo)'), 'ema_seuils': ('#C44E52', ':', 'récence + seuils'),
      'aco': ('#4C72B0', '-.', 'évaporation'), 'aco_seuils': ('#55A868', '-', 'évaporation + seuils')}
for v, (c, ls, lab) in st.items():
    C = [r['curve'] for r in sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed'])]
    xs = [s for s, _ in C[0]]; Y = np.array([[y for _, y in cv] for cv in C])
    ax.plot(xs, Y.mean(0), color=c, linestyle=ls, linewidth=2, label=lab)
for i, e in enumerate(['mort', 'arrivée + mort', 'dérive du monde', 'double mort']):
    x = i * 1020
    ax.axvline(x, color='grey', linewidth=0.8, alpha=0.6)
    ax.text(x + 20, 0.98, e, fontsize=9, color='dimgrey', va='top')
ax.set_ylim(0, 1)
ax.set_xlabel('pas en production'); ax.set_ylabel('R² moyen sur les 17 tâches')
ax.set_title('Disponibilité sans gradient de routage, 20 graines de test (moyenne)')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
plt.tight_layout(); plt.savefig('disponibilite.png', dpi=150, bbox_inches='tight')
