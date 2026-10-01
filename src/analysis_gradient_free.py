"""Experiment 3 (gradient-free thesis, seeds 40-59). Usage: python analysis_gradient_free.py results/3_gradient_free/gradient_free.jsonl"""
import os, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load, col, boot

path = sys.argv[1] if len(sys.argv) > 1 else 'gradient_free.jsonl'
rows = load(path)
V = ['ema', 'ema_thresholds', 'aco', 'aco_thresholds']
g = lambda v, k: col(rows, v, k)
def ev(v, e):
    return np.array([r['per_event'][e] for r in sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed'])])

print('seeds', sorted({r['seed'] for r in rows}), '\n')
for v in V:
    di, ti, tf, zf, al = g(v, 'availability'), g(v, 'train_r2_init'), g(v, 'train_r2_final'), g(v, 'zero_r2_final'), g(v, 'alarms')
    print(f'{v:15s} availability mean {di.mean():.3f} med {np.median(di):.3f} | init mean {ti.mean():.3f} collapses at init {int((ti<0.5).sum())}/20 '
          f'| final {tf.mean():.3f} | final zero-shot {zf.mean():.3f} | alarms {al.mean():.0f}')
    print('                per event', {e: round(float(ev(v, e).mean()), 3) for e in ['E1', 'E2', 'E3', 'E4']})
print()
alpha = 0.05 / 3
S1 = g('aco_thresholds', 'availability') - g('ema', 'availability')
S2 = ((g('ema_thresholds', 'availability') - g('ema', 'availability')) + (g('aco_thresholds', 'availability') - g('aco', 'availability'))) / 2
S3 = ((g('aco', 'availability') - g('ema', 'availability')) + (g('aco_thresholds', 'availability') - g('ema_thresholds', 'availability'))) / 2
for name, d, sign in [('S1 aco_thresholds - ema', S1, 1), ('S2 effect of thresholds', S2, 1), ('S3 effect of rule (aco - ema)', S3, 0)]:
    p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
    ok = p < alpha and (sign == 0 or d.mean() > 0)
    print(f'{name:30s} mean diff {d.mean():+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] positive {int((d>0).sum())}/20 p={p:.5f} -> {"CONFIRMED" if ok else "NOT CONFIRMED"}')

print('\nDescriptive, not preregistered: share of the effect due to initial learning')
for v in V:
    ti, di, tf = g(v, 'train_r2_init'), g(v, 'availability'), g(v, 'train_r2_final')
    ok = ti >= 0.8
    if ok.sum():
        print(f'  {v:15s} seeds healthy at init (R2>=0.8): {ok.sum():2d}/20 | on those: init {ti[ok].mean():.3f} availability {di[ok].mean():.3f} '
              f'final {tf[ok].mean():.3f} | loss init->final {np.mean(ti[ok]-tf[ok]):.3f}')
    else:
        print(f'  {v:15s} seeds healthy at init (R2>=0.8):  0/20')
both = (g('ema', 'train_r2_init') >= 0.8) & (g('aco_thresholds', 'train_r2_init') >= 0.8)
d = g('aco_thresholds', 'availability')[both] - g('ema', 'availability')[both]
print(f'  aco_thresholds - ema availability, seeds healthy for both (n={both.sum()}): mean diff {d.mean():+.3f}, positive {int((d>0).sum())}/{both.sum()}')
print(f'  same subset, loss init->final: aco_thresholds loses {np.mean((g("aco_thresholds","train_r2_init")-g("aco_thresholds","train_r2_final"))[both]):.3f}, '
      f'ema loses {np.mean((g("ema","train_r2_init")-g("ema","train_r2_final"))[both]):.3f}')

fig, ax = plt.subplots(figsize=(11, 5))
st = {'ema': ('#DD8452', '--', 'recency (status quo)'), 'ema_thresholds': ('#C44E52', ':', 'recency + thresholds'),
      'aco': ('#4C72B0', '-.', 'evaporation'), 'aco_thresholds': ('#55A868', '-', 'evaporation + thresholds')}
for v, (c, ls, lab) in st.items():
    C = [r['curve'] for r in sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed'])]
    xs = [s for s, _ in C[0]]; Y = np.array([[y for _, y in cv] for cv in C])
    ax.plot(xs, Y.mean(0), color=c, linestyle=ls, linewidth=2, label=lab)
for i, e in enumerate(['death', 'arrival + death', 'world drift', 'double death']):
    x = i * 1020
    ax.axvline(x, color='grey', linewidth=0.8, alpha=0.6)
    ax.text(x + 20, 0.98, e, fontsize=9, color='dimgrey', va='top')
ax.set_ylim(0, 1)
ax.set_xlabel('steps in production'); ax.set_ylabel('mean R² over the 17 tasks')
ax.set_title('Availability without routing gradient, 20 test seeds (mean)')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(path), 'availability.png'), dpi=150, bbox_inches='tight')
