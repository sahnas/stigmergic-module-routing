"""Experiment 5 (preregistered test against a tuned change-point bandit).
Usage: python analysis_changepoint.py results/5_changepoint"""
import collections, json, os, sys
import numpy as np
from scipy import stats
from common import load, col, boot

d0 = sys.argv[1] if len(sys.argv) > 1 else '.'
print('Tuning of the opponent on development seeds 300-302 (mean availability, per-seed values, config):')
d = collections.defaultdict(list)
for r in load(os.path.join(d0, 'tuning_seeds_300_302.jsonl')):
    d[json.dumps(r['cfg'], sort_keys=True)].append(r['availability'])
for k, v in sorted(d.items(), key=lambda kv: -np.mean(kv[1])):
    print(' ', round(np.mean(v), 3), np.round(v, 3), k)
print()
rows = load(os.path.join(d0, 'changepoint_test.jsonl'))
g = lambda v, k: col(rows, v, k)
def ev(v, e):
    return np.array([r['per_event'][e] for r in sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed'])])
print('seeds', sorted({r['seed'] for r in rows}), 'n per variant', len(g('ema', 'seed')), '\n')
for v in ['ema', 'changepoint', 'aco_thresholds']:
    di, ti, tf, zf = g(v, 'availability'), g(v, 'train_r2_init'), g(v, 'train_r2_final'), g(v, 'zero_r2_final')
    print(f'{v:15s} availability mean {di.mean():.3f} med {np.median(di):.3f} min {di.min():.3f} | init {ti.mean():.3f} collapses {int((ti<0.5).sum())}/20 '
          f'| final {tf.mean():.3f} | final zero-shot {zf.mean():.3f}')
    print('                per event', {e: round(float(ev(v, e).mean()), 3) for e in ['E1', 'E2', 'E3', 'E4']})
print()
for name, a, b in [('R1', 'aco_thresholds', 'changepoint'), ('R2', 'changepoint', 'ema')]:
    dd = g(a, 'availability') - g(b, 'availability'); p = stats.wilcoxon(dd).pvalue; lo, hi = boot(dd)
    if name == 'R1':
        verdict = 'MAINTAINED' if (p < 0.025 and dd.mean() > 0) else ('INVALIDATED' if (p < 0.025 and dd.mean() < 0) else 'UNDETERMINED')
    else:
        verdict = 'CONFIRMED' if (p < 0.025 and dd.mean() > 0) else 'NOT CONFIRMED'
    print(f'{name} {a} - {b}: mean diff {dd.mean():+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] positive {int((dd>0).sum())}/20 p={p:.5f} -> {verdict}')
both = (g('changepoint', 'train_r2_init') >= 0.8) & (g('aco_thresholds', 'train_r2_init') >= 0.8)
dd = g('aco_thresholds', 'availability')[both] - g('changepoint', 'availability')[both]
print(f'\nDescriptive: seeds healthy at start for both (n={both.sum()}) availability diff {dd.mean():+.3f}, positive {int((dd>0).sum())}/{both.sum()}')
for v in ['changepoint', 'aco_thresholds']:
    print(f'  loss init->final {v}: {np.mean((g(v,"train_r2_init")-g(v,"train_r2_final"))[both]):.3f}')
