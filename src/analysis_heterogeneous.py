"""Experiment 9 (preregistered: heterogeneous modules). Usage: python analysis_heterogeneous.py results/9_heterogeneous/heterogeneous.jsonl"""
import sys
import numpy as np
from scipy import stats
from common import load, col, boot

rows = load(sys.argv[1] if len(sys.argv) > 1 else 'heterogeneous.jsonl')
g = lambda v, k: col(rows, v, k)
V = ['aco_thresholds', 'aco_global_threshold', 'swucb:50:0.05', 'ducb:0.98:0.3']
print('seeds', sorted({r['seed'] for r in rows})[0], 'to', sorted({r['seed'] for r in rows})[-1], '| n per variant', [len(g(v, 'availability')) for v in V])
for v in V:
    di, ti, tf = g(v, 'availability'), g(v, 'train_r2_init'), g(v, 'train_r2_final')
    print(f'{v:21s} availability {di.mean():.3f} (med {np.median(di):.3f}) | init {ti.mean():.3f} collapses {int((ti<0.5).sum())}/20 | final {tf.mean():.3f}')
alpha = 0.05 / 3
for name, b, directional in [('HE1', 'aco_global_threshold', True), ('HE2', 'swucb:50:0.05', False), ('HE3', 'ducb:0.98:0.3', False)]:
    d = g('aco_thresholds', 'availability') - g(b, 'availability'); p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
    if directional:
        verdict = 'CONFIRMED' if (p < alpha and d.mean() > 0) else ('CONTRADICTED' if p < alpha else 'NOT CONFIRMED')
    else:
        verdict = ('significant, ' + ('mechanism ahead' if d.mean() > 0 else 'opponent ahead')) if p < alpha else 'not significant'
    print(f'{name} aco_thresholds - {b}: mean diff {d.mean():+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] positive {int((d>0).sum())}/20 p={p:.5f} -> {verdict}')
