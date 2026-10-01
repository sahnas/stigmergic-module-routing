"""Experiment 2 (preregistered confirmation, seeds 20-29). Usage: python analysis_confirmation.py results/2_confirmation/confirmation.jsonl"""
import sys
import numpy as np
from scipy import stats
from common import load, col, boot

rows = load(sys.argv[1] if len(sys.argv) > 1 else 'confirmation.jsonl')
g = lambda v, k: col(rows, v, k)
print('seeds', sorted({r['seed'] for r in rows}))
for v in ['soft', 'aco_thresholds', 'aco']:
    tr, ze, kr = g(v, 'train_r2'), g(v, 'zero_r2'), g(v, 'k_train_rec')
    print(f'{v:15s} retention mean {tr.mean():.3f} med {np.median(tr):.3f} min {tr.min():.3f} | '
          f'zero-shot mean {ze.mean():.3f} med {np.median(ze):.3f} min {ze.min():.3f} | k* recovery mean {kr.mean():.3f}')
print()
for name, k in [('C1', 'zero_r2'), ('C2', 'train_r2')]:
    a, b = g('aco_thresholds', k), g('soft', k); d = a - b
    p = stats.wilcoxon(a, b).pvalue
    lo, hi = boot(d)
    ok = p < 0.025 and d.mean() > 0
    print(f'{name} {k}: mean diff {d.mean():+.3f} (95% bootstrap CI {lo:+.3f} to {hi:+.3f}), '
          f'aco_thresholds wins {int((d > 0).sum())}/10, p={p:.4f} -> {"CONFIRMED" if ok else "NOT CONFIRMED"}')
    print('   per seed', np.round(d, 3))
