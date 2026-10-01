"""Experiment 7 (preregistered: no symbols). Usage: python analysis_no_symbols.py results/7_no_symbols/no_symbols.jsonl"""
import sys
import numpy as np
from scipy import stats
from common import load, col, boot

rows = load(sys.argv[1] if len(sys.argv) > 1 else 'no_symbols.jsonl')
V = ['mono', 'soft', 'rl', 'aco_thresholds']
g = lambda v, k: col(rows, v, k)
print('seeds', sorted({r['seed'] for r in rows}))
for v in V:
    pa, auc, fin, itf, al = g(v, 'phaseA_r2'), g(v, 'fewshot_auc'), g(v, 'fewshot_final'), g(v, 'interference'), g(v, 'alignment')
    comp = (fin + 12 * (pa - itf)) / 13
    print(f'{v:15s} phaseA {pa.mean():.3f} | speed {auc.mean():.3f} | final new task {fin.mean():.3f} | interference {itf.mean():.3f} '
          f'| global capacity (unclipped) {comp.mean():.3f} | alignment {np.nanmean(al) if not np.all(np.isnan(al)) else float("nan"):.3f}')
print()
def test(name, a, b, k, sign, alpha):
    d = g(a, k) - g(b, k); p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
    good = d.mean() > 0 if sign > 0 else d.mean() < 0
    verdict = 'CONFIRMED' if (p < alpha and good) else ('CONTRADICTED' if p < alpha else 'UNDETERMINED')
    print(f'{name}: {a} - {b} on {k}: mean diff {d.mean():+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] {a} higher on {int((d>0).sum())}/10 p={p:.4f} -> {verdict}')
for n, b in [('N1', 'soft'), ('N2', 'rl'), ('N3', 'mono')]:
    test(n, 'aco_thresholds', b, 'fewshot_auc', +1, 0.05 / 3)
for n, b in [('I1', 'soft'), ('I2', 'mono')]:
    test(n, 'aco_thresholds', b, 'interference', -1, 0.025)
