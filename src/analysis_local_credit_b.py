"""Experiment 10b (preregistered replication of experiment 10 with a gradient router given local losses).
Usage: python analysis_local_credit_b.py results/10_local_credit/10b_seeds_810_819.jsonl"""
import sys
import numpy as np
from scipy import stats
from common import load, col, boot

rows = load(sys.argv[1] if len(sys.argv) > 1 else '10b_seeds_810_819.jsonl')
g = lambda v, k: col(rows, v, k)
V = ['aco_thresholds', 'aco_local_both', 'soft_local_both']
print('seeds', sorted({r['seed'] for r in rows})[0], 'to', sorted({r['seed'] for r in rows})[-1], '| n', [len(g(v, 'seed')) for v in V])
for v in V:
    print(f'{v:16s} known tasks {g(v,"phaseA_r2").mean():.3f} | speed on new {g(v,"fewshot_auc").mean():.3f} | final new {g(v,"fewshot_final").mean():.3f} '
          f'| interference {g(v,"interference").mean():.3f} | alignment {g(v,"alignment").mean():.3f}')
alpha = 0.05 / 4
def test(name, a, b, k, sign):
    d = g(a, k) - g(b, k); p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
    if sign == 0:
        verdict = ('significant, ' + (a if d.mean() > 0 else b) + ' higher') if p < alpha else 'not significant'
    else:
        good = d.mean() > 0 if sign > 0 else d.mean() < 0
        verdict = 'CONFIRMED' if (p < alpha and good) else ('CONTRADICTED' if p < alpha else 'NOT CONFIRMED')
    print(f'{name} {a} - {b} on {k}: mean diff {d.mean():+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] positive {int((d>0).sum())}/{len(d)} p={p:.4f} -> {verdict}')
test('P1', 'aco_local_both', 'aco_thresholds', 'fewshot_auc', +1)
test('P2', 'aco_local_both', 'soft_local_both', 'fewshot_auc', 0)
test('P3', 'aco_local_both', 'soft_local_both', 'interference', -1)
test('P4', 'aco_local_both', 'soft_local_both', 'alignment', +1)
