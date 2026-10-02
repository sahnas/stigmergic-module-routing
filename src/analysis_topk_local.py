"""Experiment 11 (preregistered: sparse top-k routers with local losses). Usage: python analysis_topk_local.py results/11_topk_local/topk_local.jsonl"""
import sys
import numpy as np
from scipy import stats
from common import load, col, boot

rows = load(sys.argv[1] if len(sys.argv) > 1 else 'topk_local.jsonl')
g = lambda v, k: col(rows, v, k)
V = ['aco_local_both', 'topk1_local_both', 'topk2_local_both']
print('seeds', sorted({r['seed'] for r in rows})[0], 'to', sorted({r['seed'] for r in rows})[-1], '| n', [len(g(v, 'seed')) for v in V])
for v in V:
    print(f'{v:17s} known tasks {g(v,"phaseA_r2").mean():.3f} | speed on new {g(v,"fewshot_auc").mean():.3f} | final new {g(v,"fewshot_final").mean():.3f} '
          f'| interference {g(v,"interference").mean():.3f} | alignment {g(v,"alignment").mean():.3f}')
alpha = 0.05 / 6
i = 1
for opp in ['topk1_local_both', 'topk2_local_both']:
    for k in ['fewshot_auc', 'interference', 'alignment']:
        d = g('aco_local_both', k) - g(opp, k); p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
        verdict = ('significant, ' + ('traces higher' if d.mean() > 0 else 'top-k higher')) if p < alpha else 'not significant'
        print(f'K{i} aco_local_both - {opp} on {k}: mean diff {d.mean():+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] positive {int((d>0).sum())}/10 p={p:.4f} -> {verdict}')
        i += 1
