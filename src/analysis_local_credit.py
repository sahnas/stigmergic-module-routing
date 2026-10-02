"""Experiment 10 (preregistered: oracle local credit). Usage: python analysis_local_credit.py results/10_local_credit/local_credit.jsonl"""
import sys
import numpy as np
from scipy import stats
from common import load, col, boot

rows = load(sys.argv[1] if len(sys.argv) > 1 else 'local_credit.jsonl')
g = lambda v, k: col(rows, v, k)
print('seeds', sorted({r['seed'] for r in rows})[0], 'to', sorted({r['seed'] for r in rows})[-1])
for v in ['aco_thresholds', 'local_routing', 'local_full', 'soft']:
    al = g(v, 'alignment')
    print(f'{v:15s} known tasks {g(v,"phaseA_r2").mean():.3f} | speed on new {g(v,"fewshot_auc").mean():.3f} | final new {g(v,"fewshot_final").mean():.3f} '
          f'| interference {g(v,"interference").mean():.3f} | alignment {np.nanmean(al) if not np.all(np.isnan(al)) else float("nan"):.3f}')
alpha = 0.05 / 4
for name, a, k in [('O1', 'local_routing', 'phaseA_r2'), ('O2', 'local_routing', 'alignment'), ('O3', 'local_routing', 'fewshot_auc'), ('O4', 'local_full', 'fewshot_auc')]:
    d = g(a, k) - g('aco_thresholds', k); p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
    verdict = 'CONFIRMED' if (p < alpha and d.mean() > 0) else ('CONTRADICTED' if p < alpha else 'NOT CONFIRMED')
    print(f'{name} {a} - aco_thresholds on {k}: mean diff {d.mean():+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] positive {int((d>0).sum())}/10 p={p:.4f} -> {verdict}')
