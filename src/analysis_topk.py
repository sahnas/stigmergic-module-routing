"""Experiment 8 (preregistered: sparse top-k MoE). Usage: python analysis_topk.py results/8_topk_moe"""
import json, os, sys
import numpy as np
from scipy import stats
from common import boot

d0 = sys.argv[1] if len(sys.argv) > 1 else '.'
rows = []
for f in ['aco_thresholds.jsonl', 'topk1.jsonl', 'topk2.jsonl']:
    for l in open(os.path.join(d0, f)):
        r = json.loads(l)
        if r['variant'] == 'topk':
            r['variant'] = 'topk%d' % r['cfg']['k']
        rows.append(r)
def g(v, k):
    return np.array([r[k] for r in sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed'])], float)
print('seeds', sorted({r['seed'] for r in rows})[0], 'to', sorted({r['seed'] for r in rows})[-1])
for v in ['aco_thresholds', 'topk1', 'topk2']:
    print(f'{v:15s} retention {g(v,"train_r2").mean():.3f} | zero-shot {g(v,"zero_r2").mean():.3f} | k* after damage {g(v,"k_train_dmg").mean():.3f} '
          f'| k* recovered {g(v,"k_train_rec").mean():.3f} | other recovered {g(v,"other_rec").mean():.3f} | secs {g(v,"secs").mean():.1f}')
alpha = 0.05 / 6
print(f'\nTwo-sided paired Wilcoxon, Bonferroni threshold {alpha:.4f} (difference = aco_thresholds - opponent)')
for name, opp in [('T1-T3', 'topk2'), ('T4-T6', 'topk1')]:
    for key in ['zero_r2', 'train_r2', 'k_train_rec']:
        d = g('aco_thresholds', key) - g(opp, key); p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
        sig = 'significant' if p < alpha else 'not significant'
        print(f'  {name} vs {opp:6s} {key:12s} mean diff {d.mean():+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] aco higher on {int((d>0).sum())}/20 p={p:.5f} -> {sig}')
