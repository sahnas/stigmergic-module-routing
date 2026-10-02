"""Experiment 15 (preregistered: savings of a consumed specialist). Usage: python analysis_savings.py results/15_savings/savings.jsonl"""
import json, sys
import numpy as np
from scipy import stats
from common import boot

rows = sorted([json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else 'savings.jsonl')], key=lambda r: r['seed'])
g = lambda k: np.array([r[k] for r in rows], float)
print('seeds', rows[0]['seed'], 'to', rows[-1]['seed'], '| n', len(rows), '| specialist recruited during production:', sum(r['spec_r_recruited'] for r in rows), '/10 | recruits per run', g('n_recruits').mean())
print(f"specialist R2 on (r,) before the event {g('r2_specialist_before_event').mean():.3f}")
for c in ['consumed_specialist', 'other_specialist', 'fresh']:
    print(f"{c:20s} R2 start {g(c+'_r2_start').mean():+.3f} | AUC {g(c+'_auc').mean():.3f} | steps to 0.8 {g(c+'_steps_to_0.8').mean():.0f} | final {g(c+'_final').mean():.3f}")
alpha = 0.05 / 2
for name, a, b in [('S1', 'consumed_specialist', 'fresh'), ('S2', 'consumed_specialist', 'other_specialist')]:
    d = g(a + '_auc') - g(b + '_auc'); p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
    verdict = ('significant, ' + (a if d.mean() > 0 else b) + ' faster') if p < alpha else 'not significant'
    print(f'{name} {a} - {b} on AUC: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] positive {int((d>0).sum())}/10 p={p:.4f} -> {verdict}')
    ds = g(a + '_steps_to_0.8') - g(b + '_steps_to_0.8')
    print(f'    steps to 0.8, {a} - {b}: {ds.mean():+.0f} (negative = faster)')
