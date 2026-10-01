"""Experiment 6 (preregistered ablation). Usage: python analysis_ablation.py results/6_ablation/ablation.jsonl"""
import sys
import numpy as np
from scipy import stats
from common import load, col, boot

rows = load(sys.argv[1] if len(sys.argv) > 1 else 'ablation.jsonl')
g = lambda v, k: col(rows, v, k)
def ev(v, e):
    return np.array([r['per_event'][e] for r in sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed'])])
print('seeds', sorted({r['seed'] for r in rows}))
for v in ['aco_thresholds', 'aco_thresholds_random', 'aco_global_threshold']:
    di, ti, tf, al = g(v, 'availability'), g(v, 'train_r2_init'), g(v, 'train_r2_final'), g(v, 'alarms')
    print(f'{v:22s} availability {di.mean():.3f} (med {np.median(di):.3f}) | init {ti.mean():.3f} collapses {int((ti<0.5).sum())}/20 | final {tf.mean():.3f} | alarms {al.mean():.0f}')
    print('                       per event', {e: round(float(ev(v, e).mean()), 3) for e in ['E1', 'E2', 'E3', 'E4']})
for name, b in [('M1 least busy vs random', 'aco_thresholds_random'), ('M2 individual vs collective thresholds', 'aco_global_threshold')]:
    d = g('aco_thresholds', 'availability') - g(b, 'availability'); p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
    verdict = 'CONFIRMED' if (p < 0.025 and d.mean() > 0) else ('CONTRADICTED' if (p < 0.025 and d.mean() < 0) else 'UNDETERMINED')
    print(f'{name}: mean diff {d.mean():+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] positive {int((d>0).sum())}/20 p={p:.4f} -> {verdict}')
    for e in ['E1', 'E2', 'E3', 'E4']:
        de = ev('aco_thresholds', e) - ev(b, e)
        print(f'   {e}: diff {de.mean():+.3f}, positive {int((de>0).sum())}/20')
