"""Experiment 4 (preregistered test against standard non-stationary bandits, B1-B3).
Usage: python analysis_bandits.py results/4_standard_bandits"""
import json, os, sys
import numpy as np
from scipy import stats
from common import load, col, boot

d0 = sys.argv[1] if len(sys.argv) > 1 else '.'
cfg = json.load(open(os.path.join(d0, 'selected_configs.json')))
for title, f in [('MAIN TEST, seeds 80 to 99', 'main_seeds_80_99.jsonl'), ('REPLICATION, seeds 60 to 74 (interrupted run)', 'replication_seeds_60_74.jsonl')]:
    rows = load(os.path.join(d0, f))
    g = lambda v, k='availability': col(rows, v, k)
    s = sorted({r['seed'] for r in rows})
    print(title, '| seeds', s[0], 'to', s[-1], 'n =', len(g('aco_thresholds')))
    for v in ['aco_thresholds'] + list(cfg.values()):
        di, ti, tf = g(v), g(v, 'train_r2_init'), g(v, 'train_r2_final')
        print(f'  {v:18s} availability {di.mean():.3f} (med {np.median(di):.3f}) | init {ti.mean():.3f} | final {tf.mean():.3f}')
    for name, (k, v) in zip(['B1', 'B2', 'B3'], cfg.items()):
        d = g('aco_thresholds') - g(v); p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
        verdict = 'CONFIRMED' if (p < 0.05 / 3 and d.mean() > 0) else ('CONTRADICTED' if p < 0.05 / 3 else 'NOT CONFIRMED')
        print(f'  {name} aco_thresholds - {v}: mean diff {d.mean():+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] positive {int((d>0).sum())}/{len(d)} p={p:.5f} -> {verdict}')
    print()
