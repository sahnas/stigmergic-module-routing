"""Experiment 14 (preregistered: preservation of a rare specialist). Usage: python analysis_preservation.py results/14_preservation/preservation.jsonl"""
import json, sys
import numpy as np
from scipy import stats
from common import boot

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else 'preservation.jsonl')]
def sel(sc, rs, ru):
    return sorted([r for r in rows if r['scenario'] == sc and r['reserves'] == rs and r['rule'] == ru], key=lambda r: r['seed'])
def arr(rs_, k): return np.array([r[k] for r in rs_], float)
print('seeds', sorted({r['seed'] for r in rows})[0], 'to', sorted({r['seed'] for r in rows})[-1])
print(f"{'scenario':18s} {'res':3s} {'rule':16s} rare_before rare_after other_after avail  spec_recruited first_dormant recruits")
for sc in ['learned_then_rare', 'rare_from_start']:
    for rs in [0, 2]:
        for ru in ['least_committed', 'random', 'none']:
            g = sel(sc, rs, ru)
            print(f"{sc:18s} {rs:3d} {ru:16s} {arr(g,'rare_r2_before').mean():+.2f}       {arr(g,'rare_r2_after').mean():+.2f}       {arr(g,'other_r2_after').mean():+.2f}        {arr(g,'availability').mean():.3f}  "
                  f"{sum(r['spec_r_recruited'] for r in g)}/10            {sum(bool(r['first_is_dormant']) for r in g)}/10           {arr(g,'n_recruits').mean():.1f}")
print()
g = sel('learned_then_rare', 0, 'least_committed')
print('Q1 rare R2 after < 0.5:', sum(r['rare_r2_after'] < 0.5 for r in g), '/10 ; specialist among recruits:', sum(r['spec_r_recruited'] for r in g), '/10')
n = sel('learned_then_rare', 0, 'none')
print('Q2 rule none, rare R2 after < 0.5:', sum(r['rare_r2_after'] < 0.5 for r in n), '/10')
for k in ['rare_r2_after', 'availability']:
    d = arr(g, k) - arr(n, k); p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
    print(f'   least_committed - none on {k}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] positive {int((d>0).sum())}/10 p={p:.4f} -> {"significant" if p < 0.025 else "not significant"}')
g2 = sel('learned_then_rare', 2, 'least_committed')
print('Q3 two reserves: first recruit dormant', sum(bool(r['first_is_dormant']) for r in g2), '/10 ; rare R2 after > 0.8:', sum(r['rare_r2_after'] > 0.8 for r in g2), '/10')
g4 = sel('rare_from_start', 0, 'least_committed')
print('Q4 rare from start: rare R2 before < 0.5:', sum(r['rare_r2_before'] < 0.5 for r in g4), '/10')
rnd = sel('learned_then_rare', 0, 'random')
print('control random: specialist among recruits', sum(r['spec_r_recruited'] for r in rnd), '/10')
