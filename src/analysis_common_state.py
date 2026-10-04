"""Experiments 14b and 16b (preregistered). Usage: python analysis_common_state.py results/14b_16b_common_state/common_state.jsonl"""
import json, sys
import numpy as np
from scipy import stats
from common import boot

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else 'common_state.jsonl')]
def sel(cap, rule, itv='keep'):
    return sorted([r for r in rows if r['capacity'] == cap and r['rule'] == rule and r['intervention'] == itv], key=lambda r: r['seed'])
def arr(g, k): return np.array([r[k] for r in g], float)
print('seeds', sorted({r['seed'] for r in rows})[0], 'to', sorted({r['seed'] for r in rows})[-1], '| rows', len(rows))
print(f"{'capacity':8s} {'rule / intervention':22s} avail  rare_bef rare_aft other_aft killed_auc killed_aft post_recr first=rare_spec first=dormant")
for cap in ['none', 'spare', 'dormant']:
    for rule in ['least_committed', 'random', 'none']:
        g = sel(cap, rule)
        print(f"{cap:8s} {rule:22s} {arr(g,'availability').mean():.3f}  {arr(g,'rare_before').mean():+.2f}    {arr(g,'rare_after').mean():+.2f}    {arr(g,'other_after').mean():+.2f}     {arr(g,'killed_auc').mean():.2f}       {arr(g,'killed_after').mean():.2f}      {arr(g,'post_recruits').mean():5.1f}    "
              f"{sum(bool(r['first_is_rare_specialist']) for r in g)}/10           {sum(bool(r['first_is_dormant']) for r in g)}/10")
for itv in ['reset_opt', 'reset_both']:
    g = sel('none', 'least_committed', itv)
    print(f"{'none':8s} {'16b ' + itv:22s} {arr(g,'availability').mean():.3f}  {arr(g,'rare_before').mean():+.2f}    {arr(g,'rare_after').mean():+.2f}    {arr(g,'other_after').mean():+.2f}     {arr(g,'killed_auc').mean():.2f}       {arr(g,'killed_after').mean():.2f}      {arr(g,'post_recruits').mean():5.1f}")
alpha = 0.05 / 6
print(f'\nTwo-sided paired Wilcoxon, threshold {alpha:.4f}')
def test(name, a, b, k):
    d = arr(a, k) - arr(b, k); p = stats.wilcoxon(d).pvalue; lo, hi = boot(d)
    print(f'{name} on {k}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] positive {int((d>0).sum())}/10 p={p:.4f} -> {"significant" if p < alpha else "not significant"}')
for k in ['availability', 'rare_after']:
    test('U1 least_committed - none (capacity none)', sel('none', 'least_committed'), sel('none', 'none'), k)
for k in ['availability', 'rare_after']:
    test('U2 dormant - spare (least_committed)', sel('dormant', 'least_committed'), sel('spare', 'least_committed'), k)
for itv in ['reset_opt', 'reset_both']:
    test(f'W1 {itv} - keep (capacity none)', sel('none', 'least_committed', itv), sel('none', 'least_committed'), 'availability')
