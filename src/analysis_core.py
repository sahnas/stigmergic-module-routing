"""Experiment 1 (core test with symbols). Usage: python analysis_core.py results/1_core/results.jsonl"""
import os, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load

path = sys.argv[1] if len(sys.argv) > 1 else 'results.jsonl'
rows = load(path)
V = ['mono', 'oracle', 'soft', 'cumulative', 'ema', 'aco', 'global', 'aco_thresholds']
by = {v: sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed']) for v in V}
seeds = sorted({r['seed'] for r in rows})


def arr(v, key):
    return np.array([np.nan if r.get(key) is None else r[key] for r in by[v]], dtype=float)


def ci(x):
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan, len(x)
    m = x.mean(); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, h, len(x)


cols = [('train_r2', 'retention at end of phase 1'), ('zero_r2', 'zero-shot compositions'),
        ('k_train_dmg', 'k* right after damage'), ('k_train_rec', 'k* after recovery'),
        ('k_held_rec', 'zero-shot k* after recovery'), ('other_rec', 'other tasks after recovery'),
        ('reroute_latency', 'rerouting latency (steps)'), ('secs', 'time (s)')]
print(f'seeds: {seeds}\n')
for key, label in cols:
    print(label)
    for v in V:
        m, h, n = ci(arr(v, key))
        med = np.nanmedian(arr(v, key)) if n else np.nan
        print(f'  {v:15s} mean {m:8.3f} +/- {h:6.3f}   median {med:8.3f}   n={n}')
    print()
for v in ['cumulative', 'ema', 'aco', 'global', 'aco_thresholds']:
    nd = arr(v, 'n_distinct_units'); al = arr(v, 'alarms_total'); lat = arr(v, 'reroute_latency')
    print(f'{v:15s} distinct units for 5 primitives: {np.nanmean(nd):.1f}   alarms (total): {np.nanmean(al):.1f}'
          f'   successful rerouting: {int(np.sum(~np.isnan(lat)))}/10')
print()
tests = [
    ('H1', 'aco', 'cumulative', 'k_train_rec'), ('H1', 'aco', 'oracle', 'k_train_rec'),
    ('H2', 'aco_thresholds', 'aco', 'k_train_rec'), ('H2', 'aco_thresholds', 'aco', 'reroute_latency'),
    ('H3', 'aco', 'global', 'train_r2'),
    ('H4', 'aco', 'soft', 'zero_r2'), ('H4', 'aco', 'mono', 'zero_r2'),
    ('ctrl', 'aco', 'ema', 'k_train_rec'), ('ctrl', 'aco', 'ema', 'train_r2'),
    ('cost', 'aco', 'cumulative', 'train_r2'),
]
alpha = 0.05 / len(tests)
print(f'Paired Wilcoxon tests (two-sided), Bonferroni threshold {alpha:.4f}')
for h, a, b, key in tests:
    xa, xb = arr(a, key), arr(b, key)
    ok = ~np.isnan(xa) & ~np.isnan(xb)
    d = xa[ok] - xb[ok]
    if ok.sum() < 5 or np.all(d == 0):
        print(f'  {h:4s} {a} - {b} on {key}: n={ok.sum()} insufficient'); continue
    p = stats.wilcoxon(xa[ok], xb[ok]).pvalue
    wins = int(np.sum(d > 0))
    print(f'  {h:4s} {a:14s} - {b:10s} {key:16s} mean diff {d.mean():+.3f} (median {np.median(d):+.3f}) '
          f'{a} > {b} on {wins}/{ok.sum()} seeds  p={p:.4f} {"*" if p < alpha else ""}')

plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(figsize=(11, 5))
styles = {'oracle': ('#937860', ':'), 'soft': ('#8172B3', '-.'), 'cumulative': ('#C44E52', '--'),
          'ema': ('#DD8452', '--'), 'aco': ('#4C72B0', '-'), 'aco_thresholds': ('#55A868', '-')}
labels = {'oracle': 'fixed orchestration', 'soft': 'router learned by gradient (soft)', 'cumulative': 'traces without forgetting',
          'ema': 'forgetting by recency', 'aco': 'evaporation (ants)', 'aco_thresholds': 'evaporation + thresholds'}
for v, (c, ls) in styles.items():
    curves = [r['curve'] for r in by[v]]
    steps = [s for s, _ in curves[0]]
    Y = np.array([[y for _, y in cv] for cv in curves])
    m = Y.mean(0); lo, hi = np.percentile(Y, 25, 0), np.percentile(Y, 75, 0)
    ax.plot(steps, m, color=c, linestyle=ls, linewidth=2, label=labels[v])
    ax.fill_between(steps, lo, hi, color=c, alpha=0.12)
ax.set_xlabel('learning steps after the death of the unit')
ax.set_ylabel('mean R² on tasks involving k*')
ax.set_ylim(-0.3, 1.02)
ax.set_title('Compensation after permanent loss of a unit\n10 test seeds, mean and quartiles')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
plt.tight_layout()
out = os.path.join(os.path.dirname(path), 'recovery.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nfigure: {out}')
