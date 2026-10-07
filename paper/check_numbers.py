"""Recomputes every number reported in the manuscript from the released raw results.
Run from the repository root: python paper/check_numbers.py"""
import json, numpy as np
from scipy import stats
R = 'results/'
def L(p): return [json.loads(l) for l in open(R + p)]
def g(rows, v, k): return np.array([r[k] if r.get(k) is not None else np.nan for r in sorted([r for r in rows if r['variant'] == v], key=lambda r: r['seed'])], float)
chk = []
def C(label, claimed, actual, tol=0.006):
    ok = abs(claimed - actual) <= tol; chk.append(ok)
    print(('OK   ' if ok else 'DIFF '), f'{label:50s} paper {claimed:+.3f}  data {actual:+.3f}')
r = L('1_core/results.jsonl')
for v, (a, b, c) in {'mono': (-0.47, -0.50, -0.05), 'oracle': (0.94, 0.91, 0.00), 'soft': (0.81, 0.73, 0.98), 'cumulative': (0.75, 0.70, 0.00), 'ema': (0.76, 0.68, 0.75), 'aco': (0.83, 0.72, 0.92), 'global': (0.07, -0.09, 0.13), 'aco_thresholds': (0.91, 0.88, 0.97)}.items():
    C(f'E1 {v} retention', a, g(r, v, 'train_r2').mean()); C(f'E1 {v} zero-shot', b, g(r, v, 'zero_r2').mean()); C(f'E1 {v} k* recovered', c, g(r, v, 'k_train_rec').mean())
for v, m in [('soft', 324), ('ema', 524), ('aco', 555), ('global', 37), ('aco_thresholds', 4.5)]:
    C(f'E1 {v} latency median', m, np.nanmedian(g(r, v, 'reroute_latency')), 0.6)
C('E1 aco - cumulative, k* recovered', 0.92, (g(r, 'aco', 'k_train_rec') - g(r, 'cumulative', 'k_train_rec')).mean())
C('E1 aco - global, retention', 0.76, (g(r, 'aco', 'train_r2') - g(r, 'global', 'train_r2')).mean())
r = L('2_confirmation/confirmation.jsonl')
C('E2 zero-shot diff', 0.008, (g(r, 'aco_thresholds', 'zero_r2') - g(r, 'soft', 'zero_r2')).mean(), 0.0006)
C('E2 retention diff', 0.014, (g(r, 'aco_thresholds', 'train_r2') - g(r, 'soft', 'train_r2')).mean(), 0.0006)
C('E2 soft retention', 0.88, g(r, 'soft', 'train_r2').mean())
r = L('3_gradient_free/gradient_free.jsonl')
for v, (a, n, f, z) in {'ema': (0.41, 10, 0.22, 0.07), 'ema_thresholds': (0.51, 7, 0.61, 0.45), 'aco': (0.57, 4, 0.37, 0.20), 'aco_thresholds': (0.76, 0, 0.60, 0.39)}.items():
    C(f'E3 {v} availability', a, g(r, v, 'availability').mean()); C(f'E3 {v} collapses', n, (g(r, v, 'train_r2_init') < 0.5).sum(), 0.5); C(f'E3 {v} final', f, g(r, v, 'train_r2_final').mean()); C(f'E3 {v} zero-shot', z, g(r, v, 'zero_r2_final').mean())
S1 = g(r, 'aco_thresholds', 'availability') - g(r, 'ema', 'availability')
S2 = ((g(r, 'ema_thresholds', 'availability') - g(r, 'ema', 'availability')) + (g(r, 'aco_thresholds', 'availability') - g(r, 'aco', 'availability'))) / 2
S3 = ((g(r, 'aco', 'availability') - g(r, 'ema', 'availability')) + (g(r, 'aco_thresholds', 'availability') - g(r, 'ema_thresholds', 'availability'))) / 2
C('E3 S1', 0.35, S1.mean()); C('E3 S2', 0.14, S2.mean()); C('E3 S3', 0.21, S3.mean())
both = (g(r, 'ema', 'train_r2_init') >= 0.8) & (g(r, 'aco_thresholds', 'train_r2_init') >= 0.8)
C('E3 loss aco_thresholds (healthy)', 0.30, np.mean((g(r, 'aco_thresholds', 'train_r2_init') - g(r, 'aco_thresholds', 'train_r2_final'))[both]))
C('E3 loss ema (healthy)', 0.77, np.mean((g(r, 'ema', 'train_r2_init') - g(r, 'ema', 'train_r2_final'))[both]))
r = L('4_standard_bandits/main_seeds_80_99.jsonl')
for v, (a, i, f) in {'aco_thresholds': (0.74, 0.94, 0.65), 'swucb:50:0.05': (0.72, 0.69, 0.86), 'ducb:0.98:0.3': (0.72, 0.66, 0.88), 'mucb:60:0.2:0.3': (0.59, 0.90, 0.72)}.items():
    C(f'E4 {v} availability', a, g(r, v, 'availability').mean()); C(f'E4 {v} init', i, g(r, v, 'train_r2_init').mean()); C(f'E4 {v} final', f, g(r, v, 'train_r2_final').mean())
for v, c in [('swucb:50:0.05', 0.02), ('ducb:0.98:0.3', 0.03), ('mucb:60:0.2:0.3', 0.15)]:
    C(f'E4 diff vs {v}', c, (g(r, 'aco_thresholds', 'availability') - g(r, v, 'availability')).mean())
r = L('5_changepoint/changepoint_test.jsonl')
C('E5 aco_thresholds availability', 0.77, g(r, 'aco_thresholds', 'availability').mean()); C('E5 changepoint availability', 0.61, g(r, 'changepoint', 'availability').mean())
C('E5 R1', 0.16, (g(r, 'aco_thresholds', 'availability') - g(r, 'changepoint', 'availability')).mean()); C('E5 R2', 0.16, (g(r, 'changepoint', 'availability') - g(r, 'ema', 'availability')).mean())
C('E5 changepoint init', 0.95, g(r, 'changepoint', 'train_r2_init').mean()); C('E5 aco_thresholds init', 0.91, g(r, 'aco_thresholds', 'train_r2_init').mean())
both = (g(r, 'changepoint', 'train_r2_init') >= 0.8) & (g(r, 'aco_thresholds', 'train_r2_init') >= 0.8)
C('E5 loss changepoint', 0.37, np.mean((g(r, 'changepoint', 'train_r2_init') - g(r, 'changepoint', 'train_r2_final'))[both])); C('E5 loss aco_thresholds', 0.21, np.mean((g(r, 'aco_thresholds', 'train_r2_init') - g(r, 'aco_thresholds', 'train_r2_final'))[both]))
r = L('6_ablation/ablation.jsonl')
C('E6 aco_thresholds', 0.74, g(r, 'aco_thresholds', 'availability').mean()); C('E6 random recruitment', 0.52, g(r, 'aco_thresholds_random', 'availability').mean())
C('E6 M1', 0.22, (g(r, 'aco_thresholds', 'availability') - g(r, 'aco_thresholds_random', 'availability')).mean()); C('E6 M2', 0.002, (g(r, 'aco_thresholds', 'availability') - g(r, 'aco_global_threshold', 'availability')).mean(), 0.0006)
r = L('7_no_symbols/no_symbols.jsonl')
for v, (k, s, f, i) in {'mono': (0.95, 0.96, 0.98, 0.60), 'soft': (0.91, 0.89, 0.97, 2.23), 'rl': (0.13, 0.29, 0.50, 0.21), 'aco_thresholds': (0.21, 0.29, 0.55, 0.21)}.items():
    C(f'E7 {v} known', k, g(r, v, 'phaseA_r2').mean()); C(f'E7 {v} speed', s, g(r, v, 'fewshot_auc').mean()); C(f'E7 {v} final', f, g(r, v, 'fewshot_final').mean()); C(f'E7 {v} interference', i, g(r, v, 'interference').mean())
for v, a in [('soft', -8.8), ('rl', -0.24), ('aco_thresholds', -0.20)]:
    C(f'E7 {v} alignment', a, g(r, v, 'alignment').mean(), 0.06)

import os
def LT(f):
    rows=[]
    for l in open(R + '8_topk_moe/' + f):
        r=json.loads(l)
        if r['variant']=='topk': r['variant']='topk%d' % r['cfg']['k']
        rows.append(r)
    return rows
r = LT('aco_thresholds.jsonl') + LT('topk1.jsonl') + LT('topk2.jsonl')
for v, (a, b, c) in {'aco_thresholds': (0.89, 0.87, 0.97), 'topk1': (0.96, 0.95, 0.05), 'topk2': (0.93, 0.89, 0.96)}.items():
    C(f'E8 {v} retention', a, g(r, v, 'train_r2').mean()); C(f'E8 {v} zero-shot', b, g(r, v, 'zero_r2').mean()); C(f'E8 {v} k* recovered', c, g(r, v, 'k_train_rec').mean())
for opp, vals in [('topk2', (-0.03, -0.04, 0.011)), ('topk1', (-0.08, -0.07, 0.93))]:
    C(f'E8 diff zero-shot vs {opp}', vals[0], (g(r, 'aco_thresholds', 'zero_r2') - g(r, opp, 'zero_r2')).mean())
    C(f'E8 diff retention vs {opp}', vals[1], (g(r, 'aco_thresholds', 'train_r2') - g(r, opp, 'train_r2')).mean())
    C(f'E8 diff recovery vs {opp}', vals[2], (g(r, 'aco_thresholds', 'k_train_rec') - g(r, opp, 'k_train_rec')).mean(), 0.0006 if opp == 'topk2' else 0.006)
r = L('9_heterogeneous/heterogeneous.jsonl')
for v, (a, i, n, f) in {'aco_thresholds': (0.72, 0.89, 0, 0.55), 'aco_global_threshold': (0.72, 0.89, 1, 0.53), 'swucb:50:0.05': (0.71, 0.75, 2, 0.76), 'ducb:0.98:0.3': (0.68, 0.63, 7, 0.74)}.items():
    C(f'E9 {v} availability', a, g(r, v, 'availability').mean()); C(f'E9 {v} init', i, g(r, v, 'train_r2_init').mean())
    C(f'E9 {v} collapses', n, (g(r, v, 'train_r2_init') < 0.5).sum(), 0.5); C(f'E9 {v} final', f, g(r, v, 'train_r2_final').mean())
for opp, val in [('aco_global_threshold', 0.002), ('swucb:50:0.05', 0.01), ('ducb:0.98:0.3', 0.04)]:
    C(f'E9 diff vs {opp}', val, (g(r, 'aco_thresholds', 'availability') - g(r, opp, 'availability')).mean(), 0.0006 if opp == 'aco_global_threshold' else 0.006)

def LC(f): return [json.loads(l) for l in open(R + '10_local_credit/' + f)]
r = LC('local_credit.jsonl')
for v, (k, sp, i, a) in {'aco_thresholds': (0.18, 0.31, 0.21, -0.24), 'local_routing': (0.09, 0.22, 0.19, -0.27), 'local_full': (0.97, 0.94, 0.20, 0.99)}.items():
    C(f'E10 {v} known', k, g(r, v, 'phaseA_r2').mean()); C(f'E10 {v} speed', sp, g(r, v, 'fewshot_auc').mean()); C(f'E10 {v} interference', i, g(r, v, 'interference').mean()); C(f'E10 {v} alignment', a, g(r, v, 'alignment').mean())
C('E10 level1 known diff', -0.09, (g(r, 'local_routing', 'phaseA_r2') - g(r, 'aco_thresholds', 'phaseA_r2')).mean())
C('E10 level1 speed diff', -0.09, (g(r, 'local_routing', 'fewshot_auc') - g(r, 'aco_thresholds', 'fewshot_auc')).mean())
C('E10 level2 speed diff', 0.63, (g(r, 'local_full', 'fewshot_auc') - g(r, 'aco_thresholds', 'fewshot_auc')).mean())
r = LC('10b_seeds_810_819.jsonl')
for v, (k, sp, i, a) in {'aco_thresholds': (0.20, 0.30, 0.22, -0.28), 'aco_local_both': (0.97, 0.95, 0.16, 0.99), 'soft_local_both': (0.99, 0.95, 0.66, -1.65)}.items():
    C(f'E10b {v} known', k, g(r, v, 'phaseA_r2').mean()); C(f'E10b {v} speed', sp, g(r, v, 'fewshot_auc').mean()); C(f'E10b {v} interference', i, g(r, v, 'interference').mean()); C(f'E10b {v} alignment', a, g(r, v, 'alignment').mean())
C('E10b P1', 0.65, (g(r, 'aco_local_both', 'fewshot_auc') - g(r, 'aco_thresholds', 'fewshot_auc')).mean())
C('E10b P2', -0.008, (g(r, 'aco_local_both', 'fewshot_auc') - g(r, 'soft_local_both', 'fewshot_auc')).mean(), 0.0006)
r = L('11_topk_local/topk_local.jsonl')
for v, (k, sp, i, a) in {'aco_local_both': (0.97, 0.95, 0.19, 0.99), 'topk1_local_both': (0.97, 0.66, 0.28, 0.46), 'topk2_local_both': (0.99, 0.87, 1.21, 0.94)}.items():
    C(f'E11 {v} known', k, g(r, v, 'phaseA_r2').mean()); C(f'E11 {v} speed', sp, g(r, v, 'fewshot_auc').mean()); C(f'E11 {v} interference', i, g(r, v, 'interference').mean()); C(f'E11 {v} alignment', a, g(r, v, 'alignment').mean())
C('E11 K1 speed vs top1', -0.29, (g(r, 'topk1_local_both', 'fewshot_auc') - g(r, 'aco_local_both', 'fewshot_auc')).mean())
C('E11 K3 alignment vs top1', -0.53, (g(r, 'topk1_local_both', 'alignment') - g(r, 'aco_local_both', 'alignment')).mean())
C('E11 K4 speed vs top2', -0.08, (g(r, 'topk2_local_both', 'fewshot_auc') - g(r, 'aco_local_both', 'fewshot_auc')).mean())
C('E11 K6 alignment vs top2', -0.05, (g(r, 'topk2_local_both', 'alignment') - g(r, 'aco_local_both', 'alignment')).mean())

# --- experiment 17 pilot and the planning gap ---
rows = [json.loads(l) for l in open(R + 'lewm_pilot/pilot_seeds_0_4.jsonl')]
def mk(sysn, ph, k): return float(np.mean([r['systems'][sysn]['marks'][ph][k] for r in rows]))
for sysn, vals in {'single': (0.070, 1.60, 0.063, 0.058, 1.68), 'single_replay': (0.064, 0.131, 0.080, 0.072, 0.107), 'bank_persistence': (0.108, 0.132, 0.116, 0.111, 0.138), 'bank_traces': (0.108, 0.132, 0.116, 0.111, 0.138)}.items():
    for (ph, k), v in zip([('P1','accessA'),('P2','accessA'),('P2','accessB'),('P3','accessA'),('P3','accessB')], vals):
        C(f'E17 {sysn} {ph} {k}', v, mk(sysn, ph, k), 0.0006 if v < 0.2 else 0.006)
cf = json.load(open(R + 'lewm_kaggle/counterfactual.json'))
C('gap counterfactual released', 0.60, cf['counterfactual_random_actions']['released/dataset'], 0.006); C('gap counterfactual retrained', 0.55, cf['counterfactual_random_actions']['retrained/dataset'], 0.006)
C('gap in-distribution retrained', 0.006, cf['in_distribution']['retrained/dataset'], 0.0006); C('gap in-distribution released', 0.067, cf['in_distribution']['released/dataset'], 0.0006)
rk = json.load(open(R + 'lewm_kaggle/ranking.json')); C('gap ranking retrained', 0.98, rk['retrained']['spearman']['mean'], 0.006); C('gap ranking released', 0.90, rk['released']['spearman']['mean'], 0.006)
ex = json.load(open(R + 'lewm_kaggle/exploit.json'))
for mk_, (p, t) in {'released': (12.4, 43.8), 'retrained': (0.68, 37.0)}.items():
    C(f'gap exploit {mk_} predicted', p, ex[mk_]['predicted_cost_of_chosen_plan'], 0.06 if p > 1 else 0.006); C(f'gap exploit {mk_} true', t, ex[mk_]['true_cost_of_chosen_plan'], 0.06)
rm = json.load(open(R + 'lewm_kaggle/rooms.json'))
C('gap rooms released same', 0.90, rm['released']['same_room'], 0.006); C('gap rooms retrained same', 0.40, rm['retrained']['same_room'], 0.006); C('gap rooms released cross', 0.67, rm['released']['cross_room'], 0.006); C('gap rooms retrained cross', 0.19, rm['retrained']['cross_room'], 0.006)

vs = json.load(open(R + 'lewm_kaggle/viscore.json'))
for mk_, (v, so, pr) in {'released': (0.862, 0.703, 0.606), 'retrained': (0.998, 0.984, 0.982)}.items():
    C(f'viscore {mk_} veracity', v, vs[mk_]['veracity'], 0.0006); C(f'viscore {mk_} sobriety', so, vs[mk_]['sobriety'], 0.0006); C(f'viscore {mk_} product', pr, vs[mk_]['VIScore'], 0.0006)

zz = json.load(open(R + 'lewm_kaggle/zigzag.json'))['summary']
C('zigzag retrained err zig', 1.8, zz['retrained']['err_zig'], 0.06); C('zigzag released err zig', 4.4, zz['released']['err_zig'], 0.06)

# --- experiment 18 ---
import re, glob
def succ(tag):
    t = open(R + f'lewm_kaggle/exp18_{tag}_results.txt').read(); m = re.search(r"'episode_successes': array\(\[(.*?)\]\)", t, re.S)
    return sum(x.strip() == 'True' for x in m.group(1).replace('\n', ' ').split(','))
for tag, v in {'bilinear_seed43': 99, 'bilinear_seed44': 100, 'bilinear_seed45': 96, 'released_seed43': 83, 'released_seed44': 87, 'released_seed45': 80, 'linear_seed42': 57, 'markov_seed42': 96, 'ridge_x10_seed42': 96, 'ridge_d10_seed42': 97}.items():
    C(f'E18 {tag}', v, succ(tag), 0.5)
t = open(R + 'lewm_kaggle/koopman_results.txt').read(); m = re.search(r"'episode_successes': array\(\[(.*?)\]\)", t, re.S); C('E18 bilinear seed42', 97, sum(x.strip() == 'True' for x in m.group(1).replace('\n', ' ').split(',')), 0.5)
for tag, v in {'bilinear': 0.69, 'linear': 0.83, 'markov': 0.69, 'ridge_x10': 0.70, 'ridge_d10': 0.69}.items():
    C(f'E18 latent error {tag}', v, json.load(open(R + f'lewm_kaggle/exp18_koopman_{tag}_meta.json'))['test_rel_err'], 0.006)
ex18 = json.load(open(R + 'lewm_kaggle/exp18_koopman_exploit.json')); C('E18 exploit operator ratio', 1.6, ex18['operator_true_over_pred']['median'], 0.06)

# --- retrained with the official loss (context control) ---
for tag, v in {'retrained_v2_seed42': 84, 'retrained_v2_seed43': 85, 'retrained_v2_seed44': 89, 'retrained_v2_seed45': 81}.items():
    C(f'E18 {tag}', v, succ(tag), 0.5)
cx = json.load(open(R + 'lewm_kaggle/retrain_v2_meta.json'))['context_length_errors']
C('context retrained old ctx1', 1.18, cx['retrained_last_position_loss']['context_1'], 0.006); C('context retrained old ctx2', 1.03, cx['retrained_last_position_loss']['context_2'], 0.006)
C('context retrained old ctx3', 0.006, cx['retrained_last_position_loss']['context_3'], 0.0006); C('context released ctx1', 0.066, cx['released']['context_1'], 0.0006)
C('context retrained v2 ctx1', 0.005, cx['retrained_all_positions_loss']['context_1'], 0.0006); C('context retrained v2 ctx3', 0.005, cx['retrained_all_positions_loss']['context_3'], 0.0006)
print(f'\n{sum(chk)} / {len(chk)} values match')
