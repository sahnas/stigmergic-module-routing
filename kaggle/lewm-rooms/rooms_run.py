
import os, glob, json, numpy as np
from scipy import stats
D = os.path.dirname(glob.glob("/kaggle/input/**/pos_agent.npy", recursive=True)[0])
pos = np.load(f"{D}/pos_agent.npy").astype(np.float32); ep_idx = np.load(f"{D}/ep_idx.npy"); ep_len = np.load(f"{D}/ep_len.npy"); ep_off = np.load(f"{D}/ep_offset.npy")
N = len(pos); step_idx = np.arange(N) - ep_off[ep_idx]
GOAL, BUDGET, SEED, NUM = 25, 50, 42, 100
max_start_per_row = ep_len[ep_idx] - GOAL - 1
valid = np.nonzero(step_idx <= max_start_per_row)[0]
g = np.random.default_rng(SEED); pick = g.choice(len(valid) - 1, size=NUM, replace=False); rows = np.sort(valid[pick])
WALL, AXIS = 112.0, 1          # TwoRoomEnv: WALL_CENTER = 112, wall_axis = 1
start = pos[rows]; goal = pos[rows + GOAL]
cross = (start[:, AXIS] < WALL) != (goal[:, AXIS] < WALL)
succ = json.loads('''{"released": [false, true, true, true, true, false, true, true, true, true, true, true, false, true, true, false, true, true, true, true, true, true, false, true, true, true, true, true, true, false, true, true, false, true, true, true, true, true, true, true, true, false, true, true, true, false, false, true, true, true, true, true, true, true, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, true, true, true, false, true, true, true, true, false, true, true, false, true, true, true, true], "retrained": [true, false, false, true, true, false, false, false, true, false, false, true, false, false, false, false, true, true, true, false, false, true, false, false, true, true, true, true, false, false, true, false, false, false, true, false, false, true, true, true, false, false, true, false, false, false, false, false, true, false, true, true, false, false, false, false, false, false, true, false, false, true, false, false, false, false, false, false, true, true, false, false, true, false, true, false, false, false, true, false, true, false, true, false, true, false, false, false, true, false, false, false, false, false, false, true, true, true, false, false]}''')
out = dict(episodes=int(NUM), cross_room=int(cross.sum()), same_room=int((~cross).sum()))
for tag, s in succ.items():
    s = np.array(s); out[tag] = dict(overall=float(s.mean()), same_room=float(s[~cross].mean()), cross_room=float(s[cross].mean()) if cross.any() else None)
a = np.array(succ["released"]); b = np.array(succ["retrained"])
for name, m in [("same_room", ~cross), ("cross_room", cross)]:
    tab = [[int((a[m] & b[m]).sum()), int((a[m] & ~b[m]).sum())], [int((~a[m] & b[m]).sum()), int((~a[m] & ~b[m]).sum())]]
    out[f"mcnemar_{name}"] = dict(table=tab, p=float(stats.binomtest(tab[0][1], tab[0][1] + tab[1][0]).pvalue) if (tab[0][1] + tab[1][0]) else None)
d = np.linalg.norm(goal - start, axis=1); out["goal_distance_px"] = dict(same_room=float(d[~cross].mean()), cross_room=float(d[cross].mean()) if cross.any() else None)
print(json.dumps(out, indent=1)); json.dump(out, open("/kaggle/working/rooms.json", "w"), indent=1)
