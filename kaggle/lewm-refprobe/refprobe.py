import subprocess, time, os, glob, sys, json
t0 = time.time()
def run(cmd, timeout=3000):
    print(f"\n$ {cmd}  [t={time.time()-t0:.0f}s]", flush=True)
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr; print(out[:1500] + ('\n...\n' + out[-3500:] if len(out) > 5000 else out[1500:]), flush=True)
    if r.returncode != 0: raise SystemExit(f"STEP FAILED (exit {r.returncode}): {cmd}")
cache = glob.glob("/kaggle/input/**/emb_tworoom.npy", recursive=True)
if not cache: raise SystemExit("CACHE MISSING")
os.environ["CACHE_DIR"] = os.path.dirname(cache[0]); os.environ["STABLEWM_HOME"] = "/kaggle/working/stablewm"; os.environ["WANDB_MODE"] = "disabled"
run("pip install 'stable-worldmodel[train,format]==0.1.1' 'transformers<5' 2>&1 | tail -1")
code = r'''
import os, json, time, numpy as np, torch
import stable_worldmodel as swm
t0 = time.time(); rng = np.random.default_rng(0); D = os.environ["CACHE_DIR"]; dev = "cuda"
emb = np.load(f"{D}/emb_tworoom.npy").astype(np.float32); act = np.nan_to_num(np.load(f"{D}/action.npy").astype(np.float32))
ep_len = np.load(f"{D}/ep_len.npy"); ep_off = np.load(f"{D}/ep_offset.npy"); H, FS = 3, 5; SPAN = H * FS
eps = np.arange(len(ep_len)); test_eps = set(rng.choice(eps, size=len(eps) // 10, replace=False).tolist())
te = np.array([i for e in eps if e in test_eps for i in range(int(ep_off[e]) + FS, int(ep_off[e]) + int(ep_len[e]) - SPAN)])   # +FS margin so that preceding-action blocks exist
tr_frames = np.concatenate([np.arange(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e])) for e in eps if e not in test_eps])
model = swm.wm.utils.load_pretrained("quentinll/lewm-tworooms").to(dev).eval()
emb_t = torch.from_numpy(emb).to(dev)
def evaluate(actn, offset, bs=2048):
    act_t = torch.from_numpy(actn).to(dev); se = 0.0; n = 0; tg = []
    with torch.no_grad():
        for i in range(0, len(te), bs):
            idx = torch.as_tensor(te[i:i+bs], device=dev)
            hist = torch.stack([emb_t[idx + k * FS] for k in range(H)], 1)
            blocks = torch.stack([torch.cat([act_t[idx + k * FS + j + offset] for j in range(FS)], 1) for k in range(H)], 1)
            y = emb_t[idx + H * FS]; p = model.predict(hist, model.action_encoder(blocks))[:, -1]
            se += ((p - y) ** 2).sum().item(); n += y.numel(); tg.append(y)
    var = torch.cat(tg).var(0).mean().item(); return se / n / var
conventions = {
    "zscore_train_frames, block starts at frame": ((act - act[tr_frames].mean(0)) / (act[tr_frames].std(0) + 1e-8), 0),
    "zscore_all_frames, block starts at frame": ((act - act.mean(0)) / (act.std(0) + 1e-8), 0),
    "raw actions, block starts at frame": (act, 0),
    "zscore_train_frames, block precedes frame": ((act - act[tr_frames].mean(0)) / (act[tr_frames].std(0) + 1e-8), -FS),
    "zscore_train_frames, block starts at frame+1": ((act - act[tr_frames].mean(0)) / (act[tr_frames].std(0) + 1e-8), 1),
}
res = {k: evaluate(a, o) for k, (a, o) in conventions.items()}
for k, v in res.items(): print(f"reference relative error | {k}: {v:.4f}", flush=True)
json.dump(dict(test_windows=int(len(te)), action_mean_train=act[tr_frames].mean(0).tolist(), action_std_train=act[tr_frames].std(0).tolist(), reference_by_convention=res), open("/kaggle/working/refprobe.json", "w"), indent=1)
'''
open("/kaggle/working/refprobe_run.py", "w").write(code)
run("python /kaggle/working/refprobe_run.py 2>&1 | grep -v Warning")
run("rm -f /kaggle/working/refprobe_run.py; cat /kaggle/working/refprobe.json | head -5")
print(f"\nTOTAL {time.time()-t0:.0f}s")
