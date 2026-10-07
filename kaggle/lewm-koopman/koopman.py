import subprocess, time, os, glob, sys, json
t0 = time.time()
def run(cmd, timeout=20000):
    print(f"\n$ {cmd}  [t={time.time()-t0:.0f}s]", flush=True)
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr; print(out[:1500] + ('\n...\n' + out[-5000:] if len(out) > 6500 else out[1500:]), flush=True)
    if r.returncode != 0: raise SystemExit(f"STEP FAILED (exit {r.returncode}): {cmd}")
cache = glob.glob("/kaggle/input/**/emb_tworoom.npy", recursive=True); h5 = glob.glob("/kaggle/input/**/tworoom.h5", recursive=True)
if not cache or not h5: raise SystemExit(f"INPUTS MISSING cache={cache} h5={h5}")
H = "/kaggle/working/stablewm"; os.environ.update(CACHE_DIR=os.path.dirname(cache[0]), STABLEWM_HOME=H, WANDB_MODE="disabled", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
os.makedirs(f"{H}/datasets", exist_ok=True); os.symlink(h5[0], f"{H}/datasets/tworoom.h5")
run("nvidia-smi --query-gpu=name --format=csv,noheader; apt-get install -y -qq swig libegl1 libgl1 > /dev/null 2>&1; echo apt ok")
run("pip install 'stable-worldmodel[train,format]==0.1.1' 'transformers<5' hydra-core pygame pymunk shapely opencv-python-headless 2>&1 | tail -1")
run("python -c 'import hydra, stable_worldmodel, pygame, pymunk, shapely, cv2; print(\"imports ok\")'")
run("git clone https://github.com/lucas-maes/le-wm.git /kaggle/tmp/le-wm > /dev/null 2>&1; cd /kaggle/tmp/le-wm && git checkout -q 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac && git log -1 --format='le-wm %h %cd' --date=short")
# ---- 1. fit the operator predictor by ridge least squares on the frozen cache ----
fit = r'''
import os, json, time, numpy as np, torch
t0 = time.time(); D = os.environ["CACHE_DIR"]; dev = "cuda"; H, FS = 3, 5
emb = np.load(f"{D}/emb_tworoom.npy").astype(np.float32); act = np.nan_to_num(np.load(f"{D}/action.npy").astype(np.float32))
ep_len = np.load(f"{D}/ep_len.npy"); ep_off = np.load(f"{D}/ep_offset.npy")
rng = np.random.default_rng(0); eps = np.arange(len(ep_len)); test_eps = set(rng.choice(eps, size=len(eps) // 10, replace=False).tolist())
mu, sd = act.mean(0), act.std(0) + 1e-8; A = (act - mu) / sd          # dataset statistics, as eval.py's StandardScaler
starts = np.array([i for e in eps for i in range(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e]) - H * FS)]); is_test = np.array([e in test_eps for e in eps for i in range(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e]) - H * FS)])
tr, te = starts[~is_test], starts[is_test]
E = torch.from_numpy(emb).to(dev); At = torch.from_numpy(A).to(dev)
def feats(idx):
    """one-step features at the last history position: z_t, z_{t-5}, z_{t-10}, the 10-d action block, and z_t (x) block; target z_{t+5}"""
    idx = torch.as_tensor(idx, device=dev)
    z2, z1, z0 = E[idx], E[idx + FS], E[idx + 2 * FS]                   # z0 is the most recent frame (t), z2 the oldest
    a = torch.cat([At[idx + 2 * FS + j] for j in range(FS)], 1)         # block of 5 actions at t
    bil = (z0.unsqueeze(2) * a.unsqueeze(1)).reshape(len(idx), -1)      # 192 x 10
    X = torch.cat([z0, z1, z2, a, bil, torch.ones(len(idx), 1, device=dev)], 1)
    return X, E[idx + 3 * FS]
dX = 3 * 192 + 10 + 1920 + 1
XtX = torch.zeros(dX, dX, device=dev, dtype=torch.float64); XtY = torch.zeros(dX, 192, device=dev, dtype=torch.float64)
for i in range(0, len(tr), 8192):
    X, Y = feats(tr[i:i+8192]); X, Y = X.double(), Y.double(); XtX += X.T @ X; XtY += X.T @ Y
lam = 1e-2 * len(tr)
W = torch.linalg.solve(XtX + lam * torch.eye(dX, device=dev, dtype=torch.float64), XtY).float()
def rel_err(idxs):
    se, n, tg = 0.0, 0, []
    for i in range(0, len(idxs), 8192):
        X, Y = feats(idxs[i:i+8192]); P = X @ W; se += ((P - Y) ** 2).sum().item(); n += Y.numel(); tg.append(Y)
    return se / n / torch.cat(tg).var(0).mean().item()
print(f"operator predictor (ridge, {dX} features): relative one-step error train {rel_err(tr[:200000]):.4f} test {rel_err(te):.4f} [{time.time()-t0:.0f}s]", flush=True)
torch.save(dict(W=W.cpu(), mu=torch.tensor(mu), sd=torch.tensor(sd)), "/kaggle/working/koopman_weights.pt")
json.dump(dict(features=dX, ridge=lam, test_rel_err=rel_err(te)), open("/kaggle/working/koopman_meta.json", "w"))
'''
open("/kaggle/working/fit_koopman.py", "w").write(fit); run("python /kaggle/working/fit_koopman.py 2>&1 | grep -v Warning")
# ---- 2. eval.py patched: policy=koopman builds the released model and swaps in the operator predictor ----
patch = r'''
import torch as _torch, torch.nn as _nn
class _KoopmanPredictor(_nn.Module):
    """causal operator predictor on the LeWM interface: emb (B,T,192), act (B,T,10) normalised blocks -> predictions (B,T,192)"""
    def __init__(self, W):
        super().__init__(); self.register_buffer("W", W)
    def forward(self, emb, act):
        B, T, Dd = emb.shape; outs = []
        for t in range(T):
            z0 = emb[:, t]; z1 = emb[:, t - 1] if t >= 1 else emb[:, t]; z2 = emb[:, t - 2] if t >= 2 else z1
            a = act[:, t]; bil = (z0.unsqueeze(2) * a.unsqueeze(1)).reshape(B, -1)
            X = _torch.cat([z0, z1, z2, a, bil, _torch.ones(B, 1, device=emb.device, dtype=emb.dtype)], 1)
            outs.append(X @ self.W.to(emb.dtype))
        return _torch.stack(outs, 1)
def _koopman_model():
    import stable_worldmodel as swm
    m = swm.wm.utils.load_pretrained("quentinll/lewm-tworooms")
    w = _torch.load("/kaggle/working/koopman_weights.pt", map_location="cpu")
    m.predictor = _KoopmanPredictor(w["W"]); m.action_encoder = _nn.Identity(); m.pred_proj = _nn.Identity()
    return m
'''
code = open("/kaggle/tmp/le-wm/eval.py").read()
code = code.replace("import stable_worldmodel as swm", "import stable_worldmodel as swm\n" + patch, 1)
code = code.replace('        model = swm.wm.utils.load_pretrained(cfg.policy)', '        model = _koopman_model() if cfg.policy == "koopman" else swm.wm.utils.load_pretrained(cfg.policy)', 1)
assert "_koopman_model() if" in code, "patch failed"
open("/kaggle/tmp/le-wm/eval_koopman.py", "w").write(code)
run("cd /kaggle/tmp/le-wm && python eval_koopman.py --config-name=tworoom.yaml policy=koopman eval.num_eval=100 output.filename=/kaggle/working/koopman_results.txt 2>&1 | grep -vE 'arn' | tail -8", timeout=12000)
run("cat /kaggle/working/koopman_results.txt | tail -5; cat /kaggle/working/koopman_meta.json")
print(f"\nTOTAL {time.time()-t0:.0f}s")
