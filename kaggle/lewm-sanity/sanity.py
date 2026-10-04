import subprocess, time, os, glob, sys, json
t0 = time.time()
def run(cmd, timeout=6000):
    print(f"\n$ {cmd}  [t={time.time()-t0:.0f}s]", flush=True)
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr
    print(out[:1500] + ('\n...\n' + out[-3500:] if len(out) > 5000 else out[1500:]), flush=True)
    if r.returncode != 0:
        raise SystemExit(f"STEP FAILED (exit {r.returncode}): {cmd}")
cache = glob.glob("/kaggle/input/**/emb_tworoom.npy", recursive=True)
if not cache: raise SystemExit("CACHE MISSING")
D = os.path.dirname(cache[0]); os.environ["CACHE_DIR"] = D
os.environ["STABLEWM_HOME"] = "/kaggle/working/stablewm"; os.environ["WANDB_MODE"] = "disabled"
run("nvidia-smi --query-gpu=name --format=csv,noheader; ls -la $CACHE_DIR | head -20")
run("pip install 'stable-worldmodel[train,format]==0.1.1' 'transformers<5' 2>&1 | tail -2")
code = r'''
import os, json, time, numpy as np, torch, torch.nn as nn
import stable_worldmodel as swm
t0 = time.time(); torch.manual_seed(0); rng = np.random.default_rng(0)
D = os.environ["CACHE_DIR"]; dev = "cuda"
emb = np.load(f"{D}/emb_tworoom.npy").astype(np.float32); act = np.nan_to_num(np.load(f"{D}/action.npy").astype(np.float32))
ep_len = np.load(f"{D}/ep_len.npy"); ep_off = np.load(f"{D}/ep_offset.npy"); N = len(emb)
H, FS = 3, 5                       # history_size, frameskip of the Two-rooms config
SPAN = H * FS                      # frames i, i+5, i+10 as history, i+15 as target; actions in blocks of 5
# episode-level split, fixed before anything else
eps = np.arange(len(ep_len)); test_eps = set(rng.choice(eps, size=len(eps) // 10, replace=False).tolist())
starts, is_test = [], []
for e in eps:
    o, L = int(ep_off[e]), int(ep_len[e])
    for i in range(o, o + L - SPAN):
        starts.append(i); is_test.append(e in test_eps)
starts = np.array(starts); is_test = np.array(is_test)
tr, te = starts[~is_test], starts[is_test]
print("windows train", len(tr), "test", len(te), "| test episodes", len(test_eps), flush=True)
# action normalisation estimated on training frames only, kept fixed
tr_frames = np.concatenate([np.arange(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e])) for e in eps if e not in test_eps])
a_mean, a_std = act[tr_frames].mean(0), act[tr_frames].std(0) + 1e-8
actn = (act - a_mean) / a_std
emb_t = torch.from_numpy(emb).to(dev); act_t = torch.from_numpy(actn).to(dev)
def window(idx):
    idx = torch.as_tensor(idx, device=dev)
    hist = torch.stack([emb_t[idx + k * FS] for k in range(H)], 1)                      # (B, 3, 192)
    blocks = torch.stack([torch.cat([act_t[idx + k * FS + j] for j in range(FS)], 1) for k in range(H)], 1)   # (B, 3, 10)
    target = emb_t[idx + H * FS]
    return hist, blocks, target
# 1. reference: the released predictor on the test windows
model = swm.wm.utils.load_pretrained("quentinll/lewm-tworooms").to(dev).eval()
tvar = None
def eval_fn(pred_fn, idxs, bs=2048):
    se, n, tgts = 0.0, 0, []
    with torch.no_grad():
        for i in range(0, len(idxs), bs):
            h, a, y = window(idxs[i:i+bs]); p = pred_fn(h, a)
            se += ((p - y) ** 2).sum().item(); n += y.numel(); tgts.append(y)
    y_all = torch.cat(tgts); var = y_all.var(0).mean().item()
    return se / n, var
def ref_fn(h, a):
    ae = model.action_encoder(a)
    return model.predict(h, ae)[:, -1]
mse_ref, var = eval_fn(ref_fn, te); print(f"reference predictor: mse {mse_ref:.4f} relative {mse_ref/var:.4f} (target var {var:.4f}) [{time.time()-t0:.0f}s]", flush=True)
mse_copy, _ = eval_fn(lambda h, a: h[:, -1], te); print(f"copy-last baseline: relative {mse_copy/var:.4f}", flush=True)
# 2. fresh predictors trained on the cache, same windows and normalisation
import hydra
from huggingface_hub import hf_hub_download
cfg = json.load(open(hf_hub_download("quentinll/lewm-tworooms", "config.json")))
class LeWMPred(nn.Module):
    """LeWM's own predictor stack (action encoder, predictor, pred_proj) at random init"""
    def __init__(self):
        super().__init__()
        self.action_encoder = hydra.utils.instantiate(cfg["action_encoder"]); self.predictor = hydra.utils.instantiate(cfg["predictor"]); self.pred_proj = hydra.utils.instantiate(cfg["pred_proj"])
    def forward(self, h, a):
        p = self.predictor(h, self.action_encoder(a)); B, T, Dd = p.shape
        return self.pred_proj(p.reshape(B * T, Dd)).reshape(B, T, -1)[:, -1]
class Fresh(nn.Module):
    def __init__(self, d=192, a=10, hid=512):
        super().__init__(); self.net = nn.Sequential(nn.Linear(H * (d + a), hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, d))
    def forward(self, h, a): return self.net(torch.cat([h.flatten(1), a.flatten(1)], 1)) + h[:, -1]
def train_eval(name, net, epochs, lr, bs=256):
    torch.manual_seed(1); net = net.to(dev); opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-3)
    total = epochs * ((len(tr) + bs - 1) // bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.05)
    steps = 0; curve = []
    for ep in range(epochs):
        perm = rng.permutation(len(tr))
        for i in range(0, len(perm), bs):
            h, a, y = window(tr[perm[i:i+bs]]); loss = ((net(h, a) - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); steps += 1
        net.eval(); mse, _ = eval_fn(net, te); net.train(); curve.append(round(mse / var, 4))
        print(f"{name} epoch {ep+1}: test relative {mse/var:.4f} [{time.time()-t0:.0f}s]", flush=True)
    return dict(relative=curve[-1], curve=curve, steps=steps, lr=lr, batch=bs)
res = {}
res["lewm_arch_fresh_8ep"] = train_eval("lewm_arch_fresh", LeWMPred(), 8, 3e-4)
res["mlp_fresh_8ep"] = train_eval("mlp_fresh", Fresh(), 8, 3e-4)
# upper bound: the released predictor stack fine-tuned on the cache for 2 epochs
ft = LeWMPred(); ft.action_encoder.load_state_dict(model.action_encoder.state_dict()); ft.predictor.load_state_dict(model.predictor.state_dict()); ft.pred_proj.load_state_dict(model.pred_proj.state_dict())
res["released_finetuned_2ep"] = train_eval("released_finetuned", ft, 2, 5e-5)
meta = dict(windows_train=int(len(tr)), windows_test=int(len(te)), history=H, frameskip=FS, action_block_dim=int(act.shape[1] * FS),
            action_norm_source="training frames", reference_relative=mse_ref / var, copy_last_relative=mse_copy / var, target_var=var, trained=res, seconds=round(time.time() - t0))
json.dump(meta, open("/kaggle/working/sanity_meta.json", "w"), indent=1); print(json.dumps(meta, indent=1))
'''
open("/kaggle/working/sanity_run.py", "w").write(code)
run("python /kaggle/working/sanity_run.py 2>&1 | grep -v Warning")
run("rm -f /kaggle/working/sanity_run.py; cat /kaggle/working/sanity_meta.json")
print(f"\nTOTAL {time.time()-t0:.0f}s")
