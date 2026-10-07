import subprocess, time, os, glob, sys, json
t0 = time.time()
def run(cmd, timeout=20000):
    print(f"\n$ {cmd}  [t={time.time()-t0:.0f}s]", flush=True)
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr; print(out[:2000] + ('\n...\n' + out[-5000:] if len(out) > 7000 else out[2000:]), flush=True)
    if r.returncode != 0: raise SystemExit(f"STEP FAILED (exit {r.returncode}): {cmd}")
cache = glob.glob("/kaggle/input/**/emb_tworoom.npy", recursive=True)
if not cache: raise SystemExit(f"INPUTS MISSING cache={cache}")
H = "/kaggle/working/stablewm"; os.environ.update(CACHE_DIR=os.path.dirname(cache[0]), STABLEWM_HOME=H, WANDB_MODE="disabled", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
run("nvidia-smi --query-gpu=name --format=csv,noheader; apt-get install -y -qq swig libegl1 libgl1 > /dev/null 2>&1; echo apt ok")
# the [env] extra pulls gymnasium[all], whose labmaze and box2d wheels do not build on Kaggle; Two-rooms only needs pygame, pymunk, shapely, opencv
run("pip install 'stable-worldmodel[train,format]==0.1.1' 'transformers<5' hydra-core pygame pymunk shapely opencv-python-headless 2>&1 | tail -2")
run("python -c 'import hydra, stable_worldmodel, h5py, hdf5plugin, gymnasium, pygame, pymunk, shapely, cv2; print(\"imports ok\")'; pip freeze | grep -iE 'stable-worldmodel|^torch==|transformers|hydra-core|pymunk|pygame'")
train = r'''
import os, json, time, numpy as np, torch, torch.nn as nn, hydra, shutil
from huggingface_hub import hf_hub_download, snapshot_download
import stable_worldmodel as swm
t0 = time.time(); torch.manual_seed(0); rng = np.random.default_rng(0); D = os.environ["CACHE_DIR"]; dev = "cuda"
emb = np.load(f"{D}/emb_tworoom.npy").astype(np.float32); act = np.nan_to_num(np.load(f"{D}/action.npy").astype(np.float32))
ep_len = np.load(f"{D}/ep_len.npy"); ep_off = np.load(f"{D}/ep_offset.npy"); H, FS = 3, 5; SPAN = H * FS
# same episode-level split and normalisation as the sanity kernel (seed 0)
eps = np.arange(len(ep_len)); test_eps = set(rng.choice(eps, size=len(eps) // 10, replace=False).tolist())
starts = np.array([i for e in eps for i in range(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e]) - SPAN)]); is_test = np.array([e in test_eps for e in eps for i in range(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e]) - SPAN)])
tr, te = starts[~is_test], starts[is_test]
tr_frames = np.concatenate([np.arange(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e])) for e in eps if e not in test_eps])
a_mean, a_std = act[tr_frames].mean(0), act[tr_frames].std(0) + 1e-8; actn = (act - a_mean) / a_std
emb_t = torch.from_numpy(emb).to(dev); act_t = torch.from_numpy(actn).to(dev)
def window(idx):
    idx = torch.as_tensor(idx, device=dev)
    return torch.stack([emb_t[idx + k * FS] for k in range(H)], 1), torch.stack([torch.cat([act_t[idx + k * FS + j] for j in range(FS)], 1) for k in range(H)], 1), emb_t[idx + H * FS]
def window_all(idx):
    idx = torch.as_tensor(idx, device=dev)
    return torch.stack([emb_t[idx + k * FS] for k in range(H)], 1), torch.stack([torch.cat([act_t[idx + k * FS + j] for j in range(FS)], 1) for k in range(H)], 1), torch.stack([emb_t[idx + (k + 1) * FS] for k in range(H)], 1)
model = swm.wm.utils.load_pretrained("quentinll/lewm-tworooms").to(dev).eval()
def eval_fn(fn, idxs, bs=2048):
    se, n, tg = 0.0, 0, []
    with torch.no_grad():
        for i in range(0, len(idxs), bs):
            h, a, y = window(idxs[i:i+bs]); p = fn(h, a); se += ((p - y) ** 2).sum().item(); n += y.numel(); tg.append(y)
    return se / n / torch.cat(tg).var(0).mean().item()
stack = lambda m: (m.action_encoder, m.predictor, m.pred_proj)
def fwd_all(m, h, a):
    """predictions at every position (B, T, 192): position t predicts the embedding 5 frames after frame t, from the context up to t"""
    p = m.predictor(h, m.action_encoder(a)); B, T, Dd = p.shape; return m.pred_proj(p.reshape(B * T, Dd)).reshape(B, T, -1)
def fwd(m, h, a): return fwd_all(m, h, a)[:, -1]
print("released predictor relative error", round(eval_fn(lambda h, a: fwd(model, h, a), te), 4), flush=True)
# retrain the predictor stack from scratch on the frozen cache (same recipe as the sanity kernel: 8 epochs, OneCycle 3e-4, batch 256)
cfg = json.load(open(hf_hub_download("quentinll/lewm-tworooms", "config.json")))
import copy
new = copy.deepcopy(model); torch.manual_seed(1)
new.action_encoder = hydra.utils.instantiate(cfg["action_encoder"]).to(dev); new.predictor = hydra.utils.instantiate(cfg["predictor"]).to(dev); new.pred_proj = hydra.utils.instantiate(cfg["pred_proj"]).to(dev)
params = [p for m in stack(new) for p in m.parameters()]
opt = torch.optim.AdamW(params, 3e-4, weight_decay=1e-3); bs, epochs = 256, 8
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * ((len(tr) + bs - 1) // bs), pct_start=0.05)
for ep in range(epochs):
    perm = rng.permutation(len(tr))
    for i in range(0, len(perm), bs):
        h, a, y = window_all(tr[perm[i:i+bs]]); loss = ((fwd_all(new, h, a) - y) ** 2).mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()   # official loss: every position supervised
    print(f"epoch {ep+1}: test relative {eval_fn(lambda h, a: fwd(new, h, a), te):.4f} [{time.time()-t0:.0f}s]", flush=True)
# save in the folder layout load_pretrained expects, next to the released checkpoint for comparison
out = os.path.join(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), "tworoom_retrained_v2"); os.makedirs(out, exist_ok=True)
torch.save({k: v.detach().cpu() for k, v in new.state_dict().items()}, os.path.join(out, "weights.pt")); json.dump(cfg, open(os.path.join(out, "config.json"), "w"))
chk = swm.wm.utils.load_pretrained(out).to(dev).eval(); print("reloaded retrained model relative error", round(eval_fn(lambda h, a: fwd(chk, h, a), te), 4), flush=True)
# NOTE: the retrained predictor was trained with the cache's action normalisation (training-frame z-score); eval.py normalises actions with the dataset stats loaded from the h5, which the probe showed to be equivalent (0.0670 vs 0.0671 for the released predictor).
# context-length analysis: error of the prediction made from a context of 1, 2 or 3 real frames (the planner's first calls use 1, 2, 3)
old = swm.wm.utils.load_pretrained(os.environ["RETRAINED_OLD"]).to(dev).eval() if os.environ.get("RETRAINED_OLD") else None
Wop = None
def ctx_err(fn_all, idxs, L, bs=2048):
    se, n, tg = 0.0, 0, []
    with torch.no_grad():
        for i in range(0, len(idxs), bs):
            h, a, y = window_all(idxs[i:i+bs]); p = fn_all(h[:, :L], a[:, :L])[:, -1]; yy = y[:, L - 1]; se += ((p - yy) ** 2).sum().item(); n += yy.numel(); tg.append(yy)
    return se / n / torch.cat(tg).var(0).mean().item()
ctx = {}
for name, m in [("released", model), ("retrained_last_position_loss", old), ("retrained_all_positions_loss", chk)]:
    if m is None: continue
    ctx[name] = {f"context_{L}": round(ctx_err(lambda h, a: fwd_all(m, h, a), te, L), 4) for L in (1, 2, 3)}
    print(name, ctx[name], flush=True)
json.dump(dict(a_mean=a_mean.tolist(), a_std=a_std.tolist(), epochs=epochs, context_length_errors=ctx, seconds=round(time.time() - t0)), open("/kaggle/working/retrain_v2_meta.json", "w"))
'''
open("/kaggle/working/train_pred.py", "w").write(train)
rt_old = glob.glob("/kaggle/input/**/tworoom_retrained/weights.pt", recursive=True)
if rt_old: os.environ["RETRAINED_OLD"] = os.path.dirname(rt_old[0])
run("python /kaggle/working/train_pred.py 2>&1 | grep -v Warning")
run("cp -r /kaggle/working/stablewm/checkpoints/tworoom_retrained_v2 /kaggle/working/; ls -la /kaggle/working/tworoom_retrained_v2")
run("rm -f /kaggle/working/train_pred.py; ls -la /kaggle/working/")
print(f"\nTOTAL {time.time()-t0:.0f}s")
