import subprocess, time, os, glob, sys, json
t0 = time.time()
def run(cmd, timeout=12000):
    print(f"\n$ {cmd}  [t={time.time()-t0:.0f}s]", flush=True)
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr; print(out[:1500] + ('\n...\n' + out[-5000:] if len(out) > 6500 else out[1500:]), flush=True)
    if r.returncode != 0: raise SystemExit(f"STEP FAILED (exit {r.returncode}): {cmd}")
cache = glob.glob("/kaggle/input/**/emb_tworoom.npy", recursive=True); rt = glob.glob("/kaggle/input/**/tworoom_retrained/weights.pt", recursive=True)
if not cache or not rt: raise SystemExit(f"INPUTS MISSING cache={cache} retrained={rt}")
H = "/kaggle/working/stablewm"; os.environ.update(CACHE_DIR=os.path.dirname(cache[0]), RETRAINED=os.path.dirname(rt[0]), STABLEWM_HOME=H, WANDB_MODE="disabled", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", SDL_VIDEODRIVER="dummy")
run("nvidia-smi --query-gpu=name --format=csv,noheader; apt-get install -y -qq swig libegl1 libgl1 > /dev/null 2>&1; echo apt ok")
run("pip install 'stable-worldmodel[train,format]==0.1.1' 'transformers<5' hydra-core pygame pymunk shapely opencv-python-headless 2>&1 | tail -1")
run("python -c 'import hydra, stable_worldmodel, pygame, pymunk, shapely, cv2; print(\"imports ok\")'")
code = r'''
import os, json, time, numpy as np, torch
from scipy import stats
import stable_worldmodel as swm
t0 = time.time(); dev = "cuda"; H, FS = 3, 5; WARM, BLOCKS, K, M = 15, 5, 20, 40
D = os.environ["CACHE_DIR"]; act = np.nan_to_num(np.load(f"{D}/action.npy").astype(np.float32)); mu, sd = act.mean(0), act.std(0) + 1e-8
released = swm.wm.utils.load_pretrained("quentinll/lewm-tworooms").to(dev).eval(); retrained = swm.wm.utils.load_pretrained(os.environ["RETRAINED"]).to(dev).eval()
models = {"released": released, "retrained": retrained}
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
def encode(px_u8):
    px = torch.from_numpy(np.asarray(px_u8)).to(dev).permute(0, 3, 1, 2).float() / 255.0; px = (px - mean) / std
    with torch.no_grad(): return released.encode({"pixels": px.unsqueeze(1)})["emb"][:, 0].float()
def fwd(m, h, a):
    p = m.predictor(h, m.action_encoder(a)); B, T, Dd = p.shape; return m.pred_proj(p.reshape(B * T, Dd)).reshape(B, T, -1)[:, -1]
def predict_final(m, hist, A):
    """hist: (K, 3, 192) true embeddings at frames 0, 5, 10; A: (K, 40, 2) normalised actions; open-loop to frame 40"""
    h = hist.clone(); n_pred = (WARM + BLOCKS * FS - (H - 1) * FS) // FS    # frames 15, 20, ..., 40 -> 6 predictions
    with torch.no_grad():
        for k in range(n_pred):
            base = k * FS
            a = torch.stack([A[:, base + kk * FS: base + kk * FS + FS].reshape(A.shape[0], -1) for kk in range(H)], 1)
            p = fwd(m, h, a); h = torch.cat([h[:, 1:], p.unsqueeze(1)], 1)
    return p
world = swm.World("swm/TwoRoom-v1", num_envs=K, image_shape=(224, 224), max_episode_steps=200)
rngW, rngC = np.random.default_rng(10), np.random.default_rng(11)
rows = {mk: dict(spearman=[], top1_regret=[], pred_spread=[]) for mk in models}; true_spread = []
for m_ in range(M):
    world.reset(seed=[3000 + m_] * K)                       # K identical copies of the same start
    infos = world.infos; px = infos["pixels"]; px = px.reshape(-1, *px.shape[-3:]) if px.ndim == 5 else px
    warm = rngW.uniform(-1, 1, size=(WARM, 2)).astype(np.float32); cand = rngC.uniform(-1, 1, size=(K, BLOCKS * FS, 2)).astype(np.float32)
    frames = {}
    for t in range(WARM + BLOCKS * FS):
        if t in (0, 5, 10): frames[t] = np.stack([px[i] for i in range(K)])
        a = np.repeat(warm[t][None], K, 0) if t < WARM else cand[:, t - WARM]
        _, _, _, _, infos = world.envs.step(a); px = infos["pixels"]; px = px.reshape(-1, *px.shape[-3:]) if px.ndim == 5 else px
    frames[WARM + BLOCKS * FS] = np.stack([px[i] for i in range(K)])
    hist = torch.stack([encode(frames[t]) for t in (0, 5, 10)], 1); final_true = encode(frames[WARM + BLOCKS * FS])
    A = torch.as_tensor((np.concatenate([np.repeat(warm[None], K, 0), cand], 1) - mu) / sd, device=dev)
    goal = final_true[0]; true_cost = ((final_true[1:] - goal) ** 2).sum(1).cpu().numpy(); true_spread.append(float(true_cost.std() / (true_cost.mean() + 1e-9)))
    for mk, m in models.items():
        pf = predict_final(m, hist, A); pred_cost = ((pf[1:] - goal) ** 2).sum(1).cpu().numpy()
        rows[mk]["spearman"].append(float(stats.spearmanr(pred_cost, true_cost).correlation))
        rows[mk]["top1_regret"].append(float((true_cost[np.argmin(pred_cost)] - true_cost.min()) / (true_cost.mean() + 1e-9)))
        rows[mk]["pred_spread"].append(float(pred_cost.std() / (pred_cost.mean() + 1e-9)))
    if m_ % 10 == 9: print(f"start {m_+1}/{M} [{time.time()-t0:.0f}s]", flush=True)
res = {mk: {k: dict(mean=float(np.mean(v)), std=float(np.std(v))) for k, v in r.items()} for mk, r in rows.items()}
res["true_cost_spread"] = float(np.mean(true_spread)); res["starts"] = M; res["candidates_per_start"] = K - 1
d = np.array(rows["released"]["spearman"]) - np.array(rows["retrained"]["spearman"]); res["spearman_diff_released_minus_retrained"] = dict(mean=float(d.mean()), positive=int((d > 0).sum()), p_wilcoxon=float(stats.wilcoxon(d).pvalue))
d2 = np.array(rows["retrained"]["top1_regret"]) - np.array(rows["released"]["top1_regret"]); res["regret_diff_retrained_minus_released"] = dict(mean=float(d2.mean()), positive=int((d2 > 0).sum()), p_wilcoxon=float(stats.wilcoxon(d2).pvalue))
print(json.dumps(res, indent=1)); json.dump(res, open("/kaggle/working/ranking.json", "w"), indent=1)
'''
open("/kaggle/working/rk_run.py", "w").write(code)
run("python /kaggle/working/rk_run.py 2>&1 | grep -v Warning")
run("rm -f /kaggle/working/rk_run.py; cat /kaggle/working/ranking.json")
print(f"\nTOTAL {time.time()-t0:.0f}s")
