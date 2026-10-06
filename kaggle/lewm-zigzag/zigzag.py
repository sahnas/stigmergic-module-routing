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
# Zigzag probe: from cloned starts, a straight 25-step action sequence and a cancelling one (blocks alternating +d, -d)
# with the same per-step magnitude; predicted terminal displacement of each model versus the true one from the environment.
import os, json, time, numpy as np, torch
from scipy import stats
import stable_worldmodel as swm
t0 = time.time(); dev = "cuda"; H, FS, BLOCKS, M = 3, 5, 5, 40
D = os.environ["CACHE_DIR"]; act = np.nan_to_num(np.load(f"{D}/action.npy").astype(np.float32)); mu, sd = act.mean(0), act.std(0) + 1e-8
released = swm.wm.utils.load_pretrained("quentinll/lewm-tworooms").to(dev).eval(); retrained = swm.wm.utils.load_pretrained(os.environ["RETRAINED"]).to(dev).eval()
models = {"released": released, "retrained": retrained}
mean_ = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev); std_ = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
def encode(px_u8):
    px = torch.from_numpy(np.asarray(px_u8)).to(dev).permute(0, 3, 1, 2).float() / 255.0; px = (px - mean_) / std_
    with torch.no_grad(): return released.encode({"pixels": px.unsqueeze(1)})["emb"][:, 0].float()
def fwd(m, h, a):
    p = m.predictor(h, m.action_encoder(a)); B, T, Dd = p.shape; return m.pred_proj(p.reshape(B * T, Dd)).reshape(B, T, -1)[:, -1]
def predict_final(m, hist, A):
    h = hist.clone(); n_pred = (15 + BLOCKS * FS - (H - 1) * FS) // FS
    with torch.no_grad():
        for k in range(n_pred):
            base = k * FS; a = torch.stack([A[:, base + kk * FS: base + kk * FS + FS].reshape(A.shape[0], -1) for kk in range(H)], 1)
            p = fwd(m, h, a); h = torch.cat([h[:, 1:], p.unsqueeze(1)], 1)
    return p
world = swm.World("swm/TwoRoom-v1", num_envs=2, image_shape=(224, 224), max_episode_steps=200)
rngW, rngD = np.random.default_rng(10), np.random.default_rng(12)
def grab(px): return px.reshape(-1, *px.shape[-3:]) if px.ndim == 5 else px
rows = []
for m_ in range(M):
    world.reset(seed=[5000 + m_] * 2); px = grab(world.infos["pixels"]); frames = {}
    warm = rngW.uniform(-1, 1, size=(15, 2)).astype(np.float32)
    for t in range(15):
        if t in (0, 5, 10): frames[t] = px[0].copy()
        _, _, _, _, infos = world.envs.step(np.repeat(warm[t][None], 2, 0)); px = grab(infos["pixels"])
    hist = torch.stack([encode(frames[t][None]) for t in (0, 5, 10)], 1); z_start = encode(px[0][None])[0]
    d = rngD.uniform(-1, 1, size=2).astype(np.float32); d = d / (np.linalg.norm(d) + 1e-9) * 0.8
    straight = np.repeat(d[None], 25, 0); zig = np.concatenate([np.repeat((d if b % 2 == 0 else -d)[None], 5, 0) for b in range(5)])
    plans = np.stack([straight, zig])                                             # env 0 straight, env 1 zigzag
    for t in range(25):
        _, _, _, _, infos = world.envs.step(plans[:, t]); px = grab(infos["pixels"])
    z_true = encode(px); true_disp = (z_true - z_start).norm(dim=1).cpu().numpy()
    pos = infos.get("pos_agent"); 
    rec = dict(start=m_, true_disp_straight=float(true_disp[0]), true_disp_zig=float(true_disp[1]))
    for mk, m in models.items():
        A = torch.as_tensor((np.concatenate([np.repeat(warm[None], 2, 0), plans], 1) - mu) / sd, device=dev)
        pf = predict_final(m, hist.expand(2, -1, -1), A); pred_disp = (pf - z_start).norm(dim=1).cpu().numpy()
        err = (pf - z_true).norm(dim=1).cpu().numpy()
        rec[mk] = dict(pred_disp_straight=float(pred_disp[0]), pred_disp_zig=float(pred_disp[1]), err_straight=float(err[0]), err_zig=float(err[1]))
    rows.append(rec)
    if m_ % 10 == 9: print(f"start {m_+1}/{M} [{time.time()-t0:.0f}s]", flush=True)
out = dict(starts=M, true_disp_straight=float(np.mean([r["true_disp_straight"] for r in rows])), true_disp_zig=float(np.mean([r["true_disp_zig"] for r in rows])))
for mk in models:
    out[mk] = {k: float(np.mean([r[mk][k] for r in rows])) for k in rows[0][mk]}
    zr = np.array([r[mk]["pred_disp_zig"] for r in rows]) / np.maximum(np.array([r["true_disp_zig"] for r in rows]), 1e-6); out[mk]["zig_pred_over_true_median"] = float(np.median(zr))
d = np.array([r["retrained"]["err_zig"] - r["released"]["err_zig"] for r in rows]); out["zig_error_diff_retrained_minus_released"] = dict(mean=float(d.mean()), positive=int((d > 0).sum()), p_wilcoxon=float(stats.wilcoxon(d).pvalue))
d2 = np.array([r["retrained"]["err_straight"] - r["released"]["err_straight"] for r in rows]); out["straight_error_diff_retrained_minus_released"] = dict(mean=float(d2.mean()), positive=int((d2 > 0).sum()), p_wilcoxon=float(stats.wilcoxon(d2).pvalue))
print(json.dumps(out, indent=1)); json.dump(dict(summary=out, rows=rows), open("/kaggle/working/zigzag.json", "w"), indent=1)
'''
open("/kaggle/working/zz_run.py", "w").write(code)
run("python /kaggle/working/zz_run.py 2>&1 | grep -v Warning")
run("rm -f /kaggle/working/zz_run.py; cat /kaggle/working/zigzag.json")
print(f"\nTOTAL {time.time()-t0:.0f}s")
