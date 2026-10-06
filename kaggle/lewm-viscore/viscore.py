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
# VIScore factors (Wu, Balestriero, Levine, arXiv 2608.11174, Appendix B.1), computed from the paper's definitions on the
# released and retrained Two-rooms predictors, on expert windows of the frozen-encoder cache.
import os, json, time, numpy as np, torch
import stable_worldmodel as swm
t0 = time.time(); D = os.environ["CACHE_DIR"]; dev = "cuda"; H, FS, HOR = 3, 5, 5          # horizon H = 5 action blocks
emb = np.load(f"{D}/emb_tworoom.npy").astype(np.float32); act = np.nan_to_num(np.load(f"{D}/action.npy").astype(np.float32))
pos = np.load(f"{D}/pos_agent.npy").astype(np.float32); ep_len = np.load(f"{D}/ep_len.npy"); ep_off = np.load(f"{D}/ep_offset.npy")
rng = np.random.default_rng(0); eps = np.arange(len(ep_len)); test_eps = sorted(set(rng.choice(eps, size=len(eps) // 10, replace=False).tolist()))
mu, sd = act.mean(0), act.std(0) + 1e-8; A = (act - mu) / sd
emb_t = torch.from_numpy(emb).to(dev); A_t = torch.from_numpy(A).to(dev)
released = swm.wm.utils.load_pretrained("quentinll/lewm-tworooms").to(dev).eval(); retrained = swm.wm.utils.load_pretrained(os.environ["RETRAINED"]).to(dev).eval()
models = {"released": released, "retrained": retrained}
def fwd(m, h, a):
    p = m.predictor(h, m.action_encoder(a)); B, T, Dd = p.shape; return m.pred_proj(p.reshape(B * T, Dd)).reshape(B, T, -1)[:, -1]
def rollout(m, hist, blocks):
    # hist (B,3,192) true; blocks (B, 3+HOR, 10) normalised action blocks; open-loop HOR steps; returns the terminal prediction
    h = hist.clone()
    with torch.no_grad():
        for k in range(HOR):
            p = fwd(m, h, blocks[:, k:k + H]); h = torch.cat([h[:, 1:], p.unsqueeze(1)], 1)
    return p
# windows: start i, history frames i, i+5, i+10, blocks from i, terminal frame i + (H-1+HOR)*5 = i + 35
span = (H - 1 + HOR) * FS
starts = np.array([i for e in test_eps for i in range(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e]) - span)])
def blocks_at(idx):
    return torch.stack([torch.cat([A_t[idx + k * FS + j] for j in range(FS)], 1) for k in range(H + HOR)], 1)
def hist_at(idx): return torch.stack([emb_t[idx + k * FS] for k in range(H)], 1)
# task tolerance in latent units: pairs of frames whose agent moved by <= 16 px * 1.2 within 1..5 frames, median latent distance (Appendix B.3)
pairs = []
for e in test_eps[:200]:
    o, L = int(ep_off[e]), int(ep_len[e])
    for i in range(o, o + L - 5):
        for d in range(1, 6):
            if np.linalg.norm(pos[i + d] - pos[i]) <= 16 * 1.2: pairs.append((i, i + d))
pairs = np.array(pairs[:20000]); dtol = float(np.median(np.linalg.norm(emb[pairs[:, 1]] - emb[pairs[:, 0]], axis=1))); tr_sigma = float(emb[starts].var(0).sum())
print(f"d_tol (latent) {dtol:.3f} from {len(pairs)} probe pairs | tr Sigma_z {tr_sigma:.2f}", flush=True)
res = {"d_tol": dtol}
sel = rng.choice(len(starts), size=4096, replace=False); idx = torch.as_tensor(starts[sel], device=dev)
hist, blocks, target = hist_at(idx), blocks_at(idx), emb_t[idx + span]
for mk, m in models.items():
    # veracity
    term = rollout(m, hist, blocks); nmse = float(((term - target) ** 2).sum(1).mean() / tr_sigma); sigma_roll = float(np.sqrt(nmse * tr_sigma))
    veracity = float(torch.erf(torch.tensor((dtol / 2) / (np.sqrt(2) * sigma_roll))))
    # influence: at K=32 anchors, M=64 perturbations of the HOR action blocks from the empirical block covariance
    with torch.no_grad():
        one_step = fwd(m, hist, blocks[:, :H]) - emb_t[idx + H * FS]; E = torch.cov(one_step.T) + 1e-6 * torch.eye(192, device=dev)
        E_H = E * (sigma_roll ** 2 / torch.trace(E))
        ablk = blocks[:, H:].reshape(len(idx), -1); cov_a = torch.cov(ablk.T) + 1e-6 * torch.eye(ablk.shape[1], device=dev); La = torch.linalg.cholesky(cov_a)
        memps = []
        for a_i in range(32):
            base = rollout(m, hist[a_i:a_i + 1].expand(64, -1, -1), blocks[a_i:a_i + 1].expand(64, -1, -1))
            delta = (torch.randn(64, ablk.shape[1], device=dev) @ La.T).reshape(64, HOR, -1)
            pert = blocks[a_i:a_i + 1].expand(64, -1, -1).clone(); pert[:, H:] = pert[:, H:] + delta
            disp = rollout(m, hist[a_i:a_i + 1].expand(64, -1, -1), pert) - base
            S = torch.cov(disp.T); Ei = torch.linalg.inv(E_H); lam = torch.linalg.eigvalsh((Ei @ S + (Ei @ S).T) / 2).clamp(min=0)
            memps.append(0.5 * torch.log1p(lam).sum().item())
    memp = float(np.mean(memps)); influence = min(memp / 82.0, 1.0)
    # sobriety: 64 anchors, goal = own future; mini-CEM (128 candidates, 4 iterations, top-16) in normalised action space
    gen = torch.Generator(device=dev).manual_seed(1); better = 0
    for a_i in range(64):
        goal = target[a_i]; hb = hist[a_i:a_i + 1].expand(128, -1, -1); base_blocks = blocks[a_i:a_i + 1]
        with torch.no_grad():
            J_star = float(((rollout(m, hist[a_i:a_i + 1], base_blocks) - goal) ** 2).sum())
            mean = base_blocks[0, H:].clone(); std = torch.ones_like(mean)
            for it in range(4):
                cand = torch.randn(128, HOR, mean.shape[1], device=dev, generator=gen) * std + mean
                full = base_blocks.expand(128, -1, -1).clone(); full[:, H:] = cand
                J = ((rollout(m, hb, full) - goal) ** 2).sum(1); elite = cand[torch.topk(J, 16, largest=False).indices]; mean, std = elite.mean(0), elite.std(0)
            full = base_blocks.clone(); full[:, H:] = mean.unsqueeze(0); J_cem = float(((rollout(m, hist[a_i:a_i + 1], full) - goal) ** 2).sum())
        better += int(J_star > J_cem)
    p_hat = better / 64; sobriety = 1 - p_hat
    res[mk] = dict(nmse_H5=nmse, sigma_roll=sigma_roll, veracity=veracity, m_emp=memp, influence=influence, p_hat=p_hat, sobriety=sobriety, VIScore=veracity * influence * sobriety)
    print(mk, json.dumps({k: round(v, 4) for k, v in res[mk].items()}), f"[{time.time()-t0:.0f}s]", flush=True)
json.dump(res, open("/kaggle/working/viscore.json", "w"), indent=1)
'''
open("/kaggle/working/vs_run.py", "w").write(code)
run("python /kaggle/working/vs_run.py 2>&1 | grep -v Warning")
run("rm -f /kaggle/working/vs_run.py; cat /kaggle/working/viscore.json")
print(f"\nTOTAL {time.time()-t0:.0f}s")
