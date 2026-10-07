import subprocess, time, os, glob, sys, json
t0 = time.time()
def run(cmd, timeout=12000):
    print(f"\n$ {cmd}  [t={time.time()-t0:.0f}s]", flush=True)
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr; print(out[:1500] + ('\n...\n' + out[-5000:] if len(out) > 6500 else out[1500:]), flush=True)
    if r.returncode != 0: raise SystemExit(f"STEP FAILED (exit {r.returncode}): {cmd}")
cache = glob.glob("/kaggle/input/**/emb_tworoom.npy", recursive=True); rt = glob.glob("/kaggle/input/**/tworoom_retrained_v2/weights.pt", recursive=True)
if not cache or not rt: raise SystemExit(f"INPUTS MISSING cache={cache} retrained={rt}")
H = "/kaggle/working/stablewm"; os.environ.update(CACHE_DIR=os.path.dirname(cache[0]), RETRAINED=os.path.dirname(rt[0]), STABLEWM_HOME=H, WANDB_MODE="disabled", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", SDL_VIDEODRIVER="dummy")
run("nvidia-smi --query-gpu=name --format=csv,noheader; apt-get install -y -qq swig libegl1 libgl1 > /dev/null 2>&1; echo apt ok")
run("pip install 'stable-worldmodel[train,format]==0.1.1' 'transformers<5' hydra-core pygame pymunk shapely opencv-python-headless 2>&1 | tail -1")
run("python -c 'import hydra, stable_worldmodel, pygame, pymunk, shapely, cv2; print(\"imports ok\")'")
code = r'''
import os, json, time, numpy as np, torch
from scipy import stats
import stable_worldmodel as swm
t0 = time.time(); dev = "cuda"; H, FS = 3, 5; WARM, BLOCKS, M = 15, 5, 30
NS, NSTEPS, TOPK = 300, 30, 30          # their CEM: 300 samples, 30 iterations, 30 elites, var_scale 1, plan = final mean
D = os.environ["CACHE_DIR"]; act = np.nan_to_num(np.load(f"{D}/action.npy").astype(np.float32)); mu, sd = act.mean(0), act.std(0) + 1e-8
mu_t, sd_t = torch.as_tensor(mu, device=dev), torch.as_tensor(sd, device=dev)
released = swm.wm.utils.load_pretrained("quentinll/lewm-tworooms").to(dev).eval(); retrained = swm.wm.utils.load_pretrained(os.environ["RETRAINED"]).to(dev).eval()
# the bilinear operator, refitted here exactly as in experiment 18 (ridge 1e-2 x N on the training windows)
emb = np.load(f"{D}/emb_tworoom.npy").astype(np.float32); ep_len = np.load(f"{D}/ep_len.npy"); ep_off = np.load(f"{D}/ep_offset.npy")
rng0 = np.random.default_rng(0); eps = np.arange(len(ep_len)); test_eps = set(rng0.choice(eps, size=len(eps) // 10, replace=False).tolist())
starts = np.array([i for e in eps if e not in test_eps for i in range(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e]) - 15)])
E = torch.from_numpy(emb).to(dev); At = torch.from_numpy((act - mu) / sd).to(dev)
def feats(idx):
    idx = torch.as_tensor(idx, device=dev); z2, z1, z0 = E[idx], E[idx + 5], E[idx + 10]; a = torch.cat([At[idx + 10 + j] for j in range(5)], 1)
    return torch.cat([z0, z1, z2, a, (z0.unsqueeze(2) * a.unsqueeze(1)).reshape(len(idx), -1), torch.ones(len(idx), 1, device=dev)], 1), E[idx + 15]
dX = 2507; XtX = torch.zeros(dX, dX, device=dev, dtype=torch.float64); XtY = torch.zeros(dX, 192, device=dev, dtype=torch.float64)
for i in range(0, len(starts), 8192):
    X, Y = feats(starts[i:i+8192]); X, Y = X.double(), Y.double(); XtX += X.T @ X; XtY += X.T @ Y
Wop = torch.linalg.solve(XtX + 1e-2 * len(starts) * torch.eye(dX, device=dev, dtype=torch.float64), XtY).float()
class Operator(torch.nn.Module):
    def __init__(self, W): super().__init__(); self.register_buffer("W", W)
    def forward(self, emb_, act_):
        B, T, Dd = emb_.shape; outs = []
        for t in range(T):
            z0 = emb_[:, t]; z1 = emb_[:, t - 1] if t >= 1 else emb_[:, t]; z2 = emb_[:, t - 2] if t >= 2 else z1; a = act_[:, t]
            outs.append(torch.cat([z0, z1, z2, a, (z0.unsqueeze(2) * a.unsqueeze(1)).reshape(B, -1), torch.ones(B, 1, device=emb_.device, dtype=emb_.dtype)], 1) @ self.W.to(emb_.dtype))
        return torch.stack(outs, 1)
import copy
operator = copy.deepcopy(released); operator.predictor = Operator(Wop).to(dev); operator.action_encoder = torch.nn.Identity(); operator.pred_proj = torch.nn.Identity()
models = {"released": released, "retrained_v2": retrained, "bilinear": operator}
print("operator fitted", flush=True)
# bilinear error at contexts of 1, 2 and 3 real frames, on held-out windows (the planner's first calls)
te = np.array([i for e in eps if e in test_eps for i in range(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e]) - 15)])
def ctx_err(m, L, bs=2048):
    se, n, tg = 0.0, 0, []
    with torch.no_grad():
        for i in range(0, len(te), bs):
            idx = torch.as_tensor(te[i:i+bs], device=dev); h = torch.stack([E[idx + k * 5] for k in range(3)], 1); a = torch.stack([torch.cat([At[idx + k * 5 + j] for j in range(5)], 1) for k in range(3)], 1)
            y = E[idx + L * 5]; p = m.predictor(h[:, :L], a[:, :L])[:, -1]; se += ((p - y) ** 2).sum().item(); n += y.numel(); tg.append(y)
    return se / n / torch.cat(tg).var(0).mean().item()
ctx_bil = {f"context_{L}": round(ctx_err(operator, L), 4) for L in (1, 2, 3)}; print("bilinear context errors", ctx_bil, flush=True)
mean_ = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev); std_ = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
def encode(px_u8):
    px = torch.from_numpy(np.asarray(px_u8)).to(dev).permute(0, 3, 1, 2).float() / 255.0; px = (px - mean_) / std_
    with torch.no_grad(): return released.encode({"pixels": px.unsqueeze(1)})["emb"][:, 0].float()
def fwd(m, h, a):
    p = m.predictor(h, m.action_encoder(a)); B, T, Dd = p.shape; return m.pred_proj(p.reshape(B * T, Dd)).reshape(B, T, -1)[:, -1]
def predict_final(m, hist, A):
    h = hist.clone(); n_pred = (WARM + BLOCKS * FS - (H - 1) * FS) // FS
    with torch.no_grad():
        for k in range(n_pred):
            base = k * FS; a = torch.stack([A[:, base + kk * FS: base + kk * FS + FS].reshape(A.shape[0], -1) for kk in range(H)], 1)
            p = fwd(m, h, a); h = torch.cat([h[:, 1:], p.unsqueeze(1)], 1)
    return p
def cem(m, hist1, warm_n, goal, gen):
    """their CEM in the normalised action space; returns the final mean plan, its predicted cost unclipped, its predicted cost after clipping to the env bounds, and the saturation fraction"""
    mean = torch.zeros(BLOCKS * FS, 2, device=dev); std = torch.ones(BLOCKS * FS, 2, device=dev)
    hist = hist1.expand(NS, -1, -1); warm = warm_n.expand(NS, -1, -1)
    for it in range(NSTEPS):
        cand = torch.randn(NS, BLOCKS * FS, 2, device=dev, generator=gen) * std + mean; cand[0] = mean
        pf = predict_final(m, hist, torch.cat([warm, cand], 1)); cost = ((pf - goal) ** 2).sum(1)
        elite = cand[torch.topk(cost, TOPK, largest=False).indices]; mean, std = elite.mean(0), elite.std(0)
    pf = predict_final(m, hist1, torch.cat([warm_n, mean.unsqueeze(0)], 1)); pc_unclipped = float(((pf - goal) ** 2).sum())
    raw = mean * sd_t + mu_t; sat = float((raw.abs() > 1.0).float().mean()); clipped_n = (raw.clamp(-1, 1) - mu_t) / sd_t
    pf2 = predict_final(m, hist1, torch.cat([warm_n, clipped_n.unsqueeze(0)], 1)); pc_clipped = float(((pf2 - goal) ** 2).sum())
    return mean, (pc_unclipped, pc_clipped, sat)
world = swm.World("swm/TwoRoom-v1", num_envs=4, image_shape=(224, 224), max_episode_steps=200)
rngW, rngC = np.random.default_rng(10), np.random.default_rng(11); gen = torch.Generator(device=dev).manual_seed(0)
rows = {mk: dict(pred=[], pred_clipped=[], saturation=[], true=[]) for mk in models}; goal_true = []
def grab(px): return px.reshape(-1, *px.shape[-3:]) if px.ndim == 5 else px
for m_ in range(M):
    world.reset(seed=[4000 + m_] * 4); px = grab(world.infos["pixels"]); frames = {}
    warm = rngW.uniform(-1, 1, size=(WARM, 2)).astype(np.float32)
    for t in range(WARM):
        if t in (0, 5, 10): frames[t] = px[0].copy()
        _, _, _, _, infos = world.envs.step(np.repeat(warm[t][None], 4, 0)); px = grab(infos["pixels"])
    hist1 = torch.stack([encode(frames[t][None]) for t in (0, 5, 10)], 1)           # (1, 3, 192), identical for the three copies
    warm_n = torch.as_tensor((warm - mu) / sd, device=dev).unsqueeze(0)
    # the goal: a random plan executed in env 0; first a random plan evaluates both models' plans on the same start
    goal_plan = rngC.uniform(-1, 1, size=(BLOCKS * FS, 2)).astype(np.float32)
    plans = {"released": None, "retrained": None}
    # run CEM for both models against a provisional goal: the true outcome of the random plan (executed first in env 0)
    px0 = px.copy(); acts = np.zeros((4, BLOCKS * FS, 2), np.float32); acts[0] = goal_plan
    # we need the goal before planning: execute env 0 alone with the random plan while env 1 and 2 wait (masked)
    for t in range(BLOCKS * FS):
        _, _, _, _, infos = world.envs.step(acts[:, t], mask=np.array([True, False, False, False])); px = grab(infos["pixels"])
    goal = encode(px[0][None])[0]; goal_true.append(float(((goal - goal) ** 2).sum()))
    for mi, (mk, m) in enumerate(models.items()):
        plan_n, (pc, pcc, sat) = cem(m, hist1, warm_n, goal, gen); plan = np.clip((plan_n.cpu().numpy() * sd + mu), -1, 1).astype(np.float32); acts[mi + 1] = plan; rows[mk]["pred"].append(pc); rows[mk]["pred_clipped"].append(pcc); rows[mk]["saturation"].append(sat)
    for t in range(BLOCKS * FS):
        _, _, _, _, infos = world.envs.step(acts[:, t], mask=np.array([False, True, True, True])); px = grab(infos["pixels"])
    fin = encode(px[1:4])
    for mi, mk in enumerate(models): rows[mk]["true"].append(float(((fin[mi] - goal) ** 2).sum()))
    if m_ % 10 == 9: print(f"start {m_+1}/{M} [{time.time()-t0:.0f}s]", flush=True)
res = {}
for mk, r in rows.items():
    pred, predc, true, sat = np.array(r["pred"]), np.array(r["pred_clipped"]), np.array(r["true"]), np.array(r["saturation"])
    res[mk] = dict(predicted_cost_unclipped=float(pred.mean()), predicted_cost_after_clipping=float(predc.mean()), true_cost=float(true.mean()), median_ratio_true_over_pred_unclipped=float(np.median(true / np.maximum(pred, 1e-6))), median_ratio_true_over_pred_clipped=float(np.median(true / np.maximum(predc, 1e-6))), saturation_fraction=float(sat.mean()))
res["bilinear_context_errors"] = ctx_bil
d = np.array(rows["bilinear"]["true"]) - np.array(rows["released"]["true"]); res["true_cost_diff_bilinear_minus_released"] = dict(mean=float(d.mean()), positive=int((d > 0).sum()), p_wilcoxon=float(stats.wilcoxon(d).pvalue))
res["per_start"] = {mk: {k: [float(x) for x in v] for k, v in r.items()} for mk, r in rows.items()}
res["starts"] = M; res["cem"] = dict(samples=NS, iterations=NSTEPS, topk=TOPK)
print(json.dumps(res, indent=1)); json.dump(res, open("/kaggle/working/bounding.json", "w"), indent=1)
'''
open("/kaggle/working/ex_run.py", "w").write(code)
run("python /kaggle/working/ex_run.py 2>&1 | grep -v Warning")
run("rm -f /kaggle/working/ex_run.py; cat /kaggle/working/bounding.json")
print(f"\nTOTAL {time.time()-t0:.0f}s")
