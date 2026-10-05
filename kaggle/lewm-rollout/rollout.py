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
import stable_worldmodel as swm
t0 = time.time(); D = os.environ["CACHE_DIR"]; dev = "cuda"; H, FS, K = 3, 5, 10
emb = np.load(f"{D}/emb_tworoom.npy").astype(np.float32); act = np.nan_to_num(np.load(f"{D}/action.npy").astype(np.float32))
ep_len = np.load(f"{D}/ep_len.npy"); ep_off = np.load(f"{D}/ep_offset.npy")
rng = np.random.default_rng(0); eps = np.arange(len(ep_len)); test_eps = sorted(set(rng.choice(eps, size=len(eps) // 10, replace=False).tolist()))
mu, sd = act.mean(0), act.std(0) + 1e-8
released = swm.wm.utils.load_pretrained("quentinll/lewm-tworooms").to(dev).eval(); retrained = swm.wm.utils.load_pretrained(os.environ["RETRAINED"]).to(dev).eval()
models = {"released": released, "retrained": retrained}
def fwd(m, h, a):
    p = m.predictor(h, m.action_encoder(a)); B, T, Dd = p.shape; return m.pred_proj(p.reshape(B * T, Dd)).reshape(B, T, -1)[:, -1]
def rollout_errors(m, E, A, starts, bs=1024):
    """open-loop: from the true history at the start, predict K steps of 5 frames, feeding predictions back; relative error per horizon"""
    E = torch.as_tensor(E, device=dev); A = torch.as_tensor(A, device=dev); se = np.zeros(K); n = 0; var = None; tg = [[] for _ in range(K)]
    with torch.no_grad():
        for i in range(0, len(starts), bs):
            idx = torch.as_tensor(starts[i:i+bs], device=dev)
            hist = torch.stack([E[idx + k * FS] for k in range(H)], 1)
            for k in range(K):
                base = idx + k * FS
                a = torch.stack([torch.cat([A[base + kk * FS + j] for j in range(FS)], 1) for kk in range(H)], 1)
                p = fwd(m, hist, a); y = E[idx + (H + k) * FS]
                se[k] += ((p - y) ** 2).sum().item(); tg[k].append(y)
                hist = torch.cat([hist[:, 1:], p.unsqueeze(1)], 1)
            n += len(idx)
    out = []
    for k in range(K):
        y = torch.cat(tg[k]); out.append(se[k] / (len(y) * y.shape[1]) / y.var(0).mean().item())
    return out
res = {}
# in-distribution: expert test episodes, starts that allow K rollout steps
te = np.array([i for e in test_eps for i in range(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e]) - (H + K) * FS)])[:20000]
A = (act - mu) / sd
res["expert"] = {mk: rollout_errors(m, emb, A, te) for mk, m in models.items()}
print("expert rollouts (relative error at horizons 1..10 x 5 frames):", {k: [round(v, 3) for v in vv] for k, vv in res["expert"].items()}, flush=True)
# counterfactual: random actions in the environment
world = swm.World("swm/TwoRoom-v1", num_envs=20, image_shape=(224, 224), max_episode_steps=200)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
def encode(px_u8):
    px = torch.from_numpy(np.asarray(px_u8)).to(dev).permute(0, 3, 1, 2).float() / 255.0; px = (px - mean) / std
    with torch.no_grad(): return released.encode({"pixels": px.unsqueeze(1)})["emb"][:, 0].float().cpu().numpy()
rngA = np.random.default_rng(1); EP_STEPS, N_ROUNDS = 100, 5; frames, acts = [], []
for rd in range(N_ROUNDS):
    world.reset(seed=1000 + rd); infos = world.infos; px = infos["pixels"]; px = px.reshape(-1, *px.shape[-3:]) if px.ndim == 5 else px
    cur = [[] for _ in range(world.num_envs)]; cura = [[] for _ in range(world.num_envs)]
    for t in range(EP_STEPS):
        a = rngA.uniform(-1, 1, size=(world.num_envs, 2)).astype(np.float32)
        for i in range(world.num_envs): cur[i].append(px[i]); cura[i].append(a[i])
        _, _, term, trunc, infos = world.envs.step(a); px = infos["pixels"]; px = px.reshape(-1, *px.shape[-3:]) if px.ndim == 5 else px
    for i in range(world.num_envs): frames.append(np.stack(cur[i])); acts.append(np.stack(cura[i]))
F = np.concatenate(frames); A_raw = np.concatenate(acts); lens = [len(f) for f in frames]; offs = np.cumsum([0] + lens[:-1])
E = np.concatenate([encode(F[i:i+256]) for i in range(0, len(F), 256)])
starts = np.array([o + i for o, L in zip(offs, lens) for i in range(L - (H + K) * FS)])
res["random_actions"] = {mk: rollout_errors(m, E, (A_raw - mu) / sd, starts) for mk, m in models.items()}
res["windows"] = dict(expert=int(len(te)), random=int(len(starts))); res["seconds"] = round(time.time() - t0)
print("random-action rollouts:", {k: [round(v, 3) for v in vv] for k, vv in res["random_actions"].items()}, flush=True)
json.dump(res, open("/kaggle/working/rollout.json", "w"), indent=1)
'''
open("/kaggle/working/ro_run.py", "w").write(code)
run("python /kaggle/working/ro_run.py 2>&1 | grep -v Warning")
run("rm -f /kaggle/working/ro_run.py; cat /kaggle/working/rollout.json | head -40")
print(f"\nTOTAL {time.time()-t0:.0f}s")
