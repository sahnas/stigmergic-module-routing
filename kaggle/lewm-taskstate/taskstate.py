import subprocess, time, os, glob, sys, json
t0 = time.time()
def run(cmd, timeout=20000):
    print(f"\n$ {cmd}  [t={time.time()-t0:.0f}s]", flush=True)
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr; print(out[:1500] + ('\n...\n' + out[-6000:] if len(out) > 7500 else out[1500:]), flush=True)
    if r.returncode != 0: raise SystemExit(f"STEP FAILED (exit {r.returncode}): {cmd}")
need = {"TR_CACHE": "emb_tworoom.npy", "TR_V2": "tworoom_retrained_v2/weights.pt", "PT_CACHE": "emb_pusht.npy", "PT_BIL": "pusht_bilinear.pt", "PT_V2": "pusht_retrained_v2/weights.pt"}
for k, pat in need.items():
    g = glob.glob(f"/kaggle/input/**/{pat}", recursive=True)
    if not g: raise SystemExit(f"INPUT MISSING {pat}")
    os.environ[k] = os.path.dirname(g[0]) if pat.endswith(".npy") or pat.endswith("weights.pt") else g[0]
os.environ.update(STABLEWM_HOME="/kaggle/working/stablewm", WANDB_MODE="disabled")
run("pip install 'stable-worldmodel[train,format]==0.1.1' 'transformers<5' hydra-core 2>&1 | tail -1; python -c 'import hydra, stable_worldmodel; print(\"imports ok\")'")
code = r'''
import os, json, time, numpy as np, torch
import stable_worldmodel as swm
t0 = time.time(); dev = "cuda"; H, FS, HOR = 3, 5, 5
TASKS = {
  "tworoom": dict(cache=os.environ["TR_CACHE"], hf="quentinll/lewm-tworooms", hf_rev="77adaae0bc31deab21c93740d1f8bb947cd0bdec", v2=os.environ["TR_V2"], emb_file="emb_tworoom.npy", state_fn=lambda c: np.load(f"{c}/pos_agent.npy").astype(np.float32), state_names=["agent_x", "agent_y"], bil=None),
  "pusht": dict(cache=os.environ["PT_CACHE"], hf="quentinll/lewm-pusht", hf_rev="22b330c28c27ead4bfd1888615af1340e3fe9052", v2=os.environ["PT_V2"], emb_file="emb_pusht.npy", state_fn=None, state_names=["agent_x", "agent_y", "block_x", "block_y", "block_cos", "block_sin"], bil=os.environ["PT_BIL"]),
}
def pusht_state(c):
    s = np.load(f"{c}/state.npy").astype(np.float32)          # [agent_x, agent_y, block_x, block_y, block_angle, vel_x, vel_y]
    return np.stack([s[:, 0], s[:, 1], s[:, 2], s[:, 3], np.cos(s[:, 4]), np.sin(s[:, 4])], 1)
TASKS["pusht"]["state_fn"] = pusht_state
def fwd(m, h, a):
    p = m.predictor(h, m.action_encoder(a)); B, T, Dd = p.shape; return m.pred_proj(p.reshape(B * T, Dd)).reshape(B, T, -1)[:, -1]
class Bilinear:
    def __init__(self, W): self.W = W.to(dev)
    def __call__(self, h, a):
        B = h.shape[0]; z0 = h[:, -1]; z1 = h[:, -2] if h.shape[1] >= 2 else z0; z2 = h[:, -3] if h.shape[1] >= 3 else z1; ab = a[:, -1]
        return torch.cat([z0, z1, z2, ab, (z0.unsqueeze(2) * ab.unsqueeze(1)).reshape(B, -1), torch.ones(B, 1, device=dev)], 1) @ self.W
out = {}
for task, cfg in TASKS.items():
    c = cfg["cache"]; emb = np.load(f"{c}/{cfg['emb_file']}", mmap_mode="r"); act = np.nan_to_num(np.load(f"{c}/action.npy").astype(np.float32))
    ep_len = np.load(f"{c}/ep_len.npy"); ep_off = np.load(f"{c}/ep_offset.npy"); state = cfg["state_fn"](c); N = len(ep_len)
    rng = np.random.default_rng(0); eps = np.arange(N); test_eps = set(rng.choice(eps, size=N // 10, replace=False).tolist())
    mu, sd = act.mean(0), act.std(0) + 1e-8; A_t = torch.from_numpy((act - mu) / sd).to(dev)
    span = (H - 1 + HOR) * FS
    tr_frames = np.concatenate([np.arange(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e])) for e in eps if e not in test_eps])
    te_starts = np.array([i for e in eps if e in test_eps for i in range(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e]) - span)])
    rs = np.random.default_rng(1); tr_sub = rs.choice(tr_frames, size=min(300000, len(tr_frames)), replace=False); te_sub = rs.choice(te_starts, size=min(20000, len(te_starts)), replace=False)
    E = torch.from_numpy(np.asarray(emb)).to(dev); S = torch.from_numpy(state).to(dev)
    # linear probe embedding -> task state, ridge on training frames
    X = torch.cat([E[torch.as_tensor(tr_sub, device=dev)], torch.ones(len(tr_sub), 1, device=dev)], 1).double(); Y = S[torch.as_tensor(tr_sub, device=dev)].double()
    P = torch.linalg.solve(X.T @ X + 1e-3 * len(tr_sub) * torch.eye(193, device=dev, dtype=torch.float64), X.T @ Y).float()
    probe = lambda z: torch.cat([z, torch.ones(len(z), 1, device=dev)], 1) @ P
    te_frames = torch.as_tensor(np.concatenate([np.arange(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e])) for e in eps if e in test_eps]), device=dev)
    with torch.no_grad():
        yhat = probe(E[te_frames]); ytrue = S[te_frames]; var_s = ytrue.var(0); r2 = (1 - ((yhat - ytrue) ** 2).mean(0) / var_s).cpu().numpy()
    print(task, "probe held-out R2 per coordinate", dict(zip(cfg["state_names"], np.round(r2, 3))), flush=True)
    released = swm.wm.utils.load_pretrained(cfg["hf"]).to(dev).eval(); v2 = swm.wm.utils.load_pretrained(cfg["v2"]).to(dev).eval()
    if cfg["bil"] is None:   # Two-rooms: refit as in experiment 18 (ridge 1e-2 x N on training windows of three frames + 5-frame target)
        starts_tr = np.array([i for e in eps if e not in test_eps for i in range(int(ep_off[e]), int(ep_off[e]) + int(ep_len[e]) - H * FS)])
        dX = 3 * 192 + 10 + 1920 + 1; XtX = torch.zeros(dX, dX, device=dev, dtype=torch.float64); XtY = torch.zeros(dX, 192, device=dev, dtype=torch.float64)
        def feats(idx):
            idx = torch.as_tensor(idx, device=dev); z2, z1, z0 = E[idx], E[idx + FS], E[idx + 2 * FS]; a = torch.cat([A_t[idx + 2 * FS + j] for j in range(FS)], 1)
            return torch.cat([z0, z1, z2, a, (z0.unsqueeze(2) * a.unsqueeze(1)).reshape(len(idx), -1), torch.ones(len(idx), 1, device=dev)], 1), E[idx + 3 * FS]
        for i in range(0, len(starts_tr), 8192):
            Xb, Yb = feats(starts_tr[i:i+8192]); Xb, Yb = Xb.double(), Yb.double(); XtX += Xb.T @ Xb; XtY += Xb.T @ Yb
        Wb = torch.linalg.solve(XtX + 1e-2 * len(starts_tr) * torch.eye(dX, device=dev, dtype=torch.float64), XtY).float()
    else:
        Wb = torch.load(cfg["bil"], map_location="cpu")["W"]
    models = {"released": lambda h, a: fwd(released, h, a), "retrained_v2": lambda h, a: fwd(v2, h, a), "bilinear": Bilinear(Wb)}
    idx = torch.as_tensor(te_sub, device=dev)
    def blocks_at(k): return torch.cat([A_t[idx + k * FS + j] for j in range(FS)], 1)
    hist0 = torch.stack([E[idx + k * FS] for k in range(H)], 1)
    res = {"probe_r2": dict(zip(cfg["state_names"], [float(v) for v in r2])), "n_windows": int(len(te_sub)), "predictors": {}}
    with torch.no_grad():
        for name, fn in models.items():
            h = hist0.clone(); rec = {}
            for k in range(HOR):
                a = torch.stack([blocks_at(k + kk) for kk in range(H)], 1); p = fn(h, a); h = torch.cat([h[:, 1:], p.unsqueeze(1)], 1)
                tgt_idx = idx + (H + k) * FS; ztrue = E[tgt_idx]; strue = S[tgt_idx]
                lat = float(((p - ztrue) ** 2).sum(1).mean() / ztrue.var(0).sum()); sp = probe(p); st = ((sp - strue) ** 2).mean(0) / var_s
                rec[f"h{k+1}"] = dict(latent_nmse=lat, state_nmse_mean=float(st.mean()), state_nmse=dict(zip(cfg["state_names"], [float(v) for v in st])))
            res["predictors"][name] = rec
            print(task, name, "h1 latent", round(rec["h1"]["latent_nmse"], 4), "state", round(rec["h1"]["state_nmse_mean"], 4), "| h5 latent", round(rec["h5"]["latent_nmse"], 4), "state", round(rec["h5"]["state_nmse_mean"], 4), {k: round(v, 3) for k, v in rec["h5"]["state_nmse"].items()}, flush=True)
    out[task] = res
json.dump(out, open("/kaggle/working/taskstate.json", "w"), indent=1); print("done", round(time.time() - t0), "s")
'''
open("/kaggle/working/ts_run.py", "w").write(code)
run("python /kaggle/working/ts_run.py 2>&1 | grep -v Warning")
run("rm -f /kaggle/working/ts_run.py; ls -la /kaggle/working/taskstate.json")
print(f"\nTOTAL {time.time()-t0:.0f}s")
