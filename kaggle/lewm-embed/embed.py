import subprocess, time, os, glob, sys, json
t0 = time.time()
def run(cmd, timeout=11000):
    """fail fast: raises on a non-zero exit code, prints the tail of the output"""
    print(f"\n$ {cmd}  [t={time.time()-t0:.0f}s]", flush=True)
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    print((r.stdout + r.stderr)[-3000:], flush=True)
    if r.returncode != 0:
        raise SystemExit(f"STEP FAILED (exit {r.returncode}): {cmd}")
cands = glob.glob("/kaggle/input/**/tworoom.h5", recursive=True)
if not cands or os.path.getsize(cands[0]) < 1e9:
    raise SystemExit("DATASET MISSING")
H = "/kaggle/working/stablewm"; os.environ["STABLEWM_HOME"] = H; os.environ["WANDB_MODE"] = "disabled"; os.environ["H5"] = cands[0]
run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
run("pip install 'stable-worldmodel[train,env,format]==0.1.1' 2>&1 | tail -3")
run("python -c 'import hdf5plugin, h5py, stable_worldmodel, torch, numpy; print(\"versions:\", stable_worldmodel.__version__ if hasattr(stable_worldmodel, \"__version__\") else \"?\", torch.__version__, numpy.__version__, h5py.__version__)'")
run("pip freeze | grep -iE 'stable-worldmodel|stable-pretraining|^torch==|^numpy==|h5py|hdf5plugin|transformers' > /kaggle/working/environment.txt; python --version >> /kaggle/working/environment.txt; cat /kaggle/working/environment.txt")
code = r'''
import os, time, json, numpy as np, torch, h5py, hdf5plugin
import stable_worldmodel as swm
t0 = time.time()
torch.manual_seed(0)
model = swm.wm.utils.load_pretrained("quentinll/lewm-tworooms").cuda().eval()
f = h5py.File(os.environ["H5"], "r")
N = f["pixels"].shape[0]
print("frames", N, "pixels", f["pixels"].shape, f["pixels"].dtype, flush=True)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
def prep(px_u8):
    px = torch.from_numpy(px_u8).cuda().permute(0, 3, 1, 2).float() / 255.0
    px = (px - mean) / std
    if px.shape[-1] != 224:
        px = torch.nn.functional.interpolate(px, size=(224, 224), mode="bilinear", align_corners=False)
    return px
def embed(px, autocast):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        out = model.encoder(px, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        return model.projector(out).float()
# parity check against model.encode on 64 frames, in fp32 and in mixed precision
idx = np.sort(np.random.default_rng(0).choice(N, 64, replace=False))
px = prep(f["pixels"][idx])
with torch.no_grad():
    ref = model.encode({"pixels": px.unsqueeze(1)})["emb"][:, 0].float()
e32 = embed(px, False); e16 = embed(px, True)
par = dict(max_abs_fp32=float((e32 - ref).abs().max()), max_abs_mixed=float((e16 - ref).abs().max()),
           max_abs_mixed_then_fp16_storage=float((e16.half().float() - ref).abs().max()), ref_abs_mean=float(ref.abs().mean()), ref_std=float(ref.std()))
print("parity vs model.encode:", json.dumps(par), flush=True)
emb = np.zeros((N, 192), dtype=np.float16)
CH = 512
for i in range(0, N, CH):
    emb[i:i+CH] = embed(prep(f["pixels"][i:i+CH]), True).cpu().numpy().astype(np.float16)
    if (i // CH) % 200 == 0:
        el = time.time() - t0; print(f"{i}/{N} {el:.0f}s eta {el/(i+CH)*(N-i-CH)/60:.1f} min", flush=True)
assert np.isfinite(emb.astype(np.float32)).all(), "non-finite embeddings"
np.save("/kaggle/working/emb_tworoom.npy", emb)
for k in ["action", "ep_idx", "ep_len", "ep_offset", "pos_agent", "pos_target", "distance_to_target", "observation"]:
    np.save(f"/kaggle/working/{k}.npy", f[k][:])
e = emb.astype(np.float32)
json.dump(dict(parity=par, frames=int(N), emb_abs_mean=float(np.abs(e).mean()), emb_std=float(e.std()), per_dim_std_min=float(e.std(0).min()), per_dim_std_max=float(e.std(0).max()),
               model="quentinll/lewm-tworooms", precision="autocast fp16, stored float16", seconds=round(time.time()-t0)), open("/kaggle/working/embed_meta.json", "w"), indent=1)
print("done", round(time.time() - t0), "s", flush=True)
'''
open("/kaggle/working/embed_run.py", "w").write(code)
run("python /kaggle/working/embed_run.py 2>&1 | grep -v Warning")
run("ls -la /kaggle/working/*.npy /kaggle/working/*.json | awk '{print $5, $9}'; cat /kaggle/working/embed_meta.json; rm -f /kaggle/working/embed_run.py")
print(f"\nTOTAL {time.time()-t0:.0f}s")
