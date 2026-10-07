import subprocess, time, os, sys, json
t0 = time.time()
def run(cmd, timeout=40000):
    print(f"\n$ {cmd}  [t={time.time()-t0:.0f}s]", flush=True)
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr; print(out[:1500] + ('\n...\n' + out[-5000:] if len(out) > 6500 else out[1500:]), flush=True)
    if r.returncode != 0: raise SystemExit(f"STEP FAILED (exit {r.returncode}): {cmd}")
H = "/kaggle/working/stablewm"; os.environ.update(STABLEWM_HOME=H, WANDB_MODE="disabled")
run("nvidia-smi --query-gpu=name --format=csv,noheader; df -h /kaggle/working | tail -1; apt-get install -y -qq zstd > /dev/null 2>&1; which zstd")
run("pip install 'stable-worldmodel[train,format]==0.1.1' 'transformers<5' hydra-core 2>&1 | tail -1; python -c 'import hydra, stable_worldmodel, h5py, hdf5plugin; print(\"imports ok\")'")
# the dataset archive published with the model, revision recorded in the preregistration
dl = r"""
from huggingface_hub import hf_hub_download
import os, time
p = hf_hub_download("quentinll/lewm-pusht", "pusht_expert_train.h5.zst", repo_type="dataset", revision="655cd446b992", local_dir="/kaggle/working/dl")
print("archive", p, round(os.path.getsize(p) / 1e9, 2), "GB", flush=True)
"""
open("/kaggle/working/dl.py", "w").write(dl); run("python /kaggle/working/dl.py")
os.makedirs(f"{H}/datasets", exist_ok=True)
run(f"zstd -d -q /kaggle/working/dl/pusht_expert_train.h5.zst -o {H}/datasets/pusht_expert_train.h5 && rm -rf /kaggle/working/dl && ls -la {H}/datasets && df -h /kaggle/working | tail -1", timeout=3600)
code = r'''
import os, json, time, numpy as np, torch, h5py, hdf5plugin
import stable_worldmodel as swm
t0 = time.time(); dev = "cuda"
# dataset through the library's own downloader, as eval.py would resolve it
H = os.environ["STABLEWM_HOME"]; cands = [p for p in [f"{H}/pusht_expert_train.h5", f"{H}/datasets/pusht_expert_train.h5"] if os.path.exists(p)]
if not cands:
    import glob; cands = glob.glob(f"{H}/**/pusht_expert_train*.h5", recursive=True)
assert cands, "dataset not found after download"
path = cands[0]; f = h5py.File(path, "r"); print("dataset", path, round(os.path.getsize(path) / 1e9, 2), "GB; keys", list(f.keys()), flush=True)
N = f["pixels"].shape[0]; print("frames", N, "pixels", f["pixels"].shape, "action", f["action"].shape, flush=True)
for k in ("action", "ep_idx", "ep_len", "ep_offset", "state", "proprio"):
    if k in f: np.save(f"/kaggle/working/{k}.npy", f[k][:])
model = swm.wm.utils.load_pretrained("quentinll/lewm-pusht").to(dev).eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
def encode(px_u8):
    px = torch.from_numpy(np.asarray(px_u8)).to(dev).permute(0, 3, 1, 2).float() / 255.0
    if px.shape[-1] != 224: px = torch.nn.functional.interpolate(px, size=(224, 224), mode="bilinear", align_corners=False)
    px = (px - mean) / std
    with torch.no_grad(): return model.encode({"pixels": px.unsqueeze(1)})["emb"][:, 0].float().cpu().numpy()
emb = np.zeros((N, 192), np.float32); bs = 256
for i in range(0, N, bs):
    emb[i:i+bs] = encode(f["pixels"][i:i+bs])
    if (i // bs) % 400 == 0: print(f"{i}/{N} [{time.time()-t0:.0f}s]", flush=True)
np.save("/kaggle/working/emb_pusht.npy", emb)
# parity on 64 frames against the model's own encode on the same tensors
chk = encode(f["pixels"][1000:1064]); print("parity (same function, sanity)", float(np.abs(chk - emb[1000:1064]).max()), flush=True)
json.dump(dict(dataset=path, frames=int(N), pixels_shape=list(f["pixels"].shape), action_shape=list(f["action"].shape), episodes=int(f["ep_len"].shape[0]) if "ep_len" in f else None, model_revision="22b330c28c27ead4bfd1888615af1340e3fe9052", seconds=round(time.time() - t0)), open("/kaggle/working/pusht_cache_meta.json", "w"), indent=1)
'''
open("/kaggle/working/cache_run.py", "w").write(code)
run("python /kaggle/working/cache_run.py 2>&1 | grep -v Warning")
run("rm -f /kaggle/working/cache_run.py; ls -la /kaggle/working/*.npy /kaggle/working/*.json; cat /kaggle/working/pusht_cache_meta.json")
print(f"\nTOTAL {time.time()-t0:.0f}s")
