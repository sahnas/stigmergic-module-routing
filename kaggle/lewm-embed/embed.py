import subprocess, time, os, glob, sys
t0 = time.time()
def sh(cmd, timeout=3000):
    print(f"\n$ {cmd}  [t={time.time()-t0:.0f}s]", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    print((r.stdout + r.stderr)[-4000:], flush=True)
cands = glob.glob("/kaggle/input/**/tworoom.h5", recursive=True)
print("tworoom.h5 :", cands, flush=True)
if not cands: print("DONNEES ABSENTES"); sys.exit(0)
H = "/kaggle/working/stablewm"; os.environ["STABLEWM_HOME"] = H; os.environ["WANDB_MODE"] = "disabled"
os.makedirs(f"{H}/datasets", exist_ok=True)
sh("nvidia-smi --query-gpu=name --format=csv,noheader; pip install -q 'stable-worldmodel[train,env]' hdf5plugin 2>&1 | grep -ciE 'error'")
code = r'''
import os, time, numpy as np, torch, h5py, hdf5plugin
import stable_worldmodel as swm
t0 = time.time()
model = swm.wm.utils.load_pretrained("quentinll/lewm-tworooms").cuda().eval()
f = h5py.File(os.environ["H5"], "r")
N = f["pixels"].shape[0]
print("frames", N, "pixels", f["pixels"].shape, f["pixels"].dtype, "action", f["action"].shape, flush=True)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
emb = np.zeros((N, 192), dtype=np.float16)
CH = 512
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    for i in range(0, N, CH):
        px = torch.from_numpy(f["pixels"][i:i+CH]).cuda().permute(0, 3, 1, 2).float() / 255.0
        px = (px - mean) / std
        if px.shape[-1] != 224:
            px = torch.nn.functional.interpolate(px, size=(224, 224), mode="bilinear", align_corners=False)
        out = model.encoder(px, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        e = model.projector(out)
        emb[i:i+CH] = e.float().cpu().numpy().astype(np.float16)
        if (i // CH) % 100 == 0:
            el = time.time() - t0; print(f"{i}/{N} {el:.0f}s eta {el/(i+CH)*(N-i-CH)/60:.1f} min", flush=True)
np.save("/kaggle/working/emb_tworoom.npy", emb)
for k in ["action", "ep_idx", "ep_len", "ep_offset", "pos_agent", "pos_target", "distance_to_target", "observation"]:
    np.save(f"/kaggle/working/{k}.npy", f[k][:])
e = emb.astype(np.float32)
print("emb stats: mean|.| %.3f std %.3f per-dim std min %.3f max %.3f nan %d" % (np.abs(e).mean(), e.std(), e.std(0).min(), e.std(0).max(), np.isnan(e).sum()))
print("done", round(time.time() - t0), "s", flush=True)
'''
open("/kaggle/working/embed_run.py", "w").write(code)
os.environ["H5"] = cands[0]
sh("python /kaggle/working/embed_run.py 2>&1 | grep -v arn | tail -40", timeout=11000)
sh("ls -la /kaggle/working/*.npy | awk '{print $5, $9}'; rm -f /kaggle/working/embed_run.py")
print(f"\nTOTAL {time.time()-t0:.0f}s")
