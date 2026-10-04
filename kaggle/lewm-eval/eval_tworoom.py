import subprocess, time, os, glob, sys
t0 = time.time()
def sh(cmd, timeout=3000):
    print(f"\n$ {cmd}  [t={time.time()-t0:.0f}s]", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    print((r.stdout + r.stderr)[-7000:], flush=True)
    return r.returncode
cands = glob.glob("/kaggle/input/**/tworoom.h5", recursive=True)
print("tworoom.h5 :", cands, flush=True)
if not cands: print("DONNEES ABSENTES"); sys.exit(0)
H = "/kaggle/working/stablewm"
os.environ["WANDB_MODE"] = "disabled"; os.environ["STABLEWM_HOME"] = H; os.environ["MUJOCO_GL"] = "egl"; os.environ["PYOPENGL_PLATFORM"] = "egl"
os.makedirs(f"{H}/datasets", exist_ok=True); os.symlink(cands[0], f"{H}/datasets/tworoom.h5")
sh("nvidia-smi --query-gpu=name --format=csv,noheader")
sh("apt-get install -y -qq swig zstd libegl1 libgl1 > /dev/null 2>&1; git clone --depth 1 https://github.com/lucas-maes/le-wm.git /kaggle/tmp/le-wm > /dev/null 2>&1; pip install -q 'stable-worldmodel[train,env]' hdf5plugin 2>&1 | grep -ciE 'error'; pip show stable-worldmodel | head -2", timeout=2400)
sh(f"hf download quentinll/lewm-tworooms --local-dir {H}/hf_tworoom 2>&1 | tail -1; ls -la {H}/hf_tworoom; cat {H}/hf_tworoom/config.json | head -60")
conv = r'''
import json, torch, hydra
from pathlib import Path
import stable_worldmodel as swm
src = Path(swm.data.utils.get_cache_dir(), "hf_tworoom")
out = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), "tworoom", "lewm_object.ckpt")   # policy names resolve under <STABLEWM_HOME>/checkpoints/ (policy.py: _load_model_with_attribute)
cfg = json.loads((src / "config.json").read_text())
model = hydra.utils.instantiate(cfg)   # the HF config.json is a Hydra config targeting stable_worldmodel.wm.lewm.LeWM
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
print("type", type(model).__name__, "| load:", model.load_state_dict(sd, strict=True))
model.eval()
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
print("converti ->", out, round(out.stat().st_size/1e6,1), "MB | params", round(sum(p.numel() for p in model.parameters())/1e6,2), "M")
'''
open("/kaggle/tmp/le-wm/convert_tworoom.py", "w").write(conv)
sh("cd /kaggle/tmp/le-wm && python convert_tworoom.py 2>&1 | grep -v arn | tail -12; ls -la /kaggle/working/stablewm/checkpoints/tworoom/")
sh("cat /kaggle/tmp/le-wm/config/eval/solver/cem.yaml 2>/dev/null | head -30")
for pol in ["quentinll/lewm-tworooms"]:   # the current loader (swm.wm.utils.load_pretrained) takes a HF repo id or a folder with weights.pt + config.json; no conversion needed
    tag = pol.replace("/", "_")
    sh(f"cd /kaggle/tmp/le-wm && timeout 5000 python eval.py --config-name=tworoom.yaml policy={pol} world.max_episode_steps=60 output.filename=/kaggle/working/{tag}_results.txt 2>&1 | grep -vE 'arn' | tail -40", timeout=5200)
    sh(f"cat /kaggle/working/{tag}_results.txt 2>/dev/null | tail -30")
print(f"\nTOTAL {time.time()-t0:.0f}s")
