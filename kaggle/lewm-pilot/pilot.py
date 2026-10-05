import subprocess, time, os, glob, sys, json
t0 = time.time()
def run(cmd, timeout=40000):
    print(f"\n$ {cmd}  [t={time.time()-t0:.0f}s]", flush=True)
    r = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr; print(out[:2000] + ('\n...\n' + out[-6000:] if len(out) > 8000 else out[2000:]), flush=True)
    if r.returncode != 0: raise SystemExit(f"STEP FAILED (exit {r.returncode}): {cmd}")
cache = glob.glob("/kaggle/input/**/emb_tworoom.npy", recursive=True)
if not cache: raise SystemExit("CACHE MISSING")
os.environ["CACHE_DIR"] = os.path.dirname(cache[0]); os.environ["STABLEWM_HOME"] = "/kaggle/working/stablewm"; os.environ["WANDB_MODE"] = "disabled"
MODE = os.environ.get("PILOT_MODE", "pilot"); os.environ["PILOT_MODE"] = MODE     # dev: one short seed to check the thresholds; pilot: the preregistered 5 seeds
run("pip install 'stable-worldmodel[train,format]==0.1.1' 'transformers<5' 2>&1 | tail -1; pip freeze | grep -iE 'stable-worldmodel|^torch==|transformers' ")
run("cp $(find /kaggle/input -name pilot_lib.py | head -1) /kaggle/working/pilot_lib.py; sha256sum /kaggle/working/pilot_lib.py")
driver = r'''
import os, json, time, sys, numpy as np, torch, torch.nn as nn, hydra
from huggingface_hub import hf_hub_download
sys.path.insert(0, "/kaggle/working"); import pilot_lib as P
D = os.environ["CACHE_DIR"]; MODE = os.environ["PILOT_MODE"]
emb = np.load(f"{D}/emb_tworoom.npy"); act = np.load(f"{D}/action.npy"); ep_len = np.load(f"{D}/ep_len.npy"); ep_off = np.load(f"{D}/ep_offset.npy")
cfg = json.load(open(hf_hub_download("quentinll/lewm-tworooms", "config.json")))
class LeWMPred(nn.Module):
    def __init__(self):
        super().__init__(); self.action_encoder = hydra.utils.instantiate(cfg["action_encoder"]); self.predictor = hydra.utils.instantiate(cfg["predictor"]); self.pred_proj = hydra.utils.instantiate(cfg["pred_proj"])
    def forward(self, h, a):
        p = self.predictor(h, self.action_encoder(a)); B, T, Dd = p.shape
        return self.pred_proj(p.reshape(B * T, Dd)).reshape(B, T, -1)[:, -1]
make = lambda: LeWMPred().cuda()
if MODE == "dev":
    seeds, kw = [100], dict(n1=1500, n2=1200, n3=400, log_every=100)
else:
    seeds, kw = [0, 1, 2, 3, 4], dict(n1=1500, n2=1500, n3=500, log_every=100)
with open(f"/kaggle/working/pilot_{MODE}.jsonl", "a") as f:
    for s in seeds:
        data = P.Data(emb, act, ep_len, ep_off, seed=s).to("cuda")
        print(f"seed {s}: train episodes {len(data.train_eps)}, test episodes {len(data.test_eps)}", flush=True)
        res = P.run_pilot(data, make, seed=s, **kw)
        f.write(json.dumps(res) + "\n"); f.flush()
'''
open("/kaggle/working/driver.py", "w").write(driver)
run("python /kaggle/working/driver.py 2>&1 | grep -v Warning")
run("rm -f /kaggle/working/driver.py; ls -la /kaggle/working/*.jsonl")
print(f"\nTOTAL {time.time()-t0:.0f}s")
