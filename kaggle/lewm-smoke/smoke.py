import subprocess, sys, os, time, json
t0 = time.time()
def sh(cmd, timeout=1500):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    print((r.stdout + r.stderr)[-4000:], flush=True)
    return r.returncode
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("python --version; df -h /kaggle/working | tail -1; free -g | head -2; nproc")
sh("git clone --depth 1 https://github.com/lucas-maes/le-wm.git /kaggle/working/le-wm && ls /kaggle/working/le-wm && ls /kaggle/working/le-wm/config/train /kaggle/working/le-wm/config/train/data")
sh("pip install -q 'stable-worldmodel[train,env]' 2>&1 | tail -3", timeout=1700)
sh("python -c \"import stable_worldmodel, stable_pretraining; print('swm ok')\"")
sh("pip show stable-worldmodel | head -3")
sh("python - <<'PY'\nfrom huggingface_hub import HfApi\napi=HfApi()\nfor repo in ['quentinll/lewm-pusht','quentinll/lewm-tworooms','quentinll/lewm-cube','quentinll/lewm-reacher']:\n    for rt in ['model','dataset']:\n        try:\n            info=api.repo_info(repo, repo_type=rt, files_metadata=True)\n            tot=sum((s.size or 0) for s in info.siblings)\n            print(rt, repo, round(tot/1e9,2), 'GB', [ (s.rfilename, round((s.size or 0)/1e9,2)) for s in info.siblings][:8])\n        except Exception as e:\n            print(rt, repo, 'ERR', str(e)[:80])\nPY")
sh("cat /kaggle/working/le-wm/config/train/lewm.yaml | head -60")
print(f"\nTOTAL {time.time()-t0:.0f}s")
