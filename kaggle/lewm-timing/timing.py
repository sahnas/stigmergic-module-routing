import subprocess, time, os, glob, sys
t0 = time.time()
def sh(cmd, timeout=3000):
    print(f"\n$ {cmd}  [t={time.time()-t0:.0f}s]", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    print((r.stdout + r.stderr)[-6000:], flush=True)
    return r.returncode
sh("ls -la /kaggle/input/ /kaggle/input/*/ /kaggle/input/*/*/ 2>&1 | head -30")
cands = glob.glob("/kaggle/input/**/tworoom.h5", recursive=True)
print("tworoom.h5 trouve :", cands, [round(os.path.getsize(c)/1e9, 2) for c in cands], flush=True)
if not cands or os.path.getsize(cands[0]) < 1e9:
    print("DONNEES ABSENTES, arret avant toute installation GPU"); sys.exit(0)
os.environ["WANDB_MODE"] = "disabled"
os.environ["STABLEWM_HOME"] = "/kaggle/working/stablewm"
os.makedirs("/kaggle/working/stablewm/datasets", exist_ok=True)
os.symlink(cands[0], "/kaggle/working/stablewm/datasets/tworoom.h5")  # la version actuelle de stable-worldmodel cherche <STABLEWM_HOME>/datasets/<name>
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("apt-get install -y -qq swig zstd > /dev/null 2>&1; git clone --depth 1 https://github.com/lucas-maes/le-wm.git /kaggle/tmp/le-wm > /dev/null 2>&1; pip install -q 'stable-worldmodel[train,env]' 2>&1 | grep -ciE 'error'", timeout=2400)
sh("python -c \"import stable_worldmodel.data.formats as f, stable_worldmodel.data.format as g; print('formats avant:', g.list_formats())\"; python -c 'import hdf5plugin' 2>&1 | tail -1; pip install -q hdf5plugin 2>&1 | tail -1; python -c \"import stable_worldmodel.data.formats as f, stable_worldmodel.data.format as g; print('formats apres:', g.list_formats())\"; pip show stable-worldmodel | head -2")
sh("ls -la /kaggle/working/stablewm/datasets/; cat /kaggle/tmp/le-wm/config/train/data/tworoom.yaml")
sh("cd /kaggle/tmp/le-wm && timeout 2000 python train.py data=tworoom trainer.max_epochs=1 +trainer.limit_train_batches=60 +trainer.limit_val_batches=5 trainer.precision=16-mixed trainer.devices=1 2>&1 | grep -vE 'arn' | tail -80", timeout=2100)
sh("nvidia-smi --query-gpu=memory.used --format=csv; ls -la /kaggle/working/stablewm; find /kaggle/working -name '*.ckpt' -exec ls -la {} \\;")
print(f"\nTOTAL {time.time()-t0:.0f}s")
