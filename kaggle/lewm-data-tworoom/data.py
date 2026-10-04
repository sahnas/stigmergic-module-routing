import subprocess, time, os, glob
t0 = time.time()
def sh(cmd, timeout=3000):
    print(f"\n$ {cmd}  [t={time.time()-t0:.0f}s]", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    print((r.stdout + r.stderr)[-3000:], flush=True)
    return r.returncode
sh("apt-get install -y -qq zstd > /dev/null 2>&1; which zstd; df -h /kaggle/working | tail -1")
from huggingface_hub import hf_hub_download
p = hf_hub_download("quentinll/lewm-tworooms", "tworoom.tar.zst", repo_type="dataset", local_dir="/kaggle/working/dl")
print("archive:", p, round(os.path.getsize(p)/1e9, 2), "GB", f"[t={time.time()-t0:.0f}s]", flush=True)
os.makedirs("/kaggle/working/stablewm", exist_ok=True)
sh(f"cd /kaggle/working/stablewm && tar --zstd -xf {p} && ls -la /kaggle/working/stablewm && find /kaggle/working/stablewm -maxdepth 2 -type f -exec ls -la {{}} \\;", timeout=2400)
sh(f"rm -f {p}; rm -rf /kaggle/working/dl; df -h /kaggle/working | tail -1")
# si l'archive contient un sous-dossier, remonter les .h5 a la racine de stablewm
for f in glob.glob("/kaggle/working/stablewm/**/*.h5", recursive=True):
    dst = os.path.join("/kaggle/working/stablewm", os.path.basename(f))
    if f != dst:
        os.replace(f, dst); print("deplace", f, "->", dst)
sh("ls -la /kaggle/working/stablewm; python -c \"import h5py,glob\nfor f in glob.glob('/kaggle/working/stablewm/*.h5'):\n    h=h5py.File(f,'r'); print(f, list(h.keys())[:10]); [print('  ',k, h[k].shape, h[k].dtype) for k in list(h.keys())[:10]]\"")
print(f"\nTOTAL {time.time()-t0:.0f}s")
