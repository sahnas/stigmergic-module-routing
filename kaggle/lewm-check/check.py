import os, glob, subprocess
print(subprocess.run("ls -la /kaggle/input/lewm-data-tworoom/ /kaggle/input/lewm-data-tworoom/* 2>&1 | head -20", shell=True, capture_output=True, text=True).stdout)
for f in glob.glob("/kaggle/input/lewm-data-tworoom/**/*.h5", recursive=True):
    print(f, round(os.path.getsize(f)/1e9, 3), "GB")
    try:
        import h5py
        h = h5py.File(f, "r")
        for k in list(h.keys())[:12]:
            print("  ", k, getattr(h[k], "shape", None), getattr(h[k], "dtype", None))
    except Exception as e:
        print("  h5 ERR", str(e)[:200])
