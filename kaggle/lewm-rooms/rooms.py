import subprocess, sys
r = subprocess.run([sys.executable, "/kaggle/input/datasets/hassanakaou/lewm-rooms-lib/rooms_run.py"], capture_output=True, text=True)
print(r.stdout[-6000:], r.stderr[-3000:])
if r.returncode != 0: raise SystemExit("FAILED")
