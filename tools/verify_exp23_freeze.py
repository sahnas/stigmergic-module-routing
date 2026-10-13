"""Fail if any frozen experiment-23 file differs from the preregistered bytes."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "kaggle/lewm-objective/freeze.json").read_text())
for name, expected in manifest["files"].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"Changed frozen file: {name}; record an amendment before submitting")
print(json.dumps(dict(pass_=True, files=len(manifest["files"]), protocol=manifest["protocol"])))
